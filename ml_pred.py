import openml
from methods import methods
import time
import lzma
import pickle
import os
import submitit
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, OrdinalHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import cross_val_score, KFold
from smac import HyperparameterOptimizationFacade, Scenario


def get_rf_cs():
    cs = ConfigurationSpace()

    n_estimators = UniformIntegerHyperparameter(
                    name="n_estimators", lower=16, upper=128, log=True)
    cs.add(n_estimators)
    max_features = UniformFloatHyperparameter(
        name="max_features", lower=0.1, upper=1.0)
    cs.add(max_features)
    min_samples_split = UniformIntegerHyperparameter(
        name="min_samples_split", lower=2, upper=20)
    cs.add(min_samples_split)
    min_samples_leaf = UniformIntegerHyperparameter(
        name="min_samples_leaf", lower=1, upper=20)
    cs.add(min_samples_leaf)
    bootstrap = CategoricalHyperparameter(
        name="bootstrap", choices=[True, False])
    cs.add(bootstrap)

    return cs


def get_nn_cs():
    cs = ConfigurationSpace()

    cs.add([
        UniformIntegerHyperparameter(
            'depth', default_value=3, lower=1, upper=3, log=False
        ),
        UniformIntegerHyperparameter(
            'width', default_value=64, lower=16, upper=1024, log=True
        ),
        UniformFloatHyperparameter(
            'alpha', lower=10**-8, upper=1, default_value=10**-3, log=True
        ),
        UniformFloatHyperparameter(
            'learning_rate_init', lower=10**-5, upper=1, default_value=10**-3, log=True
        )
    ])

    return cs


def get_gb_cs():
    cs = ConfigurationSpace()

    cs.add([
        UniformIntegerHyperparameter(
            'max_depth', lower=6, upper=30, default_value=6, log=True
        ),
        UniformIntegerHyperparameter(
            'max_leaf_nodes', lower=2, upper=64, default_value=32, log=True
        ),
        UniformFloatHyperparameter(
            'learning_rate', lower=2**-10, upper=1, default_value=0.1, log=True
        ),
        UniformFloatHyperparameter(
            'l2_regularization', lower=2**-10, upper=2**10, default_value=0.1, log=True
        )
    ])
    return cs

def reduce(task_id, cv, method_name, n_components, algo):
    task = openml.tasks.get_task(task_id, download_splits=False)
    train_indices, test_indices = task.get_train_test_split_indices(
                repeat=0,
                fold=cv,
                sample=0,
            )
    df = pd.read_parquet(f"benchmark_dim_red/ml_data_reduced/{task_id}_{cv}_{method_name}_{n_components}.parquet.xz")

    if algo == 'rf':
        clf = RandomForestClassifier
    elif algo == 'gb':
        clf = HistGradientBoostingClassifier
    elif algo == 'mlp':
        clf = MLPClassifier

    def train(config, seed):
        if algo == 'rf':
            model = RandomForestClassifier(n_estimators=config["n_estimators"],
                                                max_features=config["max_features"],
                                                min_samples_split=config["min_samples_split"],
                                                min_samples_leaf=config["min_samples_leaf"],
                                                bootstrap=config["bootstrap"],
                                                random_state=42)
        elif algo == 'gb':
            model = HistGradientBoostingClassifier(max_depth=config["max_depth"],
                                                    max_leaf_nodes=config["max_leaf_nodes"],
                                                    learning_rate=config["learning_rate"],
                                                    l2_regularization=config["l2_regularization"],
                                                    random_state=42)
        elif algo == 'mlp':
            model = MLPClassifier(hidden_layer_sizes=(config['width'],)*config['depth'],
                                  alpha=config['alpha'],
                                  learning_rate_init=config['learning_rate_init'],
                                  random_state=42)
        
        return -np.mean(cross_val_score(model, Xtrain, Ytrain, error_score='raise', scoring='accuracy', cv=KFold(n_splits=3, shuffle=True, random_state=42)))

    X = df.values
    Y = task.get_dataset().get_data(target=task.target_name)[1]

    Xtrain = X[train_indices]
    Ytrain = Y[train_indices]
    Xtest = X[test_indices]
    Ytest = Y[test_indices]

    if algo == 'rf':
        cs = get_rf_cs()
    elif algo == 'gb':
        cs = get_gb_cs()
    elif algo == 'mlp':
        cs = get_nn_cs()

    s = Scenario(
        configspace=cs,
        output_directory=f'benchmark_dim_red/ml_data_reduced/smac/{task_id}_{cv}_{method_name}_{n_components}',
        deterministic=True,
        n_trials=30,
        seed=0
    )

    smac = HyperparameterOptimizationFacade(scenario=s, target_function=train, overwrite=True)
    inc = smac.optimize()

    if algo == 'rf':
        clf = RandomForestClassifier(n_estimators=inc["n_estimators"],
                                            max_features=inc["max_features"],
                                            min_samples_split=inc["min_samples_split"],
                                            min_samples_leaf=inc["min_samples_leaf"],
                                            bootstrap=inc["bootstrap"],
                                            random_state=42)
    elif algo == 'gb':
        clf = HistGradientBoostingClassifier(max_depth=inc["max_depth"],
                                                max_leaf_nodes=inc["max_leaf_nodes"],
                                                learning_rate=inc["learning_rate"],
                                                l2_regularization=inc["l2_regularization"],
                                                random_state=42)
    elif algo == 'mlp':
        clf = MLPClassifier(hidden_layer_sizes=(inc['width'],)*inc['depth'],
                            alpha=inc['alpha'],
                            learning_rate_init=inc['learning_rate_init'],
                            random_state=42)
        
    clf.fit(Xtrain, Ytrain)

    Ypred = clf.predict(Xtest)
    acc = accuracy_score(Ytest, Ypred)
    f1 = f1_score(Ytest, Ypred, average='weighted')
    bal_acc = balanced_accuracy_score(Ytest, Ypred)

    with lzma.open(f"benchmark_dim_red/ml_data_reduced/{task_id}_{cv}_{method_name}_{n_components}_{algo}_mlpred.pkl.xz", "wb") as file:
        pickle.dump([acc, f1, bal_acc, clf, inc], file)


def parallel_executor(task_id, cv, algo):
    os.makedirs(
        f"benchmark_dim_red/ml_data_reduced/{task_id}_{cv}", exist_ok=True
    )
    for method_name in methods.keys():
        for n_components in [5, 10, 20, 40]:
            if os.path.exists(f"benchmark_dim_red/ml_data_reduced/{task_id}_{cv}_{method_name}_{n_components}_{algo}_mlpred.pkl.xz"):
                continue
            try:
                reduce(task_id, cv, method_name, n_components, algo)
            except Exception as e:
                print(e)


if __name__ == "__main__":
    executor = submitit.AutoExecutor('benchmark_dim_red/logs_ml', 'slurm')
    executor.update_parameters(
        timeout_min=60*48,
        slurm_partition="PARTITION" ,
        slurm_array_parallelism=1024,
        cpus_per_task=1,
        mem_gb=2.8 * 1, 
        tasks_per_node=1,
        slurm_job_name="m_dim_red",
        slurm_setup=["module -q load GCCcore/.12.2.0", "module -q load Python/3.10.8", 
                    "export PYTHONPATH=/benchmark_dim_red",
                    "export OMP_NUM_THREADS=1",
                    "export OPENBLAS_NUM_THREADS=1",
                    "export MKL_NUM_THREADS=1",
                    "export BLIS_NUM_THREADS=1",
                    "source /venvs/bench_dim_red/bin/activate"],
        slurm_additional_parameters={"account": "ACCOUNT"},
    )


    with executor.collect_jobs():
        suite = openml.study.get_suite(99)
        print(suite)
        cnt = 0
        for task_id in suite.tasks:
            task = openml.tasks.get_task(task_id, download_splits=False)
            data, targets, categorical, attribute_names = task.get_dataset().get_data(target=task.target_name)
            if data.shape[1] <= 20:
                continue    
            for cv in range(1, 10):
                
                for method_name in methods.keys():
                    for n_components in [5, 10, 20, 40]:
                        if data.shape[1] <= n_components:
                            continue
                        if data.shape[0] > 10_000:
                            continue
                        
                        for algo in ['rf', 'gb', 'mlp']:
                            if os.path.exists(f"benchmark_dim_red/ml_data_reduced/{task_id}_{cv}_{method_name}_{n_components}_{algo}_mlpred.pkl.xz"):
                                continue
                            print(f'benchmark_dim_red/ml_data_reduced/{task_id}_{cv}_{method_name}_{n_components}_{algo}_mlpred.pkl.xz')
                            executor.submit(reduce, task_id, cv, method_name, n_components, algo)
                        
    
    executor.submit_progressively(50, 50, 60)


