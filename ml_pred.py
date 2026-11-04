from pathlib import Path
from experiment_settings import ML_ALGORITHMS, ML_RED_DIMS, method_names, ml_evaluation_task_configs, ml_method_allowed_for_config, ml_pred_path, ml_reduced_path
import os

import openml
from methods import methods
import time
import pickle
import submitit
import tempfile
from ml_reduce import read_reduced
from experiment_settings import ml_pred_path
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


def get_classifier_configspace(algo):
    if algo == 'rf':
        return get_rf_cs()
    if algo == 'gb':
        return get_gb_cs()
    if algo == 'mlp':
        return get_nn_cs()
    raise ValueError(f"Unknown ML prediction algorithm: {algo}")


def make_classifier(algo, config):
    if algo == 'rf':
        classifier = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_features=config["max_features"],
            min_samples_split=config["min_samples_split"],
            min_samples_leaf=config["min_samples_leaf"],
            bootstrap=config["bootstrap"],
            random_state=42,
        )
    elif algo == 'gb':
        classifier = HistGradientBoostingClassifier(
            max_depth=config["max_depth"],
            max_leaf_nodes=config["max_leaf_nodes"],
            learning_rate=config["learning_rate"],
            l2_regularization=config["l2_regularization"],
            random_state=42,
        )
    elif algo == 'mlp':
        classifier = MLPClassifier(
            hidden_layer_sizes=(config['width'],) * config['depth'],
            alpha=config['alpha'],
            learning_rate_init=config['learning_rate_init'],
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown ML prediction algorithm: {algo}")
    return make_pipeline(SimpleImputer(strategy="median"), classifier)


def tune_classifier(Xtrain, Ytrain, algo):
    cs = get_classifier_configspace(algo)

    def train(config, seed):
        try:
            model = make_classifier(algo, config)
            score = float(np.mean(cross_val_score(
                model,
                Xtrain,
                Ytrain,
                error_score='raise',
                scoring='accuracy',
                cv=KFold(n_splits=3, shuffle=True, random_state=42),
            )))
        except (FloatingPointError, ValueError):
            return 1.0
        return 1.0 - score if np.isfinite(score) else 1.0

    with tempfile.TemporaryDirectory(prefix="ml_pred_smac_") as output_directory:
        scenario = Scenario(
            configspace=cs,
            output_directory=output_directory,
            deterministic=True,
            n_trials=30,
            seed=0,
        )
        return HyperparameterOptimizationFacade(
            scenario=scenario,
            target_function=train,
            overwrite=True,
        ).optimize()


def reduce(task_id, cv, method_name, n_components, algo):
    result_path = ml_pred_path(task_id, cv, method_name, n_components, algo)
    if result_path.exists():
        return
    result_path.parent.mkdir(parents=True, exist_ok=True)
    task = openml.tasks.get_task(task_id, download_splits=False)
    train_indices, test_indices = task.get_train_test_split_indices(
                repeat=0,
                fold=cv,
                sample=0,
            )
    df = read_reduced(task_id, cv, method_name, n_components)

    X = df.values
    Y = task.get_dataset().get_data(target=task.target_name)[1]

    Xtrain = X[train_indices]
    Ytrain = Y[train_indices]
    Xtest = X[test_indices]
    Ytest = Y[test_indices]

    inc = tune_classifier(Xtrain, Ytrain, algo)
    clf = make_classifier(algo, inc)
    clf.fit(Xtrain, Ytrain)

    Ypred = clf.predict(Xtest)
    acc = accuracy_score(Ytest, Ypred)
    f1 = f1_score(Ytest, Ypred, average='weighted')
    bal_acc = balanced_accuracy_score(Ytest, Ypred)

    temporary_path = result_path.with_name(f".{result_path.name}.{os.getpid()}.tmp")
    try:
        Path(temporary_path).parent.mkdir(parents=True, exist_ok=True)
        with open(temporary_path, "wb") as file:
            pickle.dump([acc, f1, bal_acc, clf, inc], file)
        os.replace(temporary_path, result_path)
    finally:
        temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    executor = submitit.AutoExecutor("logs/ml_pred", cluster="slurm")
    executor.update_parameters(timeout_min=24 * 60, cpus_per_task=1, mem_gb=2.8)
    with executor.batch():
        (method_names_, _) = method_names()
        for (task_id, folds, n_features, n_instances) in ml_evaluation_task_configs():
            for cv in range(folds):
                for method_name in method_names_:
                    for red_dim in ML_RED_DIMS:
                        if n_features <= red_dim or not ml_method_allowed_for_config(method_name, n_instances, red_dim):
                            continue
                        reduced = ml_reduced_path(task_id, cv, method_name, red_dim)
                        for algorithm in ML_ALGORITHMS:
                            if not os.path.exists(ml_pred_path(task_id, cv, method_name, red_dim, algorithm)) and os.path.exists(reduced):
                                executor.submit(reduce, task_id, cv, method_name, red_dim, algorithm)
