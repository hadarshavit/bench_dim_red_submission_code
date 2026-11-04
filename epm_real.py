from pathlib import Path
from experiment_settings import BASE_DIR, CV_FOLDS, expected_real_reducers, method_weight, real_data_path
import pandas as pd
import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, OrdinalHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from smac import HyperparameterOptimizationFacade, Scenario
import submitit
from methods import methods, with_weight
import pickle
from sklearn.preprocessing import MinMaxScaler
import os
import tempfile
from ioh import get_problem, ProblemClass
from functools import partial
import torch
from sampling import create_initial_sample
from real_world_benchs.real_world_benchs import *


benchs = {
    "Rover": Rover,
    "SVM": SVM,
    "Mujoco_ant": partial(Mujoco, 'ant'),
    "Mujoco_half_cheetah": partial(Mujoco, 'half_cheetah'),
    "Mujoco_hopper": partial(Mujoco, 'hopper'),
    "Mujoco_humanoid": partial(Mujoco, 'humanoid'),
    "Mujoco_swimmer": partial(Mujoco, 'swimmer'),
    "Mujoco_walker_2d": partial(Mujoco, 'walker_2d'),
    "Mopta": Mopta,
    "LassoReal_diabetes": partial(LassoReal, 'diabetes'),
    "LassoReal_breast_cancer": partial(LassoReal, 'breast_cancer'),
    "LassoReal_leukemia": partial(LassoReal, 'leukemia'),
    "LassoReal_dna": partial(LassoReal, 'dna'),
    "LassoReal_rcv1": partial(LassoReal, 'rcv1'),
    "LassoSynthetic_synt_simple": partial(LassoSynthetic, 'synt_simple'),
    "LassoSynthetic_synt_medium": partial(LassoSynthetic, 'synt_medium'),
    "LassoSynthetic_synt_high": partial(LassoSynthetic, 'synt_high'),
    "LassoSynthetic_synt_hard": partial(LassoSynthetic, 'synt_hard')
}


def execute(fid, method_name, seed, red_dim, sample, cv):
    weight = method_name in with_weight

    bench = benchs[fid]()
    lb, ub = bench.get_bounds()

    if len(lb) < red_dim:
        return

    with open(
        f"results/real_data/{fid}/{seed}_data.pkl",
        "rb",
    ) as file:
        x, y = pickle.load(file)


    method_class = methods[method_name]
    reducer = method_class(n_components=red_dim)
    reducer.load(f"results/{sample}/real/data_reduced/{fid}/{method_name}_{red_dim}_{seed}_{weight}_reducer.pkl")
    x = reducer.transform(x)

    kf = KFold(n_splits=5)
    for i, (train_index, test_index) in enumerate(kf.split(x)):
         if i == cv:
              break

    Xtrain = x[train_index]
    Xtest = x[test_index]
    scaler = MinMaxScaler((1e-5 , 1))
    y = y.reshape(-1, 1)

    Ytrain = scaler.fit_transform(y[train_index])
    Ytrain[Ytrain < 1e-5] = 1e-5
    Ytrain = np.log10(Ytrain).reshape(-1)
    Ytest = scaler.transform(y[test_index])
    Ytest[Ytest < 1e-5] = 1e-5
    Ytest = np.log10(Ytest).reshape(-1)

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

    def train(config, seed):
        model = RandomForestRegressor(n_estimators=config["n_estimators"],
                                            max_features=config["max_features"],
                                            min_samples_split=config["min_samples_split"],
                                            min_samples_leaf=config["min_samples_leaf"],
                                            bootstrap=config["bootstrap"],
                                            random_state=42)

        return -np.mean(cross_val_score(model, Xtrain, Ytrain, error_score='raise', scoring='neg_mean_squared_error', cv=KFold(n_splits=3, shuffle=True, random_state=42)))

    with tempfile.TemporaryDirectory(prefix="epm_real_") as smac_tmp:
        s = Scenario(
            configspace=cs,
            output_directory=smac_tmp,
            deterministic=True,

            n_trials=30,
            seed=0,
        )

        smac = HyperparameterOptimizationFacade(scenario=s, target_function=train, overwrite=True)
        inc = smac.optimize()

    model = RandomForestRegressor(n_estimators=inc["n_estimators"],
                                            max_features=inc["max_features"],
                                            min_samples_split=inc["min_samples_split"],
                                            min_samples_leaf=inc["min_samples_leaf"],
                                            bootstrap=inc["bootstrap"],
                                            random_state=42)
    model.fit(Xtrain, Ytrain)
    preds = model.predict(Xtest)
    mse = mean_squared_error(Ytest, preds)
    Path(f'results/{sample}/real/epm/res/{fid}/{method_name}_{red_dim}_{seed}_{weight}_{cv}.pkl').parent.mkdir(parents=True, exist_ok=True)
    with open(
        f'results/{sample}/real/epm/res/{fid}/{method_name}_{red_dim}_{seed}_{weight}_{cv}.pkl',
        "wb",
    ) as file:
        pickle.dump(dict(
            model=model, inc=inc, mse=mse, preds=preds),
            file)


if __name__ == "__main__":
    executor = submitit.AutoExecutor("logs/epm_real", cluster="slurm")
    executor.update_parameters(timeout_min=24 * 60, cpus_per_task=1, mem_gb=2.8)
    with executor.batch():
        for (sample, fid, method_name, red_dim, seed, reducer) in expected_real_reducers():
            weight = method_weight(method_name)
            for cv in CV_FOLDS:
                result = BASE_DIR / str(sample) / 'real' / 'epm' / 'res' / fid / f'{method_name}_{red_dim}_{seed}_{weight}_{cv}.pkl'
                if not os.path.exists(result) and os.path.exists(reducer) and os.path.exists(real_data_path(fid, seed)):
                    executor.submit(execute, fid, method_name, seed, red_dim, sample, cv)
