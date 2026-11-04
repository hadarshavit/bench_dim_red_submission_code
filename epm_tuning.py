from pathlib import Path
import submitit
from experiment_settings import BASE_DIR, CV_FOLDS, INSTANCES, SYNTHETIC_FIDS, epm_tuning_config_path, method_weight, tuning_configs
import os
import pickle
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from ioh import ProblemClass, get_problem
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from smac import HyperparameterOptimizationFacade, Scenario

from methods import with_weight
from methods_cs import get_configspace, make_method
from sampling import create_initial_sample

TUNING_FIDS = range(1, 25)
TUNING_INSTANCE = 100
TUNING_N_TRIALS = 30
TUNING_WORKERS = 24


def tuning_workers() -> int:
    try:
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", TUNING_WORKERS))
    except ValueError:
        cpus = TUNING_WORKERS
    return max(1, min(TUNING_WORKERS, cpus))


def config_path(sample, inpt_dim, method_name, red_dim, weight):
    return f"results/{sample}/epm_tuning/configs/{inpt_dim}/{method_name}_{red_dim}_{weight}.pkl"


def train_reducer(fid, instance, inpt_dim, method_name, red_dim, sample, config):
    f = get_problem(fid=fid, dimension=inpt_dim, instance=instance, problem_class=ProblemClass.BBOB)
    x = create_initial_sample(inpt_dim, sample_type="sobol", n=sample, lower_bound=-5, upper_bound=5, seed=42)
    y = x.apply(lambda u: f(u), axis=1)
    reducer = make_method(method_name, red_dim, config)
    reducer.fit(x, y)
    return reducer


def epm_data(fid, instance, inpt_dim):
    f = get_problem(fid=fid, dimension=inpt_dim, instance=instance, problem_class=ProblemClass.BBOB)
    x = create_initial_sample(inpt_dim, sample_type="sobol", n=min(50 * inpt_dim, 5000), lower_bound=-5, upper_bound=5, seed=1)
    y = x.apply(lambda u: f(u), axis=1).values.reshape(-1, 1)
    scaler = MinMaxScaler((1e-5, 1))
    y_scaled = scaler.fit_transform(y)
    y_scaled[y_scaled < 1e-5] = 1e-5
    return x, np.log10(y_scaled).reshape(-1)


def score_config(fid, inpt_dim, method_name, red_dim, sample, config):
    reducer = train_reducer(fid, TUNING_INSTANCE, inpt_dim, method_name, red_dim, sample, config)
    x_orig, y = epm_data(fid, TUNING_INSTANCE, inpt_dim)
    x = reducer.transform(x_orig)
    model = RandomForestRegressor(n_estimators=64, max_features=0.5, random_state=42)
    return -np.mean(cross_val_score(
        model,
        x,
        y,
        error_score="raise",
        scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
    ))


def tune_config(method_name, inpt_dim, red_dim, sample):
    weight = method_name in with_weight
    out = config_path(sample, inpt_dim, method_name, red_dim, weight)
    if os.path.exists(out):
        return

    cs = get_configspace(method_name, inpt_dim, red_dim, sample)
    if len(cs) == 0:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "wb") as file:
            pickle.dump({}, file)
        return

    def train(config, seed):
        cfg = dict(config)
        with ProcessPoolExecutor(max_workers=tuning_workers()) as executor:
            futures = [
                executor.submit(score_config, fid, inpt_dim, method_name, red_dim, sample, cfg)
                for fid in TUNING_FIDS
            ]
            try:
                losses = [future.result() for future in futures]
            except Exception:
                return 1e12
        return float(np.mean(losses))

    with tempfile.TemporaryDirectory(prefix="epm_tuning_") as smac_tmp:
        scenario = Scenario(configspace=cs, output_directory=smac_tmp, deterministic=True, n_trials=TUNING_N_TRIALS, seed=0)
        inc = HyperparameterOptimizationFacade(scenario=scenario, target_function=train, overwrite=True).optimize()

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as file:
        pickle.dump(dict(inc), file)


def execute(fid, method_name, instance, inpt_dim, red_dim, cv, sample):
    weight = method_name in with_weight
    result = f"results/{sample}/epm_tuning/res/{fid}/{inpt_dim}/{instance}_{method_name}_{red_dim}_{weight}_{cv}.pkl"
    if os.path.exists(result):
        return
    with open(config_path(sample, inpt_dim, method_name, red_dim, weight), "rb") as file:
        config = pickle.load(file)

    reducer = train_reducer(fid, instance, inpt_dim, method_name, red_dim, sample, config)
    x_orig, y = epm_data(fid, instance, inpt_dim)
    x = reducer.transform(x_orig)

    for i, (train_index, test_index) in enumerate(KFold(n_splits=5).split(x)):
        if i == cv:
            break
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    cs = get_rf_configspace()

    def train(config, seed):
        model = RandomForestRegressor(
            n_estimators=config["n_estimators"],
            max_features=config["max_features"],
            min_samples_split=config["min_samples_split"],
            min_samples_leaf=config["min_samples_leaf"],
            bootstrap=config["bootstrap"],
            random_state=42,
        )
        return -np.mean(cross_val_score(model, x_train, y_train, error_score="raise", scoring="neg_mean_squared_error", cv=KFold(n_splits=3, shuffle=True, random_state=42)))

    with tempfile.TemporaryDirectory(prefix="epm_tuning_eval_") as smac_tmp:
        scenario = Scenario(configspace=cs, output_directory=smac_tmp, deterministic=True, n_trials=30, seed=0)
        inc = HyperparameterOptimizationFacade(scenario=scenario, target_function=train, overwrite=True).optimize()

    model = RandomForestRegressor(**dict(inc), random_state=42)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    os.makedirs(os.path.dirname(result), exist_ok=True)
    with open(result, "wb") as file:
        pickle.dump(dict(model=model, inc=inc, reducer_config=config, mse=mean_squared_error(y_test, preds), preds=preds), file)


def get_rf_configspace():
    from ConfigSpace import ConfigurationSpace
    from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter

    cs = ConfigurationSpace()
    cs.add(UniformIntegerHyperparameter("n_estimators", 16, 128, log=True))
    cs.add(UniformFloatHyperparameter("max_features", 0.1, 1.0))
    cs.add(UniformIntegerHyperparameter("min_samples_split", 2, 20))
    cs.add(UniformIntegerHyperparameter("min_samples_leaf", 1, 20))
    cs.add(CategoricalHyperparameter("bootstrap", [True, False]))
    return cs


if __name__ == "__main__":
    executor = submitit.AutoExecutor("logs/epm_tuning", cluster="slurm")
    executor.update_parameters(timeout_min=24 * 60, cpus_per_task=1, mem_gb=2.8)
    with executor.batch():
        for (sample, inpt_dim, method_name, red_dim) in tuning_configs():
            if not os.path.exists(epm_tuning_config_path(sample, inpt_dim, method_name, red_dim)):
                executor.submit(tune_config, method_name, inpt_dim, red_dim, sample)
        for (sample, inpt_dim, method_name, red_dim) in tuning_configs():
            weight = method_weight(method_name)
            config = epm_tuning_config_path(sample, inpt_dim, method_name, red_dim)
            for fid in SYNTHETIC_FIDS:
                for instance in INSTANCES:
                    for cv in CV_FOLDS:
                        result = BASE_DIR / str(sample) / 'epm_tuning' / 'res' / str(fid) / str(inpt_dim) / f'{instance}_{method_name}_{red_dim}_{weight}_{cv}.pkl'
                        if not os.path.exists(result) and os.path.exists(config):
                            executor.submit(execute, fid, method_name, instance, inpt_dim, red_dim, cv, sample)
