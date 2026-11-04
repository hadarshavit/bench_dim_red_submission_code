from pathlib import Path
import submitit
from experiment_settings import ML_CLUSTER_TUNING_METRICS, ML_RED_DIMS, method_names, ml_cluster_tuning_config_path, ml_cluster_tuning_result_path, ml_method_allowed_for_config, ml_method_allowed_for_tuning, ml_tuned_evaluation_task_configs
import os
import pickle
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics.cluster import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    homogeneity_completeness_v_measure,
)
from smac import HyperparameterOptimizationFacade, Scenario

from methods_cs import get_configspace, make_method
from ml_reduce import load_ml_data
from experiment_settings import (
    ml_cluster_tuning_config_path,
    ml_cluster_tuning_result_path,
    ml_method_allowed_for_config,
    ml_tuning_task_configs,
)

TUNING_METRICS = {"homogeneity": 0, "completeness": 1}
TUNING_N_TRIALS = 30
TUNING_WORKERS = 24


def tuning_workers() -> int:
    try:
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", TUNING_WORKERS))
    except ValueError:
        cpus = TUNING_WORKERS
    return max(1, min(TUNING_WORKERS, cpus))


def cluster_scores(data, targets):
    labels = HDBSCAN().fit_predict(data)
    return (
        adjusted_rand_score(targets, labels),
        adjusted_mutual_info_score(targets, labels),
        homogeneity_completeness_v_measure(targets, labels),
        fowlkes_mallows_score(targets, labels),
    )


def score_config(task_id, method_name, red_dim, metric, config):
    try:
        data, targets, train_indices, test_indices = load_ml_data(task_id, 0)
        reducer = make_method(method_name, red_dim, config, ml=True)
        reducer.fit(data[train_indices, :], targets.iloc[train_indices])
        reduced = reducer.transform(data)
        hcv = cluster_scores(
            reduced[test_indices], targets.iloc[test_indices]
        )[2]
        return float(hcv[TUNING_METRICS[metric]])
    except Exception:
        return 0.0


def tune_config(method_name, red_dim, metric):
    if metric not in TUNING_METRICS:
        raise ValueError(f"Unknown clustering tuning metric: {metric}")
    output = ml_cluster_tuning_config_path(method_name, red_dim, metric)
    if os.path.exists(output):
        return

    configspace = get_configspace(method_name, red_dim, red_dim, 100, ml=True)
    if len(configspace) == 0:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "wb") as file:
            pickle.dump({}, file)
        return

    task_ids = [
        task_id
        for task_id, _, n_features, n_instances in ml_tuning_task_configs()
        if n_features > red_dim
        and ml_method_allowed_for_config(method_name, n_instances, red_dim)
    ]

    def train(config, seed):
        candidate = dict(config)
        with ProcessPoolExecutor(max_workers=tuning_workers()) as executor:
            scores = list(executor.map(
                score_config,
                task_ids,
                [method_name] * len(task_ids),
                [red_dim] * len(task_ids),
                [metric] * len(task_ids),
                [candidate] * len(task_ids),
            ))
        return 1.0 - float(np.mean(scores))

    with tempfile.TemporaryDirectory(prefix="ml_cluster_tuning_") as smac_tmp:
        scenario = Scenario(
            configspace=configspace,
            output_directory=smac_tmp,
            deterministic=True,
            n_trials=TUNING_N_TRIALS,
            seed=0,
        )
        incumbent = HyperparameterOptimizationFacade(
            scenario=scenario,
            target_function=train,
            overwrite=True,
        ).optimize()

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as file:
        pickle.dump(dict(incumbent), file)


def execute(task_id, cv, method_name, red_dim, metric):
    output = ml_cluster_tuning_result_path(
        task_id, cv, method_name, red_dim, metric
    )
    if os.path.exists(output):
        return
    with open(
        ml_cluster_tuning_config_path(method_name, red_dim, metric), "rb"
    ) as file:
        reducer_config = pickle.load(file)

    data, targets, train_indices, test_indices = load_ml_data(task_id, cv)
    reducer = make_method(method_name, red_dim, reducer_config, ml=True)
    reducer.fit(data[train_indices, :], targets.iloc[train_indices])
    reduced = reducer.transform(data)
    ari, ami, hcv, fms = cluster_scores(
        reduced[test_indices], targets.iloc[test_indices]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as file:
        pickle.dump([ari, ami, hcv, fms], file)


if __name__ == "__main__":
    executor = submitit.AutoExecutor("logs/ml_cluster_tuning", cluster="slurm")
    executor.update_parameters(timeout_min=24 * 60, cpus_per_task=1, mem_gb=2.8)
    with executor.batch():
        (method_names_, _) = method_names()
        for method_name in method_names_:
            for red_dim in ML_RED_DIMS:
                if not ml_method_allowed_for_tuning(method_name, red_dim):
                    continue
                for metric in ML_CLUSTER_TUNING_METRICS:
                    if not os.path.exists(ml_cluster_tuning_config_path(method_name, red_dim, metric)):
                        executor.submit(tune_config, method_name, red_dim, metric)
        (method_names_, _) = method_names()
        for (task_id, folds, n_features, n_instances) in ml_tuned_evaluation_task_configs():
            for cv in range(folds):
                for method_name in method_names_:
                    for red_dim in ML_RED_DIMS:
                        if n_features <= red_dim or not ml_method_allowed_for_config(method_name, n_instances, red_dim) or (not ml_method_allowed_for_tuning(method_name, red_dim)):
                            continue
                        for metric in ML_CLUSTER_TUNING_METRICS:
                            config = ml_cluster_tuning_config_path(method_name, red_dim, metric)
                            if not os.path.exists(ml_cluster_tuning_result_path(task_id, cv, method_name, red_dim, metric)) and os.path.exists(config):
                                executor.submit(execute, task_id, cv, method_name, red_dim, metric)
