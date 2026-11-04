from pathlib import Path
import submitit
from experiment_settings import ML_ALGORITHMS, ML_RED_DIMS, method_names, ml_method_allowed_for_config, ml_method_allowed_for_tuning, ml_pred_tuning_config_path, ml_pred_tuning_result_path, ml_tuned_evaluation_task_configs
import os
import pickle
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from smac import HyperparameterOptimizationFacade, Scenario

from methods_cs import get_configspace, make_method
from ml_pred import get_classifier_configspace, make_classifier, tune_classifier
from ml_reduce import load_ml_data
from experiment_settings import (
    ml_pred_tuning_config_path,
    ml_pred_tuning_result_path,
    ml_method_allowed_for_config,
    ml_tuning_task_configs,
)

TUNING_N_TRIALS = 30
TUNING_WORKERS = 24


def tuning_workers() -> int:
    try:
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", TUNING_WORKERS))
    except ValueError:
        cpus = TUNING_WORKERS
    return max(1, min(TUNING_WORKERS, cpus))


def score_config(task_id, method_name, red_dim, algorithm, config):
    try:
        data, targets, train_indices, test_indices = load_ml_data(task_id, 0)
        reducer = make_method(method_name, red_dim, config, ml=True)
        reducer.fit(data[train_indices, :], targets.iloc[train_indices])
        reduced = reducer.transform(data)
        classifier_config = dict(
            get_classifier_configspace(algorithm).get_default_configuration()
        )
        classifier = make_classifier(algorithm, classifier_config)
        classifier.fit(reduced[train_indices], targets.iloc[train_indices])
        predictions = classifier.predict(reduced[test_indices])
        return float(accuracy_score(targets.iloc[test_indices], predictions))
    except Exception:
        return 0.0


def tune_config(method_name, red_dim, algorithm):
    output = ml_pred_tuning_config_path(method_name, red_dim, algorithm)
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
            accuracies = list(executor.map(
                score_config,
                task_ids,
                [method_name] * len(task_ids),
                [red_dim] * len(task_ids),
                [algorithm] * len(task_ids),
                [candidate] * len(task_ids),
            ))
        return 1.0 - float(np.mean(accuracies))

    with tempfile.TemporaryDirectory(prefix="ml_pred_tuning_") as smac_tmp:
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


def execute(task_id, cv, method_name, red_dim, algorithm):
    output = ml_pred_tuning_result_path(
        task_id, cv, method_name, red_dim, algorithm
    )
    if os.path.exists(output):
        return
    with open(
        ml_pred_tuning_config_path(method_name, red_dim, algorithm), "rb"
    ) as file:
        reducer_config = pickle.load(file)

    data, targets, train_indices, test_indices = load_ml_data(task_id, cv)
    reducer = make_method(method_name, red_dim, reducer_config, ml=True)
    reducer.fit(data[train_indices, :], targets.iloc[train_indices])
    reduced = reducer.transform(data)

    incumbent = tune_classifier(
        reduced[train_indices], targets.iloc[train_indices], algorithm
    )
    classifier = make_classifier(algorithm, incumbent)
    classifier.fit(reduced[train_indices], targets.iloc[train_indices])
    predictions = classifier.predict(reduced[test_indices])

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as file:
        pickle.dump(
            [
                accuracy_score(targets.iloc[test_indices], predictions),
                f1_score(targets.iloc[test_indices], predictions, average="weighted"),
                balanced_accuracy_score(targets.iloc[test_indices], predictions),
                classifier,
                incumbent,
            ],
            file,
        )


if __name__ == "__main__":
    executor = submitit.AutoExecutor("logs/ml_pred_tuning", cluster="slurm")
    executor.update_parameters(timeout_min=24 * 60, cpus_per_task=1, mem_gb=2.8)
    with executor.batch():
        (method_names_, _) = method_names()
        for method_name in method_names_:
            for red_dim in ML_RED_DIMS:
                if not ml_method_allowed_for_tuning(method_name, red_dim):
                    continue
                for algorithm in ML_ALGORITHMS:
                    if not os.path.exists(ml_pred_tuning_config_path(method_name, red_dim, algorithm)):
                        executor.submit(tune_config, method_name, red_dim, algorithm)
        (method_names_, _) = method_names()
        for (task_id, folds, n_features, n_instances) in ml_tuned_evaluation_task_configs():
            for cv in range(folds):
                for method_name in method_names_:
                    for red_dim in ML_RED_DIMS:
                        if n_features <= red_dim or not ml_method_allowed_for_config(method_name, n_instances, red_dim) or (not ml_method_allowed_for_tuning(method_name, red_dim)):
                            continue
                        for algorithm in ML_ALGORITHMS:
                            config = ml_pred_tuning_config_path(method_name, red_dim, algorithm)
                            if not os.path.exists(ml_pred_tuning_result_path(task_id, cv, method_name, red_dim, algorithm)) and os.path.exists(config):
                                executor.submit(execute, task_id, cv, method_name, red_dim, algorithm)
