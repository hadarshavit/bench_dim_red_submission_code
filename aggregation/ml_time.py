import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aggregation.bop import read_many
from aggregation.common import RESULTS_DIR, run_parallel_database, submit, aggregation_methods
from experiment_settings import (
    ML_RED_DIMS,
    ml_evaluation_task_configs,
    ml_method_allowed_for_config,
)


import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DB_PATH = RESULTS_DIR / "aggregation" / "ml_time.duck"
TABLE = "ml_time_results"
FIGURE_TASK_IDS = {9981, 9985}
KEYS = ("task_id", "cv", "method", "n_components")
COLUMNS = (
    "task_id", "cv", "method", "n_components", "n_samples", "n_features",
    "fit_time", "fit_cpu_time", "transform_time", "transform_cpu_time",
)
SCHEMA = """
    task_id INTEGER, cv INTEGER, method VARCHAR, n_components INTEGER,
    n_samples INTEGER, n_features INTEGER, fit_time DOUBLE,
    fit_cpu_time DOUBLE, transform_time DOUBLE, transform_cpu_time DOUBLE
"""
METHODS = aggregation_methods(ml=True)[0]


def aggregate_task(task_id, n_features, n_samples, configs):
    metadata = {
        f"{task_id}_{cv}_{method}_{red_dim}_time.pkl": (cv, method, red_dim)
        for cv, method, red_dim in configs
    }
    rows = []
    unreadable = []
    for name, data in read_many(RESULTS_DIR / "ml_data_reduced", set(metadata)):
        cv, method, red_dim = metadata[name]
        try:
            (
                start_fit,
                end_fit,
                start_cpu_fit,
                end_cpu_fit,
                start_transform,
                end_transform,
                start_cpu_transform,
                end_cpu_transform,
            ) = pickle.loads(data)
            rows.append((
                task_id, cv, method, red_dim, n_samples, n_features,
                end_fit - start_fit, end_cpu_fit - start_cpu_fit,
                end_transform - start_transform,
                end_cpu_transform - start_cpu_transform,
            ))
        except Exception as exc:
            unreadable.append(f"{name}: {type(exc).__name__}: {exc}")
    if len(rows) != len(metadata):
        print(
            f"ML timing task {task_id}: expected={len(metadata)}, "
            f"found={len(rows)}, unreadable={len(unreadable)}",
            flush=True,
        )
    return pd.DataFrame(rows, columns=COLUMNS) if rows else None


def pending_jobs(completed):
    for task_id, folds, n_features, n_samples in ml_evaluation_task_configs():
        if task_id not in FIGURE_TASK_IDS:
            continue
        configs = [
            (cv, method, red_dim)
            for cv in range(folds)
            for method in METHODS
            for red_dim in ML_RED_DIMS
            if n_features > red_dim
            and ml_method_allowed_for_config(method, n_samples, red_dim)
            and (task_id, cv, method, red_dim) not in completed
        ]
        if configs:
            yield task_id, n_features, n_samples, configs


def run_aggregation(db_path: str = str(DB_PATH)) -> None:
    run_parallel_database(
        db_path, TABLE, SCHEMA, COLUMNS, KEYS, pending_jobs, aggregate_task
    )


def main() -> None:
    submit(run_aggregation, DB_PATH, "aggregate_ml_time")


if __name__ == "__main__":
    from aggregation.ml_time import main

    main()
