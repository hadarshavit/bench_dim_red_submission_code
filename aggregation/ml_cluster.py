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


DB_PATH = RESULTS_DIR / "aggregation" / "ml_cluster.duck"
TABLE = "ml_cluster_results"
KEYS = ("task_id", "cv", "method", "n_components")
COLUMNS = (
    "task_id", "cv", "method", "n_components", "n_samples", "n_features",
    "ari", "ami", "homogeneity", "completeness", "v_measure", "fms",
)
SCHEMA = """
    task_id INTEGER, cv INTEGER, method VARCHAR, n_components INTEGER,
    n_samples INTEGER, n_features INTEGER, ari DOUBLE, ami DOUBLE,
    homogeneity DOUBLE, completeness DOUBLE, v_measure DOUBLE, fms DOUBLE
"""
METHODS = aggregation_methods(ml=True)[0]


def aggregate_task(
    task_id: int,
    n_features: int,
    n_samples: int,
    configs: list[tuple[int, str, int]],
) -> pd.DataFrame | None:
    metadata = {
        f"{task_id}_{cv}_{method}_{red_dim}_cluster.pkl": (cv, method, red_dim)
        for cv, method, red_dim in configs
    }
    rows = []
    unreadable = []
    for name, data in read_many(RESULTS_DIR / "ml_data_reduced", set(metadata)):
        cv, method, red_dim = metadata[name]
        try:
            ari, ami, hcv, fms = pickle.loads(data)
            homogeneity, completeness, v_measure = hcv
            rows.append((
                task_id, cv, method, red_dim, n_samples, n_features,
                float(ari), float(ami), float(homogeneity), float(completeness),
                float(v_measure), float(fms),
            ))
        except Exception as exc:
            unreadable.append(f"{name}: {type(exc).__name__}: {exc}")
    if len(rows) != len(metadata):
        print(
            f"ML clustering task {task_id}: expected={len(metadata)}, found={len(rows)}, "
            f"unreadable={len(unreadable)}",
            flush=True,
        )
    return pd.DataFrame(rows, columns=COLUMNS) if rows else None


def pending_jobs(completed: set[tuple]):
    for task_id, folds, n_features, n_samples in ml_evaluation_task_configs():
        configs = []
        for cv in range(folds):
            for method in METHODS:
                for red_dim in ML_RED_DIMS:
                    if (
                        n_features > red_dim
                        and ml_method_allowed_for_config(method, n_samples, red_dim)
                        and (task_id, cv, method, red_dim) not in completed
                    ):
                        configs.append((cv, method, red_dim))
        if configs:
            yield task_id, n_features, n_samples, configs


def run_aggregation(db_path: str = str(DB_PATH)) -> None:
    run_parallel_database(
        db_path, TABLE, SCHEMA, COLUMNS, KEYS, pending_jobs, aggregate_task
    )


def main() -> None:
    submit(run_aggregation, DB_PATH, "aggregate_ml_cluster")


if __name__ == "__main__":
    from aggregation.ml_cluster import main

    main()
