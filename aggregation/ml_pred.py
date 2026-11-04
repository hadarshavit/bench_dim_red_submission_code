import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aggregation.bop import read_many
from aggregation.common import RESULTS_DIR, run_parallel_database, submit, aggregation_methods
from experiment_settings import (
    ML_ALGORITHMS,
    ML_RED_DIMS,
    ml_evaluation_task_configs,
    ml_method_allowed_for_config,
)


DB_PATH = RESULTS_DIR / "aggregation" / "ml_pred.duck"
TABLE = "ml_pred_results"
KEYS = ("task_id", "cv", "method", "n_components", "algo")
COLUMNS = (
    "task_id", "cv", "method", "n_components", "n_samples", "n_features",
    "algo", "acc", "f1", "bal_acc",
)
SCHEMA = """
    task_id INTEGER, cv INTEGER, method VARCHAR, n_components INTEGER,
    n_samples INTEGER, n_features INTEGER, algo VARCHAR,
    acc DOUBLE, f1 DOUBLE, bal_acc DOUBLE
"""
METHODS = aggregation_methods(ml=True)[0]


def aggregate_task(
    task_id: int,
    n_features: int,
    n_samples: int,
    configs: list[tuple[int, str, int, str]],
) -> pd.DataFrame | None:
    metadata = {
        f"{task_id}_{cv}_{method}_{red_dim}_{algo}_mlpred.pkl":
            (cv, method, red_dim, algo)
        for cv, method, red_dim, algo in configs
    }
    rows = []
    unreadable = []
    for name, data in read_many(RESULTS_DIR / "ml_data_reduced", set(metadata)):
        cv, method, red_dim, algo = metadata[name]
        try:
            acc, f1, bal_acc, *_ = pickle.loads(data)
            rows.append((
                task_id, cv, method, red_dim, n_samples, n_features, algo,
                float(acc), float(f1), float(bal_acc),
            ))
        except Exception as exc:
            unreadable.append(f"{name}: {type(exc).__name__}: {exc}")
    if len(rows) != len(metadata):
        print(
            f"ML prediction task {task_id}: expected={len(metadata)}, found={len(rows)}, "
            f"unreadable={len(unreadable)}",
            flush=True,
        )
        for message in unreadable:
            print(f"  unreadable: {message}", flush=True)
    return pd.DataFrame(rows, columns=COLUMNS) if rows else None


def pending_jobs(completed: set[tuple]):
    for task_id, folds, n_features, n_samples in ml_evaluation_task_configs():
        configs = []
        for cv in range(folds):
            for method in METHODS:
                for red_dim in ML_RED_DIMS:
                    if n_features <= red_dim or not ml_method_allowed_for_config(
                        method, n_samples, red_dim
                    ):
                        continue
                    for algo in ML_ALGORITHMS:
                        if (task_id, cv, method, red_dim, algo) not in completed:
                            configs.append((cv, method, red_dim, algo))
        if configs:
            yield task_id, n_features, n_samples, configs


def run_aggregation(db_path: str = str(DB_PATH)) -> None:
    run_parallel_database(
        db_path, TABLE, SCHEMA, COLUMNS, KEYS, pending_jobs, aggregate_task
    )


def main() -> None:
    submit(run_aggregation, DB_PATH, "aggregate_ml_pred")


if __name__ == "__main__":
    from aggregation.ml_pred import main

    main()
