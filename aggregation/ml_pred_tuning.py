import pickle

import pandas as pd

from aggregation.bop import read_many
from aggregation.common import RESULTS_DIR, run_parallel_database, submit, aggregation_methods
from aggregation.ml_pred import COLUMNS, KEYS, SCHEMA
from experiment_settings import (
    ML_ALGORITHMS,
    ML_RED_DIMS,
    ml_method_allowed_for_config,
    ml_method_allowed_for_tuning,
    ml_tuned_evaluation_task_configs,
)


DB_PATH = RESULTS_DIR / "aggregation" / "ml_pred_tuning.duck"
TABLE = "ml_pred_results"
METHODS = aggregation_methods(ml=True)[0]


def aggregate_task(task_id, n_features, n_samples, configs):
    metadata = {
        f"{task_id}_{cv}_{method}_{red_dim}_{algorithm}_mlpred.pkl":
            (cv, method, red_dim, algorithm)
        for cv, method, red_dim, algorithm in configs
    }
    rows = []
    unreadable = []
    directory = RESULTS_DIR / "ml_tuning" / "pred"
    for name, data in read_many(directory, set(metadata)):
        cv, method, red_dim, algorithm = metadata[name]
        try:
            accuracy, f1, balanced_accuracy, *_ = pickle.loads(
                data
            )
            rows.append((
                task_id, cv, method, red_dim, n_samples, n_features, algorithm,
                float(accuracy), float(f1), float(balanced_accuracy),
            ))
        except Exception as exc:
            unreadable.append(f"{name}: {type(exc).__name__}: {exc}")
    if len(rows) != len(metadata):
        print(
            f"Tuned ML prediction task {task_id}: expected={len(metadata)}, "
            f"found={len(rows)}, unreadable={len(unreadable)}",
            flush=True,
        )
    return pd.DataFrame(rows, columns=COLUMNS) if rows else None


def pending_jobs(completed):
    for task_id, folds, n_features, n_samples in ml_tuned_evaluation_task_configs():
        configs = []
        for cv in range(folds):
            for method in METHODS:
                for red_dim in ML_RED_DIMS:
                    if (
                        n_features <= red_dim
                        or not ml_method_allowed_for_config(method, n_samples, red_dim)
                        or not ml_method_allowed_for_tuning(method, red_dim)
                    ):
                        continue
                    for algorithm in ML_ALGORITHMS:
                        key = (task_id, cv, method, red_dim, algorithm)
                        if key not in completed:
                            configs.append((cv, method, red_dim, algorithm))
        if configs:
            yield task_id, n_features, n_samples, configs


def run_aggregation(db_path: str = str(DB_PATH)) -> None:
    run_parallel_database(
        db_path, TABLE, SCHEMA, COLUMNS, KEYS, pending_jobs, aggregate_task
    )


def main() -> None:
    submit(run_aggregation, DB_PATH, "aggregate_ml_pred_tuning")


if __name__ == "__main__":
    from aggregation.ml_pred_tuning import main

    main()
