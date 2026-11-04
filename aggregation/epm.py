import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aggregation.bop import read_many
from aggregation.common import RESULTS_DIR, run_parallel_database, submit, aggregation_methods
from experiment_settings import (
    CV_FOLDS,
    INPUT_DIMS,
    INSTANCES,
    RED_DIMS,
    SAMPLES,
    SYNTHETIC_FIDS,
    method_allowed_for_config,
)


DB_PATH = RESULTS_DIR / "aggregation" / "epm.duck"
TABLE = "epm_results"
KEYS = ("sample", "fid", "inpt_dim", "red_dim")
COLUMNS = (
    "sample", "fid", "inpt_dim", "red_dim", "instance", "method_name", "weight",
    "cv", "mse",
)
SCHEMA = """
    sample INTEGER, fid INTEGER, inpt_dim INTEGER, red_dim INTEGER,
    instance INTEGER, method_name VARCHAR, weight BOOLEAN, cv INTEGER, mse DOUBLE
"""
METHODS, WEIGHTED_METHODS = aggregation_methods()


def read_epm_directory(
    directory, sample: int, fid, inpt_dim: int, red_dims: list[int], real: bool = False
) -> pd.DataFrame | None:
    metadata = {}
    expected = {}
    for red_dim in red_dims:
        for method_name in METHODS:
            if not method_allowed_for_config(method_name, sample, red_dim):
                continue
            key = red_dim, method_name
            expected[key] = len(INSTANCES) * len(CV_FOLDS)
            weight = method_name in WEIGHTED_METHODS
            for instance in INSTANCES:
                for cv in CV_FOLDS:
                    if real:
                        name = f"{method_name}_{red_dim}_{instance}_{weight}_{cv}.pkl"
                    else:
                        name = f"{instance}_{method_name}_{red_dim}_{weight}_{cv}.pkl"
                    metadata[name] = key, instance, weight, cv

    rows = defaultdict(list)
    unreadable = []
    found = set()
    for name, data in read_many(directory, set(metadata)):
        found.add(name)
        key, instance, weight, cv = metadata[name]
        try:
            result = pickle.loads(data)
            mse = float(np.asarray(result["mse"]).squeeze())
            rows[key].append((
                sample, fid, inpt_dim, key[0], instance, key[1], weight, cv, mse
            ))
        except Exception as exc:
            unreadable.append(f"{name}: {type(exc).__name__}: {exc}")

    if len(found) != len(metadata) or unreadable:
        print(
            f"{directory}: missing={len(set(metadata) - found)}, unreadable={len(unreadable)}",
            flush=True,
        )

    complete = []
    for red_dim in red_dims:
        red_rows = []
        incomplete = []
        for method_name in METHODS:
            key = red_dim, method_name
            if key not in expected:
                continue
            if len(rows[key]) != expected[key]:
                incomplete.append(method_name)
            else:
                red_rows.extend(rows[key])
        if incomplete:
            print(f"Skipping {(sample, fid, inpt_dim, red_dim)}; incomplete: {', '.join(incomplete)}")
        else:
            complete.extend(red_rows)
    return pd.DataFrame(complete, columns=COLUMNS) if complete else None


def aggregate_directory(
    sample: int, fid: int, inpt_dim: int, red_dims: list[int]
) -> pd.DataFrame | None:
    directory = RESULTS_DIR / str(sample) / "epm" / "res" / str(fid) / str(inpt_dim)
    return read_epm_directory(directory, sample, fid, inpt_dim, red_dims)


def pending_jobs(completed: set[tuple]):
    for sample in SAMPLES:
        for fid in SYNTHETIC_FIDS:
            for inpt_dim in INPUT_DIMS:
                red_dims = [
                    red_dim for red_dim in RED_DIMS
                    if inpt_dim > red_dim
                    and (sample, fid, inpt_dim, red_dim) not in completed
                ]
                if red_dims:
                    yield sample, fid, inpt_dim, red_dims


def run_aggregation(db_path: str = str(DB_PATH)) -> None:
    run_parallel_database(
        db_path, TABLE, SCHEMA, COLUMNS, KEYS, pending_jobs, aggregate_directory
    )


def main() -> None:
    submit(run_aggregation, DB_PATH, "aggregate_epm")


if __name__ == "__main__":
    from aggregation.epm import main

    main()
