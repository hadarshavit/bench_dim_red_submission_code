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
    INSTANCES,
    REAL_FID_DIMS,
    RED_DIMS,
    SAMPLES,
    method_allowed_for_config,
    real_fids,
)


DB_PATH = RESULTS_DIR / "aggregation" / "bop_real.duck"
TABLE = "bop_real_results"
KEYS = ("sample", "fid", "red_dim")
COLUMNS = (
    "sample", "fid", "red_dim", "method_name", "iteration", "value", "rank",
    "n_instances",
)
SCHEMA = """
    sample INTEGER, fid VARCHAR, red_dim INTEGER, method_name VARCHAR,
    iteration INTEGER, value DOUBLE, rank DOUBLE, n_instances INTEGER
"""
METHODS, WEIGHTED_METHODS = aggregation_methods()


def aggregate_directory(sample: int, fid: str, red_dims: list[int]) -> pd.DataFrame | None:
    directory = RESULTS_DIR / str(sample) / "real" / "bop_reduced" / fid
    metadata = {}
    expected = {}
    for red_dim in red_dims:
        for method_name in METHODS:
            if not method_allowed_for_config(method_name, sample, red_dim):
                continue
            key = red_dim, method_name
            expected[key] = len(INSTANCES)
            weight = method_name in WEIGHTED_METHODS
            for instance in INSTANCES:
                metadata[f"{method_name}_{red_dim}_{instance}_{weight}.pkl"] = key

    curves = defaultdict(list)
    unreadable = []
    found = set()
    for name, data in read_many(directory, set(metadata)):
        found.add(name)
        try:
            result = pickle.loads(data)
            curves[metadata[name]].append(
                np.minimum.accumulate(np.asarray(result[2]).reshape(-1))
            )
        except Exception as exc:
            unreadable.append(f"{name}: {type(exc).__name__}: {exc}")

    if len(found) != len(metadata) or unreadable:
        print(
            f"{directory}: missing={len(set(metadata) - found)}, unreadable={len(unreadable)}",
            flush=True,
        )

    frames = []
    for red_dim in red_dims:
        method_frames = []
        incomplete = []
        for method_name in METHODS:
            key = red_dim, method_name
            if key not in expected:
                continue
            values = curves[key]
            if len(values) != expected[key] or len({len(value) for value in values}) != 1:
                incomplete.append(method_name)
                continue
            mean_curve = np.mean(np.stack(values), axis=0)
            method_frames.append(pd.DataFrame({
                "sample": sample,
                "fid": fid,
                "red_dim": red_dim,
                "method_name": method_name,
                "iteration": np.arange(len(mean_curve)),
                "value": mean_curve,
                "n_instances": len(INSTANCES),
            }))
        if incomplete:
            print(f"Skipping {(sample, fid, red_dim)}; incomplete: {', '.join(incomplete)}")
            continue
        frame = pd.concat(method_frames, ignore_index=True)
        frame["rank"] = frame.groupby("iteration")["value"].rank(method="min")
        frames.append(frame.loc[:, COLUMNS])
    return pd.concat(frames, ignore_index=True) if frames else None


def pending_jobs(completed: set[tuple]):
    for sample in SAMPLES:
        for fid in real_fids():
            red_dims = [
                red_dim for red_dim in RED_DIMS
                if REAL_FID_DIMS[fid] >= red_dim
                and (sample, fid, red_dim) not in completed
            ]
            if red_dims:
                yield sample, fid, red_dims


def run_aggregation(db_path: str = str(DB_PATH)) -> None:
    run_parallel_database(
        db_path, TABLE, SCHEMA, COLUMNS, KEYS, pending_jobs, aggregate_directory
    )


def main() -> None:
    submit(run_aggregation, DB_PATH, "aggregate_bop_real")


if __name__ == "__main__":
    from aggregation.bop_real import main

    main()
