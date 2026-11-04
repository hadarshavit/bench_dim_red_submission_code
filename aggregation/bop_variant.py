import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from aggregation.bop import read_many
from aggregation.common import RESULTS_DIR, aggregation_methods
from experiment_settings import INSTANCES, method_allowed_for_config


OPTIMA_PATH = Path(__file__).resolve().parents[1] / "analyse" / "optima.pkl"
COLUMNS = (
    "sample", "fid", "inpt_dim", "red_dim", "method_name", "iteration",
    "regret", "rank", "n_instances",
)
SCHEMA = """
    sample INTEGER, fid INTEGER, inpt_dim INTEGER, red_dim INTEGER,
    method_name VARCHAR, iteration INTEGER, regret DOUBLE, rank DOUBLE,
    n_instances INTEGER
"""
KEYS = ("sample", "fid", "inpt_dim", "red_dim")
METHODS, WEIGHTED_METHODS = aggregation_methods()


def aggregate_bop_directory(
    result_subdir: str,
    sample: int,
    fid: int,
    inpt_dim: int,
    red_dim: int,
) -> pd.DataFrame | None:
    directory = RESULTS_DIR / str(sample) / result_subdir / str(fid) / str(inpt_dim)
    metadata = {}
    expected = {}
    for method in METHODS:
        if not method_allowed_for_config(method, sample, red_dim):
            continue
        expected[method] = len(INSTANCES)
        weight = method in WEIGHTED_METHODS
        for instance in INSTANCES:
            metadata[f"{instance}_{method}_{red_dim}_{weight}.pkl"] = (
                method,
                instance,
            )

    with OPTIMA_PATH.open("rb") as file:
        optima = pickle.load(file)

    curves = defaultdict(list)
    unreadable = []
    found = set()
    for name, data in read_many(directory, set(metadata)):
        found.add(name)
        method, instance = metadata[name]
        try:
            result = pickle.loads(data)
            values = np.asarray(result[2]).reshape(-1)
            curves[method].append(
                np.minimum.accumulate(values) - optima[(fid, instance, inpt_dim)]
            )
        except Exception as exc:
            unreadable.append(f"{name}: {type(exc).__name__}: {exc}")

    if len(found) != len(metadata) or unreadable:
        print(
            f"{directory}: missing={len(set(metadata) - found)}, "
            f"unreadable={len(unreadable)}",
            flush=True,
        )

    frames = []
    incomplete = []
    for method in expected:
        values = curves[method]
        if len(values) != expected[method] or len({len(value) for value in values}) != 1:
            incomplete.append(f"{method} ({len(values)}/{expected[method]})")
            continue
        mean_curve = np.mean(np.stack(values), axis=0)
        frames.append(pd.DataFrame({
            "sample": sample,
            "fid": fid,
            "inpt_dim": inpt_dim,
            "red_dim": red_dim,
            "method_name": method,
            "iteration": np.arange(len(mean_curve)),
            "regret": mean_curve,
            "n_instances": len(INSTANCES),
        }))

    if incomplete:
        print(f"Skipping {(sample, fid, inpt_dim, red_dim)}; incomplete: {', '.join(incomplete)}")
        return None
    if not frames:
        return None
    result = pd.concat(frames, ignore_index=True)
    result["rank"] = result.groupby("iteration")["regret"].rank(method="min")
    return result.loc[:, list(COLUMNS)]
