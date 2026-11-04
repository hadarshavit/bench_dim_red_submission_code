import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aggregation.common import RESULTS_DIR, run_parallel_database, submit
from aggregation.epm import COLUMNS, SCHEMA, read_epm_directory
from experiment_settings import REAL_FID_DIMS, RED_DIMS, SAMPLES, real_fids


DB_PATH = RESULTS_DIR / "aggregation" / "epm_real.duck"
TABLE = "epm_real_results"
KEYS = ("sample", "fid", "inpt_dim", "red_dim")
SCHEMA = SCHEMA.replace("fid INTEGER", "fid VARCHAR")


def aggregate_directory(sample: int, fid: str, red_dims: list[int]) -> pd.DataFrame | None:
    inpt_dim = REAL_FID_DIMS[fid]
    directory = RESULTS_DIR / str(sample) / "real" / "epm" / "res" / fid
    return read_epm_directory(
        directory, sample, fid, inpt_dim, red_dims, real=True
    )


def pending_jobs(completed: set[tuple]):
    for sample in SAMPLES:
        for fid in real_fids():
            inpt_dim = REAL_FID_DIMS[fid]
            red_dims = [
                red_dim for red_dim in RED_DIMS
                if inpt_dim >= red_dim
                and (sample, fid, inpt_dim, red_dim) not in completed
            ]
            if red_dims:
                yield sample, fid, red_dims


def run_aggregation(db_path: str = str(DB_PATH)) -> None:
    run_parallel_database(
        db_path, TABLE, SCHEMA, COLUMNS, KEYS, pending_jobs, aggregate_directory
    )


def main() -> None:
    submit(run_aggregation, DB_PATH, "aggregate_epm_real")


if __name__ == "__main__":
    from aggregation.epm_real import main

    main()
