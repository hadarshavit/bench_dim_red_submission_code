import fcntl
import os
import pickle
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import submitit


REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

from aggregation.common import aggregation_methods
from experiment_settings import method_allowed_for_config


RESULTS_DIR = Path("results")
DB_PATH = RESULTS_DIR / "aggregation" / "bop.duck"
LOG_DIR = Path("logs/logs_bop_aggregation")
OPTIMA_PATH = REPO_DIR / "analyse" / "optima.pkl"
SAMPLES = (100, 200, 500, 1000)
RED_DIMS = (5, 10, 20, 40)
INPUT_DIMS = (20, 40, 60, 120, 240, 500, 1000)
FIDS = range(1, 25)
INSTANCES = range(31)
METHODS, WEIGHTED_METHODS = aggregation_methods()


def read_many(directory: Path, names: set[str]):
    for name in sorted(names):
        path = directory / name
        if path.is_file():
            yield name, path.read_bytes()


def aggregate_directory(
    sample: int, input_dim: int, fid: int, red_dims: list[int]
) -> pd.DataFrame | None:
    with OPTIMA_PATH.open("rb") as file:
        optima = pickle.load(file)

    directory = RESULTS_DIR / str(sample) / "bop_reduced" / str(fid) / str(input_dim)
    metadata = {}
    expected: dict[tuple[int, str], int] = {}
    for red_dim in red_dims:
        for method_name in METHODS:
            if not method_allowed_for_config(method_name, sample, red_dim):
                continue
            key = (red_dim, method_name)
            expected[key] = len(INSTANCES)
            weight = method_name in WEIGHTED_METHODS
            for instance in INSTANCES:
                name = f"{instance}_{method_name}_{red_dim}_{weight}.pkl"
                metadata[name] = (key, instance)

    curves: dict[tuple[int, str], list[np.ndarray]] = defaultdict(list)
    errors = []
    found = set()
    for name, data in read_many(directory, set(metadata)):
        found.add(name)
        key, instance = metadata[name]
        try:
            result = pickle.loads(data)
            curve = np.asarray(result[2]).reshape(-1)
            curves[key].append(
                np.minimum.accumulate(curve) - optima[(fid, instance, input_dim)]
            )
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    missing = sorted(set(metadata) - found)
    if missing or errors:
        print(
            f"{directory}: missing={len(missing)}, unreadable={len(errors)}",
            flush=True,
        )
        for message in [*missing[:10], *errors[:10]]:
            print(f"  {message}", flush=True)

    complete_frames = []
    for red_dim in red_dims:
        frames = []
        incomplete = []
        for method_name in METHODS:
            key = (red_dim, method_name)
            if key not in expected:
                continue
            values = curves[key]
            if len(values) != expected[key] or len({len(value) for value in values}) != 1:
                incomplete.append(f"{method_name} ({len(values)}/{expected[key]})")
                continue
            mean_curve = np.mean(np.stack(values), axis=0)
            frames.append(pd.DataFrame({
                "index": np.arange(len(mean_curve)),
                0: mean_curve,
                "method_name": method_name,
                "instance": 30,
                "fid": fid,
                "inpt_dim": input_dim,
                "red_dim": red_dim,
                "sample": sample,
            }))

        if incomplete:
            print(
                f"Skipping sample={sample}, input_dim={input_dim}, red_dim={red_dim}, fid={fid}; "
                f"incomplete methods: {', '.join(incomplete)}",
                flush=True,
            )
            continue

        result = pd.concat(frames, ignore_index=True)
        result["rank"] = result.groupby("index")[0].rank(method="min")
        result = result.rename(columns={"index": "iteration", 0: "regret"})
        result = result.drop(columns="instance")
        result["n_instances"] = len(INSTANCES)
        complete_frames.append(result)

    if not complete_frames:
        return None
    return pd.concat(complete_frames, ignore_index=True)


def initialize_database(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bop_results(
            sample INTEGER,
            fid INTEGER,
            inpt_dim INTEGER,
            red_dim INTEGER,
            method_name VARCHAR,
            iteration INTEGER,
            regret DOUBLE,
            rank DOUBLE,
            n_instances INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bop_completed(
            sample INTEGER,
            fid INTEGER,
            inpt_dim INTEGER,
            red_dim INTEGER,
            completed_at TIMESTAMP,
            PRIMARY KEY(sample, fid, inpt_dim, red_dim)
        )
    """)


def completed_configs(conn: duckdb.DuckDBPyConnection) -> set[tuple[int, int, int, int]]:
    return {
        tuple(map(int, row))
        for row in conn.execute(
            "SELECT sample, fid, inpt_dim, red_dim FROM bop_completed"
        ).fetchall()
    }


def write_batch(conn: duckdb.DuckDBPyConnection, frame: pd.DataFrame) -> int:
    conn.register("bop_batch", frame)
    try:
        conn.execute("BEGIN")
        conn.execute("""
            DELETE FROM bop_results
            USING (
                SELECT DISTINCT sample, fid, inpt_dim, red_dim FROM bop_batch
            ) AS batch
            WHERE bop_results.sample = batch.sample
              AND bop_results.fid = batch.fid
              AND bop_results.inpt_dim = batch.inpt_dim
              AND bop_results.red_dim = batch.red_dim
        """)
        conn.execute("""
            INSERT INTO bop_results
            SELECT sample, fid, inpt_dim, red_dim, method_name,
                   iteration, regret, rank, n_instances
            FROM bop_batch
        """)
        conn.execute("""
            INSERT OR REPLACE INTO bop_completed
            SELECT DISTINCT sample, fid, inpt_dim, red_dim, current_timestamp
            FROM bop_batch
        """)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.unregister("bop_batch")
    return len(frame)


def pending_jobs(completed: set[tuple[int, int, int, int]]):
    jobs = []
    for sample in SAMPLES:
        for input_dim in INPUT_DIMS:
            for fid in FIDS:
                red_dims = [
                    red_dim
                    for red_dim in RED_DIMS
                    if input_dim > red_dim
                    and (sample, fid, input_dim, red_dim) not in completed
                ]
                if red_dims:
                    jobs.append((sample, input_dim, fid, red_dims))
    return jobs


def run_aggregation(db_path: str = str(DB_PATH)) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another BOP aggregation is writing {path}") from exc

        conn = duckdb.connect(str(path))
        try:
            initialize_database(conn)
            jobs = pending_jobs(completed_configs(conn))
            if not jobs:
                print("All BOP configurations are already aggregated.")
                return

            workers = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
            written_rows = 0
            completed_directories = 0
            failed_directories = 0
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(aggregate_directory, *job): job for job in jobs
                }
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        frame = future.result()
                    except Exception as exc:
                        print(f"Failed directory {job[:3]}: {type(exc).__name__}: {exc}", flush=True)
                        failed_directories += 1
                        continue
                    if frame is not None:
                        written_rows += write_batch(conn, frame)
                    completed_directories += 1
                    if completed_directories % 10 == 0:
                        print(
                            f"Processed {completed_directories}/{len(jobs)} directories; "
                            f"wrote {written_rows} rows",
                            flush=True,
                        )
            if failed_directories:
                raise RuntimeError(f"Failed to aggregate {failed_directories}/{len(jobs)} directories")
            conn.execute("CHECKPOINT")
            print(f"Finished {path}: wrote {written_rows} rows", flush=True)
        finally:
            conn.close()
            fcntl.flock(lock, fcntl.LOCK_UN)


def main() -> None:
    executor = submitit.AutoExecutor(LOG_DIR, cluster="slurm")
    executor.update_parameters(
        timeout_min=1 * 60,
        cpus_per_task=24,
        mem_gb=32,
        tasks_per_node=1,
        slurm_job_name="aggregate_bop",
        slurm_setup=[


            f"export PYTHONPATH={REPO_DIR}",
            "export OMP_NUM_THREADS=1",
            "export OPENBLAS_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            "export BLIS_NUM_THREADS=1",

        ],
    )
    job = executor.submit(run_aggregation, str(DB_PATH))
    print(f"Submitted BOP aggregation job {job.job_id}; database: {DB_PATH}")


if __name__ == "__main__":

    from aggregation.bop import main

    main()
