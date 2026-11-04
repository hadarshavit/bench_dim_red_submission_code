import fcntl
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable

import duckdb
import pandas as pd
import submitit

from experiment_settings import method_names


REPO_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path("results")
LOG_ROOT = Path("logs")
LEGACY_RBM_METHODS = ("BernoulliRBM", "BernoulliRBMweighted")


def aggregation_methods(ml: bool = False):
    methods, weighted = method_names()
    return tuple(method for method in methods if method not in LEGACY_RBM_METHODS), weighted


def initialize_database(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    schema: str,
    keys: tuple[str, ...],
) -> None:
    conn.execute(f"CREATE TABLE IF NOT EXISTS {table}({schema})")
    key_list = ", ".join(keys)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table}_completed AS
        SELECT {key_list}, current_timestamp AS completed_at
        FROM {table} WHERE false
    """)


def completed_configs(
    conn: duckdb.DuckDBPyConnection, table: str, keys: tuple[str, ...]
) -> set[tuple]:
    return set(conn.execute(
        f"SELECT {', '.join(keys)} FROM {table}_completed"
    ).fetchall())


def write_batch(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    columns: tuple[str, ...],
    keys: tuple[str, ...],
    frame: pd.DataFrame,
) -> int:
    if frame.empty:
        return 0
    batch = f"{table}_batch"
    conn.register(batch, frame)
    match = " AND ".join(f"target.{key} = batch.{key}" for key in keys)
    key_list = ", ".join(keys)
    try:
        conn.execute("BEGIN")
        conn.execute(f"""
            DELETE FROM {table} AS target
            USING (SELECT DISTINCT {key_list} FROM {batch}) AS batch
            WHERE {match}
        """)
        conn.execute(f"""
            INSERT INTO {table} ({', '.join(columns)})
            SELECT {', '.join(columns)} FROM {batch}
        """)
        conn.execute(f"""
            DELETE FROM {table}_completed AS target
            USING (SELECT DISTINCT {key_list} FROM {batch}) AS batch
            WHERE {match}
        """)
        conn.execute(f"""
            INSERT INTO {table}_completed
            SELECT DISTINCT {key_list}, current_timestamp FROM {batch}
        """)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.unregister(batch)
    return len(frame)


def run_parallel_database(
    db_path: str,
    table: str,
    schema: str,
    columns: tuple[str, ...],
    keys: tuple[str, ...],
    make_jobs: Callable[[set[tuple]], Iterable[tuple]],
    worker: Callable[..., pd.DataFrame | None],
) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another aggregation is writing {path}") from exc

        conn = duckdb.connect(str(path))
        try:
            initialize_database(conn, table, schema, keys)
            jobs = list(make_jobs(completed_configs(conn, table, keys)))
            if not jobs:
                print(f"All {table} configurations are already aggregated.")
                return

            workers = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
            written = processed = failures = 0
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(worker, *job): job for job in jobs}
                for future in as_completed(futures):
                    try:
                        frame = future.result()
                    except Exception as exc:
                        failures += 1
                        print(
                            f"Failed job {futures[future]}: {type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        continue
                    if frame is not None:
                        written += write_batch(conn, table, columns, keys, frame)
                    processed += 1
                    if processed % 10 == 0:
                        print(
                            f"Processed {processed}/{len(jobs)} jobs; wrote {written} rows",
                            flush=True,
                        )
            if failures:
                raise RuntimeError(f"Failed {failures}/{len(jobs)} aggregation jobs")
            conn.execute("CHECKPOINT")
            print(f"Finished {path}: wrote {written} rows", flush=True)
        finally:
            conn.close()
            fcntl.flock(lock, fcntl.LOCK_UN)


def submit(function: Callable[[str], None], db_path: Path, name: str) -> None:
    n_cores = 1
    executor = submitit.AutoExecutor(LOG_ROOT / f"logs_{name}", cluster="slurm")
    executor.update_parameters(
        timeout_min=24 * 60,
        cpus_per_task=n_cores,
        mem_gb=2.0 * n_cores,
        tasks_per_node=1,
        slurm_job_name=name,
        slurm_setup=[


            f"export PYTHONPATH={REPO_DIR}",
            "export OMP_NUM_THREADS=1",
            "export OPENBLAS_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            "export BLIS_NUM_THREADS=1",
        ],
    )
    job = executor.submit(function, str(db_path))
    print(f"Submitted {name} job {job.job_id}; database: {db_path}")
