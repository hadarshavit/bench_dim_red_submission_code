from aggregation.bop_variant import COLUMNS, KEYS, SCHEMA, aggregate_bop_directory
from aggregation.common import RESULTS_DIR, run_parallel_database, submit
from experiment_settings import SYNTHETIC_FIDS, TUNING_INPUT_DIMS, TUNING_RED_DIM, TUNING_SAMPLE


DB_PATH = RESULTS_DIR / "aggregation" / "bop_tuning.duck"
TABLE = "bop_results"


def aggregate_directory(sample: int, fid: int, inpt_dim: int, red_dim: int):
    return aggregate_bop_directory(
        "bop_reduced_tuning", sample, fid, inpt_dim, red_dim
    )


def pending_jobs(completed):
    for fid in SYNTHETIC_FIDS:
        for inpt_dim in TUNING_INPUT_DIMS:
            key = (TUNING_SAMPLE, fid, inpt_dim, TUNING_RED_DIM)
            if key not in completed:
                yield key


def run_aggregation(db_path: str = str(DB_PATH)) -> None:
    run_parallel_database(
        db_path, TABLE, SCHEMA, COLUMNS, KEYS, pending_jobs, aggregate_directory
    )


def main() -> None:
    submit(run_aggregation, DB_PATH, "aggregate_bop_tuning")


if __name__ == "__main__":
    from aggregation.bop_tuning import main

    main()
