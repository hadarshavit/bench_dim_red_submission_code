import argparse
from pathlib import Path

import duckdb

try:
    from .common import AGGREGATION_DIR, FIGURE_DIR, box_figure, write_figure
except ImportError:
    from common import AGGREGATION_DIR, FIGURE_DIR, box_figure, write_figure


DATASETS = (9985, 9981)
TIMINGS = (
    ("fit_cpu_time", "train_time"),
    ("transform_cpu_time", "transform_time"),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the ML reducer timing figures used by the JAIR paper."
    )
    parser.add_argument("--aggregation-dir", type=Path, default=AGGREGATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR / "time")
    args = parser.parse_args()

    with duckdb.connect(
        str(args.aggregation_dir / "ml_time.duck"), read_only=True
    ) as conn:
        df = conn.execute("SELECT * FROM ml_time_results").fetchdf()

    for dataset in DATASETS:
        subset = df[df.task_id == dataset].copy()
        for column, suffix in TIMINGS:
            value = f"{column}_hours"
            subset[value] = subset[column] / 3600
            write_figure(
                box_figure(subset, value, "Running time (hours)", log_y=True),
                args.output_dir / f"{dataset}_{suffix}.pdf",
            )


if __name__ == "__main__":
    main()
