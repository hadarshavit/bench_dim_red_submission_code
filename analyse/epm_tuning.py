import argparse
from pathlib import Path

import duckdb

try:
    from .common import AGGREGATION_DIR, FIGURE_DIR, write_figure
    from .epm import synthetic_figure
except ImportError:
    from common import AGGREGATION_DIR, FIGURE_DIR, write_figure
    from epm import synthetic_figure


def main() -> None:
    parser = argparse.ArgumentParser(description="Create figures for tuned EPM reducers.")
    parser.add_argument("--aggregation-dir", type=Path, default=AGGREGATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR / "epm_tuning")
    args = parser.parse_args()

    with duckdb.connect(
        str(args.aggregation_dir / "epm_tuning.duck"), read_only=True
    ) as conn:
        for inpt_dim in (120, 240):
            write_figure(
                synthetic_figure(conn, 100, inpt_dim, 10),
                args.output_dir / f"agg_sample_red_inpt_100_{inpt_dim}_10.pdf",
            )


if __name__ == "__main__":
    main()
