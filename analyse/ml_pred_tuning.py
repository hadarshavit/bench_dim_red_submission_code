import argparse
from pathlib import Path

import duckdb

try:
    from .common import AGGREGATION_DIR, FIGURE_DIR, box_figure, write_figure
except ImportError:
    from common import AGGREGATION_DIR, FIGURE_DIR, box_figure, write_figure


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create prediction figures for tuned ML reducers."
    )
    parser.add_argument("--aggregation-dir", type=Path, default=AGGREGATION_DIR)
    parser.add_argument(
        "--output-dir", type=Path, default=FIGURE_DIR / "ml_pred_tuning"
    )
    args = parser.parse_args()

    with duckdb.connect(
        str(args.aggregation_dir / "ml_pred_tuning.duck"), read_only=True
    ) as conn:
        df = conn.execute("SELECT * FROM ml_pred_results").fetchdf()
    for algorithm in ("rf", "gb", "mlp"):
        for red_dim in (5, 10, 20, 40):
            subset = df[(df.n_components == red_dim) & (df.algo == algorithm)]
            write_figure(
                box_figure(subset, "acc", "Accuracy"),
                args.output_dir / f"agg_{red_dim}_{algorithm}_acc.pdf",
            )


if __name__ == "__main__":
    main()
