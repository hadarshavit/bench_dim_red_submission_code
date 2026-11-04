import argparse
from pathlib import Path

import duckdb

try:
    from .common import AGGREGATION_DIR, FIGURE_DIR, box_figure, write_figure
except ImportError:
    from common import AGGREGATION_DIR, FIGURE_DIR, box_figure, write_figure


METRICS = {"homogeneity": "Homogeneity", "completeness": "Completeness"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create clustering figures for tuned ML reducers."
    )
    parser.add_argument("--aggregation-dir", type=Path, default=AGGREGATION_DIR)
    parser.add_argument(
        "--output-dir", type=Path, default=FIGURE_DIR / "ml_cluster_tuning"
    )
    args = parser.parse_args()

    with duckdb.connect(
        str(args.aggregation_dir / "ml_cluster_tuning.duck"), read_only=True
    ) as conn:
        df = conn.execute("SELECT * FROM ml_cluster_results").fetchdf()
    for tuning_metric in METRICS:
        tuned = df[df.tuning_metric == tuning_metric]
        for score_metric, title in METRICS.items():
            for red_dim in (5, 10, 20, 40):
                subset = tuned[tuned.n_components == red_dim]
                write_figure(
                    box_figure(subset, score_metric, title),
                    args.output_dir
                    / f"agg_{red_dim}_tuned_{tuning_metric}_{score_metric}.pdf",
                )


if __name__ == "__main__":
    main()
