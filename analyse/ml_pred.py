import argparse
from pathlib import Path

import duckdb
import pandas as pd

try:
    from .common import (
        AGGREGATION_DIR, FIGURE_DIR, box_figure, summary_heatmap_figure,
        write_figure,
    )
except ImportError:
    from common import (
        AGGREGATION_DIR, FIGURE_DIR, box_figure, summary_heatmap_figure,
        write_figure,
    )


SIZES = {"small": (0, 60), "medium": (60, 200), "large": (200, 1_000_000)}
SLIPMAP_MAX_SAMPLES = 1_000


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the ML prediction figures used by the JAIR paper.")
    parser.add_argument("--aggregation-dir", type=Path, default=AGGREGATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR / "ml")
    args = parser.parse_args()

    with duckdb.connect(str(args.aggregation_dir / "ml_pred.duck"), read_only=True) as conn:
        all_results = conn.execute("SELECT * FROM ml_pred_results").fetchdf()
    limited_results = all_results[all_results.n_samples <= SLIPMAP_MAX_SAMPLES].copy()
    df = all_results[all_results.method != "slipmap"].copy()
    for algo in ("rf", "gb", "mlp"):
        by_dimension = df[
            (df.algo == algo) & df.n_components.isin((5, 10, 20, 40))
        ].copy()
        k_scenarios = [f"k={value}" for value in (5, 10, 20, 40)]
        by_dimension["scenario"] = pd.Categorical(
            by_dimension.n_components.map(lambda value: f"k={value}"),
            categories=k_scenarios,
            ordered=True,
        )
        by_dimension["rank"] = by_dimension.groupby(
            ["task_id", "cv", "algo", "n_components"]
        ).acc.rank(method="average", ascending=False)
        write_figure(
            summary_heatmap_figure(
                by_dimension, "scenario", "acc", "Accuracy", higher_is_better=True
            ),
            args.output_dir / f"summary_k_{algo}_acc.pdf",
        )
        limited_by_dimension = limited_results[
            (limited_results.algo == algo)
            & limited_results.n_components.isin((5, 10, 20, 40))
        ].copy()
        limited_by_dimension["scenario"] = pd.Categorical(
            limited_by_dimension.n_components.map(lambda value: f"k={value}"),
            categories=k_scenarios,
            ordered=True,
        )
        limited_by_dimension["rank"] = limited_by_dimension.groupby(
            ["task_id", "cv", "algo", "n_components"]
        ).acc.rank(method="average", ascending=False)
        write_figure(
            summary_heatmap_figure(
                limited_by_dimension,
                "scenario",
                "acc",
                "Accuracy",
                higher_is_better=True,
            ),
            args.output_dir / f"summary_k_le1000_{algo}_acc.pdf",
        )
        for red_dim in (5, 10, 20, 40):
            subset = df[(df.n_components == red_dim) & (df.algo == algo)]
            write_figure(
                box_figure(subset, "acc", "Accuracy"),
                args.output_dir / f"agg_{red_dim}_{algo}_acc.pdf",
            )
        for size, (minimum, maximum) in SIZES.items():
            subset = df[
                (df.n_components == 5)
                & (df.algo == algo)
                & (df.n_features >= minimum)
                & (df.n_features <= maximum)
            ]
            write_figure(
                box_figure(subset, "acc", "Accuracy"),
                args.output_dir / f"agg_{size}_5_{algo}_acc.pdf",
            )
        size_frames = []
        for size, (minimum, maximum) in SIZES.items():
            subset = df[
                (df.n_components == 5)
                & (df.algo == algo)
                & (df.n_features >= minimum)
                & (df.n_features <= maximum)
            ].copy()
            subset["scenario"] = size.capitalize()
            size_frames.append(subset)
        by_size = pd.concat(size_frames, ignore_index=True)
        by_size["scenario"] = pd.Categorical(
            by_size["scenario"], categories=["Small", "Medium", "Large"], ordered=True
        )
        by_size["rank"] = by_size.groupby(
            ["task_id", "cv", "algo", "scenario"], observed=True
        ).acc.rank(method="average", ascending=False)
        write_figure(
            summary_heatmap_figure(
                by_size, "scenario", "acc", "Accuracy", higher_is_better=True
            ),
            args.output_dir / f"summary_size_{algo}_acc.pdf",
        )


if __name__ == "__main__":
    main()
