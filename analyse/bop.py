import argparse
from pathlib import Path

import duckdb
import pandas as pd

try:
    from .common import (
        AGGREGATION_DIR, FIGURE_DIR, heatmap_figure, line_figure,
        summary_heatmap_figure, write_figure,
    )
except ImportError:
    from common import (
        AGGREGATION_DIR, FIGURE_DIR, heatmap_figure, line_figure,
        summary_heatmap_figure, write_figure,
    )


SYNTHETIC_FIGURES = (
    (100, 120, 5), (100, 120, 10), (100, 120, 20), (100, 120, 40),
    (100, 40, 20), (100, 60, 20), (100, 240, 20), (100, 500, 20),
    (100, 1000, 20), (100, 40, 5), (200, 40, 5), (500, 40, 5),
    (1000, 40, 5), (100, 1000, 40), (200, 1000, 40),
    (500, 1000, 40), (1000, 1000, 40),
)
REAL_FIGURES = (
    (100, 5), (100, 10), (100, 20), (100, 40),
    (200, 5), (500, 5), (1000, 5),
)


def synthetic_data(conn, sample: int, inpt_dim: int, red_dim: int):
    return conn.execute("""
        SELECT iteration + 1 AS evaluation, method_name, AVG(rank) AS mean_rank
        FROM bop_results
        WHERE sample = ? AND inpt_dim = ? AND red_dim = ? AND iteration >= 30
        GROUP BY iteration, method_name
        ORDER BY iteration, method_name
    """, [sample, inpt_dim, red_dim]).fetchdf()


def real_data(conn, sample: int, red_dim: int):
    return conn.execute("""
        SELECT iteration + 1 AS evaluation, method_name, AVG(rank) AS mean_rank
        FROM bop_real_results
        WHERE sample = ? AND red_dim = ? AND iteration >= 30
        GROUP BY iteration, method_name
        ORDER BY iteration, method_name
    """, [sample, red_dim]).fetchdf()


def synthetic_final_data(conn, sample: int, inpt_dim: int, red_dim: int):
    return conn.execute("""
        WITH final AS (
            SELECT fid, method_name, regret, rank
            FROM bop_results
            WHERE sample = ? AND inpt_dim = ? AND red_dim = ?
            QUALIFY iteration = MAX(iteration) OVER (PARTITION BY fid, method_name)
        )
        SELECT fid, method_name, rank,
            (regret - MIN(regret) OVER (PARTITION BY fid)) /
            NULLIF(MAX(regret) OVER (PARTITION BY fid) - MIN(regret) OVER (PARTITION BY fid), 0)
            AS normalized_final
        FROM final
    """, [sample, inpt_dim, red_dim]).fetchdf()


def real_final_data(conn, sample: int, red_dim: int):
    return conn.execute("""
        WITH final AS (
            SELECT fid, method_name, value, rank
            FROM bop_real_results
            WHERE sample = ? AND red_dim = ?
            QUALIFY iteration = MAX(iteration) OVER (PARTITION BY fid, method_name)
        )
        SELECT fid, method_name, rank,
            (value - MIN(value) OVER (PARTITION BY fid)) /
            NULLIF(MAX(value) OVER (PARTITION BY fid) - MIN(value) OVER (PARTITION BY fid), 0)
            AS normalized_final
        FROM final
    """, [sample, red_dim]).fetchdf()


def write_final_summary(conn, configs, loader, path: Path) -> None:
    frames = []
    for label, config in configs:
        frame = loader(conn, *config).copy()
        frame["scenario"] = label
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    write_figure(
        summary_heatmap_figure(
            data,
            "scenario",
            "normalized_final",
            "Normalised final value",
            higher_is_better=False,
        ),
        path,
    )


def synthetic_figure(conn, sample: int, inpt_dim: int, red_dim: int):
    return line_figure(
        synthetic_data(conn, sample, inpt_dim, red_dim),
        "evaluation", "mean_rank", "Mean Rank",
    )


def real_figure(conn, sample: int, red_dim: int):
    return line_figure(
        real_data(conn, sample, red_dim),
        "evaluation", "mean_rank", "Mean Rank",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the BOP figures used by the JAIR paper.")
    parser.add_argument("--aggregation-dir", type=Path, default=AGGREGATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR / "bop")
    args = parser.parse_args()

    with duckdb.connect(str(args.aggregation_dir / "bop.duck"), read_only=True) as conn:
        for sample, inpt_dim, red_dim in SYNTHETIC_FIGURES:
            df = synthetic_data(conn, sample, inpt_dim, red_dim)
            write_figure(
                line_figure(df, "evaluation", "mean_rank", "Mean Rank"),
                args.output_dir / f"rank_{sample}_{inpt_dim}_{red_dim}.pdf",
            )
            write_figure(
                heatmap_figure(df, "evaluation", "mean_rank"),
                args.output_dir / f"heatmap_{sample}_{inpt_dim}_{red_dim}.pdf",
            )
        write_final_summary(
            conn,
            [(f"k={k}", (100, 120, k)) for k in (5, 10, 20, 40)],
            synthetic_final_data,
            args.output_dir / "summary_red_dim_100_120.pdf",
        )
        write_final_summary(
            conn,
            [(f"D={d}", (100, d, 20)) for d in (40, 60, 120, 240, 500, 1000)],
            synthetic_final_data,
            args.output_dir / "summary_inpt_dim_100_20.pdf",
        )
        write_final_summary(
            conn,
            [(f"N={n}", (n, 40, 5)) for n in (100, 200, 500, 1000)],
            synthetic_final_data,
            args.output_dir / "summary_sample_40_5.pdf",
        )
        write_final_summary(
            conn,
            [(f"N={n}", (n, 1000, 40)) for n in (100, 200, 500, 1000)],
            synthetic_final_data,
            args.output_dir / "summary_sample_1000_40.pdf",
        )
    with duckdb.connect(str(args.aggregation_dir / "bop_real.duck"), read_only=True) as conn:
        for sample, red_dim in REAL_FIGURES:
            df = real_data(conn, sample, red_dim)
            write_figure(
                line_figure(df, "evaluation", "mean_rank", "Mean Rank"),
                args.output_dir / f"rank_real_{sample}_{red_dim}.pdf",
            )
            write_figure(
                heatmap_figure(df, "evaluation", "mean_rank"),
                args.output_dir / f"heatmap_real_{sample}_{red_dim}.pdf",
            )
        write_final_summary(
            conn,
            [(f"k={k}", (100, k)) for k in (5, 10, 20, 40)],
            real_final_data,
            args.output_dir / "summary_real_red_dim_100.pdf",
        )
        write_final_summary(
            conn,
            [(f"N={n}", (n, 5)) for n in (100, 200, 500, 1000)],
            real_final_data,
            args.output_dir / "summary_real_sample_5.pdf",
        )


if __name__ == "__main__":
    main()
