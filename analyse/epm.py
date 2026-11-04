import argparse
from pathlib import Path

import duckdb
import pandas as pd

try:
    from .common import (
        AGGREGATION_DIR, FIGURE_DIR, box_figure, summary_heatmap_figure,
        write_figure, write_legend,
    )
except ImportError:
    from common import (
        AGGREGATION_DIR, FIGURE_DIR, box_figure, summary_heatmap_figure,
        write_figure, write_legend,
    )


SYNTHETIC_FIGURES = (
    (100, 120, 5), (100, 120, 10), (100, 120, 20), (100, 120, 40),
    (100, 40, 20), (100, 60, 20), (100, 240, 20), (100, 500, 20),
    (100, 1000, 20), (100, 40, 5), (200, 40, 5), (500, 40, 5),
    (1000, 40, 5), (100, 1000, 40), (200, 1000, 40),
    (500, 1000, 40), (1000, 1000, 40),
)
REAL_FIGURES = (
    (1000, 5), (1000, 10), (1000, 20), (1000, 40),
    (100, 5), (200, 5), (500, 5),
)


NORMALIZED_RMSE_QUERY = """
    WITH averaged AS (
        SELECT fid, method_name, cv, AVG(SQRT(mse)) AS rmse
        FROM {table}
        WHERE sample = ? AND {conditions} red_dim = ?
        GROUP BY fid, method_name, cv
    ), normalized AS (
        SELECT *,
            (rmse - MIN(rmse) OVER (PARTITION BY fid)) /
            NULLIF(MAX(rmse) OVER (PARTITION BY fid) - MIN(rmse) OVER (PARTITION BY fid), 0)
            AS normalized_rmse
        FROM averaged
    )
    SELECT fid, cv, method_name, normalized_rmse FROM normalized
    ORDER BY method_name
"""


def synthetic_figure(conn, sample: int, inpt_dim: int, red_dim: int):
    return box_figure(
        synthetic_data(conn, sample, inpt_dim, red_dim),
        "normalized_rmse", "Normalised RMSE",
    )


def synthetic_data(conn, sample: int, inpt_dim: int, red_dim: int):
    query = NORMALIZED_RMSE_QUERY.format(
        table="epm_results", conditions="inpt_dim = ? AND "
    )
    return conn.execute(query, [sample, inpt_dim, red_dim]).fetchdf()


def real_figure(conn, sample: int, red_dim: int):
    return box_figure(
        real_data(conn, sample, red_dim), "normalized_rmse", "Normalised RMSE"
    )


def real_data(conn, sample: int, red_dim: int):
    query = NORMALIZED_RMSE_QUERY.format(table="epm_real_results", conditions="")
    return conn.execute(query, [sample, red_dim]).fetchdf()


def write_summary(conn, configs, loader, path: Path) -> None:
    frames = []
    for label, config in configs:
        frame = loader(conn, *config).copy()
        frame["scenario"] = label
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["rank"] = data.groupby(["scenario", "fid", "cv"])[
        "normalized_rmse"
    ].rank(method="average")
    write_figure(
        summary_heatmap_figure(
            data,
            "scenario",
            "normalized_rmse",
            "Normalised RMSE",
            higher_is_better=False,
        ),
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the EPM figures used by the JAIR paper.")
    parser.add_argument("--aggregation-dir", type=Path, default=AGGREGATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR / "epm")
    args = parser.parse_args()

    with duckdb.connect(str(args.aggregation_dir / "epm.duck"), read_only=True) as conn:
        for sample, inpt_dim, red_dim in SYNTHETIC_FIGURES:
            write_figure(
                synthetic_figure(conn, sample, inpt_dim, red_dim),
                args.output_dir / f"agg_sample_red_inpt_{sample}_{inpt_dim}_{red_dim}.pdf",
            )
        write_summary(
            conn,
            [(f"k={k}", (100, 120, k)) for k in (5, 10, 20, 40)],
            synthetic_data,
            args.output_dir / "summary_red_dim_100_120.pdf",
        )
        write_summary(
            conn,
            [(f"D={d}", (100, d, 20)) for d in (40, 60, 120, 240, 500, 1000)],
            synthetic_data,
            args.output_dir / "summary_inpt_dim_100_20.pdf",
        )
        write_summary(
            conn,
            [(f"N={n}", (n, 40, 5)) for n in (100, 200, 500, 1000)],
            synthetic_data,
            args.output_dir / "summary_sample_40_5.pdf",
        )
        write_summary(
            conn,
            [(f"N={n}", (n, 1000, 40)) for n in (100, 200, 500, 1000)],
            synthetic_data,
            args.output_dir / "summary_sample_1000_40.pdf",
        )
    with duckdb.connect(str(args.aggregation_dir / "epm_real.duck"), read_only=True) as conn:
        for sample, red_dim in REAL_FIGURES:
            write_figure(
                real_figure(conn, sample, red_dim),
                args.output_dir / f"real_agg_sample_red_dim_{sample}_{red_dim}.pdf",
            )
        write_summary(
            conn,
            [(f"k={k}", (1000, k)) for k in (5, 10, 20, 40)],
            real_data,
            args.output_dir / "summary_real_red_dim_1000.pdf",
        )
        write_summary(
            conn,
            [(f"N={n}", (n, 5)) for n in (100, 200, 500, 1000)],
            real_data,
            args.output_dir / "summary_real_sample_5.pdf",
        )
    write_legend(args.output_dir.parent / "legend.pdf")


if __name__ == "__main__":
    main()
