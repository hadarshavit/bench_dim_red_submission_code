from pathlib import Path

import duckdb

try:
    from .bop import synthetic_data
    from .common import heatmap_figure, line_figure, write_figure
except ImportError:
    from bop import synthetic_data
    from common import heatmap_figure, line_figure, write_figure


def create_variant_figures(
    db_path: Path,
    output_dir: Path,
    sample: int,
    input_dims: tuple[int, ...],
    red_dim: int,
) -> None:
    with duckdb.connect(str(db_path), read_only=True) as conn:
        for inpt_dim in input_dims:
            df = synthetic_data(conn, sample, inpt_dim, red_dim)
            stem = f"{sample}_{inpt_dim}_{red_dim}"
            write_figure(
                line_figure(df, "evaluation", "mean_rank", "Mean Rank"),
                output_dir / f"rank_{stem}.pdf",
            )
            write_figure(
                heatmap_figure(df, "evaluation", "mean_rank"),
                output_dir / f"heatmap_{stem}.pdf",
            )
