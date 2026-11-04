import argparse
from pathlib import Path

try:
    from .bop_variant import create_variant_figures
    from .common import AGGREGATION_DIR, FIGURE_DIR
except ImportError:
    from bop_variant import create_variant_figures
    from common import AGGREGATION_DIR, FIGURE_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Create figures for tuned BO reducers.")
    parser.add_argument("--aggregation-dir", type=Path, default=AGGREGATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR / "bop_tuning")
    args = parser.parse_args()
    create_variant_figures(
        args.aggregation_dir / "bop_tuning.duck",
        args.output_dir,
        sample=100,
        input_dims=(120, 240),
        red_dim=10,
    )


if __name__ == "__main__":
    main()
