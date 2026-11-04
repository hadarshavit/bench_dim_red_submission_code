from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import qmc


def _as_bounds(bound: Sequence[float] | float, dim: int, name: str) -> np.ndarray:
    values = np.asarray(bound, dtype=float)
    if values.ndim == 0:
        return np.full(dim, float(values))
    if values.shape != (dim,):
        raise ValueError(f"{name} must be a scalar or a sequence with length {dim}")
    return values


def create_initial_sample(
    dim: int,
    n: int | None = None,
    sample_coefficient: int = 50,
    lower_bound: Sequence[float] | float = 0,
    upper_bound: Sequence[float] | float = 1,
    sample_type: str = "lhs",
    seed: int | None = None,
) -> pd.DataFrame:
    """Create an initial design sample compatible with pflacco.sampling.

    Supports the sampling modes used in this repository: latin hypercube
    (``lhs``), uniform random (``random``), and Sobol (``sobol``).
    """
    if dim < 1:
        raise ValueError("dim must be at least 1")
    if n is None:
        n = sample_coefficient * dim
    if n < 1:
        raise ValueError("n must be at least 1")

    lower = _as_bounds(lower_bound, dim, "lower_bound")
    upper = _as_bounds(upper_bound, dim, "upper_bound")
    if np.any(upper <= lower):
        raise ValueError("upper_bound must be greater than lower_bound for every dimension")

    sample_type = sample_type.lower()
    if sample_type == "lhs":
        sampler = qmc.LatinHypercube(d=dim, seed=seed)
        sample = sampler.random(n)
    elif sample_type == "sobol":
        sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
        sample = sampler.random(n)
    elif sample_type == "random":
        rng = np.random.default_rng(seed)
        sample = rng.random((n, dim))
    else:
        raise ValueError("sample_type must be one of 'lhs', 'random', or 'sobol'")

    sample = qmc.scale(sample, lower, upper)
    return pd.DataFrame(sample, columns=range(dim))
