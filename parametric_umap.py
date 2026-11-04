"""PyTorch implementation of the core Parametric UMAP training loop.

This intentionally mirrors the method-level behavior of
``umap.parametric_umap.ParametricUMAP`` without the TensorFlow/Keras dependency:

* UMAP builds the fuzzy simplicial graph, including supervised ``y`` handling.
* A feed-forward encoder maps input samples to the embedding space.
* The encoder is trained on positive graph edges and random negative samples
  with UMAP's attractive/repulsive cross-entropy loss.

The implementation is focused on encoder-based transform support for benchmark
reducers. Keras-specific extras such as reconstruction losses, ONNX export, and
landmark continuation are intentionally out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import os
import tempfile
import importlib
import importlib.metadata
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch
from sklearn.utils import check_random_state
from torch import nn
import torch.nn.functional as F


os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), "benchmark_dim_red_numba_cache"),
)


def _load_umap_class():
    """Load umap.umap_.UMAP without executing umap/__init__.py.

    umap-learn's package __init__ eagerly imports TensorFlow-backed
    ParametricUMAP when TensorFlow is installed. On this cluster that import can
    segfault before our PyTorch implementation starts. Loading the submodule via
    a lightweight package shim avoids that optional TensorFlow path.
    """
    existing = sys.modules.get("umap.umap_")
    if existing is not None and hasattr(existing, "UMAP"):
        return existing.UMAP
    try:
        root = Path(importlib.util.find_spec("umap").submodule_search_locations[0])
    except Exception as exc:
        raise ImportError("Could not locate installed umap-learn package") from exc

    package = sys.modules.get("umap")
    if package is None or not hasattr(package, "__path__"):
        package = types.ModuleType("umap")
        package.__path__ = [str(root)]
        package.__file__ = str(root / "__init__.py")
        try:
            package.__version__ = importlib.metadata.version("umap-learn")
        except importlib.metadata.PackageNotFoundError:
            package.__version__ = "unknown"
        sys.modules["umap"] = package

    cls = importlib.import_module("umap.umap_").UMAP
    setattr(package, "UMAP", cls)
    return cls


UMAP = _load_umap_class()


def _as_float32_array(X) -> np.ndarray:
    if hasattr(X, "values"):
        X = X.values
    X = np.asarray(X, dtype=np.float32).copy()
    if X.ndim != 2:
        X = X.reshape((X.shape[0], -1))
    return np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


class _Encoder(nn.Module):
    def __init__(self, input_dim: int, n_components: int, hidden_dims: Sequence[int]):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            last_dim = int(hidden_dim)
        layers.append(nn.Linear(last_dim, int(n_components)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class _GraphEdges:
    head: np.ndarray
    tail: np.ndarray
    probability: np.ndarray


class ParametricUMAP:
    """Parametric UMAP reducer with a PyTorch encoder.

    Parameters are intentionally close to ``umap.UMAP`` and upstream
    ``ParametricUMAP``. Unknown keyword arguments are passed to the internal
    ``UMAP`` instance used for graph construction.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        target_metric: str = "categorical",
        n_epochs: int | None = None,
        batch_size: int | None = None,
        negative_sample_rate: int = 5,
        repulsion_strength: float = 1.0,
        learning_rate: float = 1e-3,
        hidden_dims: Sequence[int] = (100, 100, 100),
        n_training_epochs: int = 10,
        random_state: int | np.random.RandomState | None = 42,
        device: str | torch.device | None = None,
        verbose: int | bool = 0,
        **umap_kwargs,
    ) -> None:
        self.n_components = int(n_components)
        self.n_neighbors = int(n_neighbors)
        self.min_dist = float(min_dist)
        self.metric = metric
        self.target_metric = target_metric
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.negative_sample_rate = int(negative_sample_rate)
        self.repulsion_strength = float(repulsion_strength)
        self.learning_rate = float(learning_rate)
        self.hidden_dims = tuple(int(dim) for dim in hidden_dims)
        self.n_training_epochs = int(n_training_epochs)
        self.random_state = random_state
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.verbose = verbose
        self.umap_kwargs = dict(umap_kwargs)

        self.encoder_: _Encoder | None = None
        self.umap_: UMAP | None = None
        self.embedding_: np.ndarray | None = None
        self.input_dim_: int | None = None
        self.loss_history_: list[float] = []

    def _make_umap(self) -> UMAP:
        return UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            target_metric=self.target_metric,
            n_epochs=self.n_epochs,
            negative_sample_rate=self.negative_sample_rate,
            repulsion_strength=self.repulsion_strength,
            random_state=self.random_state,
            n_jobs=1,
            **self.umap_kwargs,
        )

    @staticmethod
    def _graph_edges(graph) -> _GraphEdges:
        graph = graph.tocoo()
        graph.sum_duplicates()
        mask = graph.data > 0
        head = graph.row[mask].astype(np.int64)
        tail = graph.col[mask].astype(np.int64)
        probability = graph.data[mask].astype(np.float64)
        probability = probability / probability.sum()
        return _GraphEdges(head=head, tail=tail, probability=probability)

    def _batch_size(self, n_samples: int) -> int:
        if self.batch_size is not None:
            return max(1, min(int(self.batch_size), n_samples))
        return max(1, min(1000, n_samples))

    def _loss(self, z_head: torch.Tensor, z_tail: torch.Tensor, z_neg: torch.Tensor) -> torch.Tensor:
        if self.umap_ is None:
            raise RuntimeError("UMAP graph has not been fitted.")
        a = torch.as_tensor(float(self.umap_._a), dtype=z_head.dtype, device=z_head.device)
        b = torch.as_tensor(float(self.umap_._b), dtype=z_head.dtype, device=z_head.device)

        positive_dist = torch.sum((z_head - z_tail) ** 2, dim=1)
        positive_logits = -torch.log1p(a * torch.pow(positive_dist + 1e-8, b))
        attraction = F.softplus(-positive_logits).mean()

        z_head_expanded = z_head.repeat_interleave(self.negative_sample_rate, dim=0)
        negative_dist = torch.sum((z_head_expanded - z_neg) ** 2, dim=1)
        negative_logits = -torch.log1p(a * torch.pow(negative_dist + 1e-8, b))
        repulsion = F.softplus(negative_logits).mean() * self.repulsion_strength
        return attraction + repulsion

    def fit(self, X, y=None):
        X_np = _as_float32_array(X)
        self.input_dim_ = X_np.shape[1]

        rng = check_random_state(self.random_state)
        if isinstance(self.random_state, (int, np.integer)):
            torch.manual_seed(int(self.random_state))

        self.umap_ = self._make_umap()
        if y is None:
            self.umap_.fit(X_np)
        else:
            self.umap_.fit(X_np, np.asarray(y))

        edges = self._graph_edges(self.umap_.graph_)
        batch_size = self._batch_size(len(X_np))
        steps_per_epoch = max(1, int(np.ceil(len(edges.head) / batch_size)))
        total_steps = max(1, steps_per_epoch * max(1, self.n_training_epochs))

        self.encoder_ = _Encoder(self.input_dim_, self.n_components, self.hidden_dims).to(self.device)
        optimizer = torch.optim.Adam(self.encoder_.parameters(), lr=self.learning_rate)
        X_tensor = torch.as_tensor(X_np, dtype=torch.float32, device=self.device)
        n_samples = len(X_np)
        self.loss_history_ = []

        self.encoder_.train()
        for step in range(total_steps):
            edge_idx = rng.choice(len(edges.head), size=batch_size, replace=True, p=edges.probability)
            head_idx = edges.head[edge_idx]
            tail_idx = edges.tail[edge_idx]
            neg_idx = rng.randint(0, n_samples, size=batch_size * self.negative_sample_rate)

            z_head = self.encoder_(X_tensor[head_idx])
            z_tail = self.encoder_(X_tensor[tail_idx])
            z_neg = self.encoder_(X_tensor[neg_idx])
            loss = self._loss(z_head, z_tail, z_neg)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.encoder_.parameters(), 4.0)
            optimizer.step()
            self.loss_history_.append(float(loss.detach().cpu()))

        self.embedding_ = self.transform(X_np)
        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.embedding_

    def transform(self, X, batch_size: int | None = None) -> np.ndarray:
        if self.encoder_ is None:
            raise RuntimeError("ParametricUMAP has not been fitted.")
        X_np = _as_float32_array(X)
        batch = batch_size or self._batch_size(len(X_np))
        outputs = []
        self.encoder_.eval()
        with torch.no_grad():
            for start in range(0, len(X_np), batch):
                xb = torch.as_tensor(X_np[start : start + batch], dtype=torch.float32, device=self.device)
                outputs.append(self.encoder_(xb).cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def __getstate__(self):
        state = dict(self.__dict__)
        encoder = state.pop("encoder_", None)
        state["device"] = str(self.device)
        if encoder is not None:
            state["encoder_state_dict_"] = encoder.cpu().state_dict()
            state["encoder_class_"] = (self.input_dim_, self.n_components, self.hidden_dims)
            encoder.to(self.device)
        return state

    def __setstate__(self, state):
        encoder_state = state.pop("encoder_state_dict_", None)
        encoder_spec = state.pop("encoder_class_", None)
        self.__dict__.update(state)
        self.device = torch.device(self.device)
        if encoder_state is not None and encoder_spec is not None:
            input_dim, n_components, hidden_dims = encoder_spec
            self.encoder_ = _Encoder(input_dim, n_components, hidden_dims).to(self.device)
            self.encoder_.load_state_dict(encoder_state)
        else:
            self.encoder_ = None
