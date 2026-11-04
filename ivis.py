"""PyTorch implementation of the Ivis estimator API."""

from __future__ import annotations

import io
import json
import math
import os
import shutil
import tempfile
from copy import copy
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch import nn
from annoy import AnnoyIndex


class NotFittedError(Exception):
    pass


def _as_numpy(values):
    if isinstance(values, np.ndarray):
        return values
    if hasattr(values, "toarray") and callable(values.toarray):
        return np.asarray(values.toarray())
    return np.asarray(values)


def _flatten(values):
    values = _as_numpy(values)
    if values.ndim > 2:
        values = values.reshape(values.shape[0], -1)
    return np.asarray(values, dtype=np.float32)


def _seed_everything(seed: Optional[int]):
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pairwise_distances(x: np.ndarray, metric: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if metric in {"angular", "cosine"}:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        sim = (x @ x.T) / (norms @ norms.T)
        distances = 1.0 - sim
    elif metric == "euclidean":
        diff = x[:, None, :] - x[None, :, :]
        distances = np.sqrt(np.maximum(np.sum(diff * diff, axis=-1), 0.0))
    elif metric == "manhattan":
        distances = np.sum(np.abs(x[:, None, :] - x[None, :, :]), axis=-1)
    elif metric == "chebyshev":
        distances = np.max(np.abs(x[:, None, :] - x[None, :, :]), axis=-1)
    elif metric == "dot":
        distances = -(x @ x.T)
    else:
        raise ValueError(f"Unsupported KNN metric: {metric}")
    np.fill_diagonal(distances, np.inf)
    return distances


def _build_knn_matrix(x: np.ndarray, k: int, metric: str) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be positive")
    if k >= x.shape[0]:
        raise ValueError("k must be smaller than the number of rows")
    distances = _pairwise_distances(x, metric)
    return np.argsort(distances, axis=1)[:, :k]


def _annoy_metric(metric: str) -> str:
    if metric == "cosine":
        return "angular"
    supported = {"angular", "euclidean", "manhattan", "hamming", "dot"}
    if metric not in supported:
        raise ValueError(f"Unsupported Annoy metric: {metric}")
    return metric


class _AnnoyKnnMatrix:
    def __init__(
        self,
        x: np.ndarray,
        k: int,
        metric: str,
        n_trees: int,
        search_k: int,
        precompute: bool,
        index_path: Optional[str],
        build_index_on_disk: bool,
    ):
        if k <= 0:
            raise ValueError("k must be positive")
        if k >= x.shape[0]:
            raise ValueError("k must be smaller than the number of rows")
        self.nrows = int(x.shape[0])
        self.k = int(k)
        self.metric = _annoy_metric(metric)
        self.search_k = int(search_k)
        self.index_dims = int(x.shape[1])
        self.index_path = index_path
        self.precomputed_neighbours = None
        self._tmpdir = None
        self.index = AnnoyIndex(self.index_dims, self.metric)

        if index_path is not None and Path(index_path).exists():
            self.index.load(index_path)
        else:
            if index_path is None:
                self._tmpdir = tempfile.TemporaryDirectory()
                index_path = str(Path(self._tmpdir.name) / "annoy.index")
            self.index_path = index_path
            if build_index_on_disk:
                self.index.on_disk_build(index_path)
            for idx, vector in enumerate(x):
                self.index.add_item(idx, np.asarray(vector, dtype=np.float32))
            self.index.build(n_trees)
            if not build_index_on_disk:
                self.index.save(index_path)

        if precompute:
            self.precomputed_neighbours = np.asarray(
                [self[i] for i in range(self.nrows)], dtype=np.int64
            )

    def __getitem__(self, idx: int) -> np.ndarray:
        if self.precomputed_neighbours is not None:
            return self.precomputed_neighbours[idx]
        return np.asarray(
            self.index.get_nns_by_item(idx, self.k, search_k=self.search_k),
            dtype=np.int64,
        )

    def __getstate__(self):
        state = dict(self.__dict__)
        state["index"] = None
        state["_tmpdir"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.index = AnnoyIndex(self.index_dims, self.metric)
        self.index.load(self.index_path)


def _same_label_neighbours(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    neighbour_rows = []
    for label in labels:
        neighbour_rows.append(np.flatnonzero(labels == label))
    return np.asarray(neighbour_rows, dtype=object)


def _distance(anchor: torch.Tensor, other: torch.Tensor, metric: str) -> torch.Tensor:
    delta = anchor - other
    if metric in {"pn", "euclidean"}:
        return torch.sqrt(torch.clamp(torch.sum(delta * delta, dim=-1), min=1e-12))
    if metric in {"manhattan", "manhattan_pn"}:
        return torch.sum(torch.abs(delta), dim=-1)
    if metric in {"chebyshev", "chebyshev_pn"}:
        return torch.max(torch.abs(delta), dim=-1).values
    if metric in {"cosine", "cosine_pn"}:
        return torch.relu(1.0 - nn.functional.cosine_similarity(anchor, other, dim=-1))
    raise ValueError(f"Unsupported distance: {metric}")


def _triplet_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, distance: str | Callable) -> torch.Tensor:
    if callable(distance):
        stacked = torch.stack([anchor, positive, negative], dim=0)
        result = distance(None, stacked)
        if not isinstance(result, torch.Tensor):
            result = torch.as_tensor(result, dtype=anchor.dtype, device=anchor.device)
        return result.mean()

    if distance in {"softmax_ratio", "softmax_ratio_pn"}:
        pos = _distance(anchor, positive, "euclidean").reshape(-1)
        if distance == "softmax_ratio_pn":
            neg_anchor = _distance(anchor, negative, "euclidean").reshape(-1)
            neg_positive = _distance(positive, negative, "euclidean").reshape(-1)
            neg = torch.minimum(neg_anchor, neg_positive)
        else:
            neg = _distance(anchor, negative, "euclidean").reshape(-1)
        probs = torch.softmax(torch.cat([pos, neg]), dim=0)
        ideal = torch.cat([torch.zeros_like(pos), torch.ones_like(neg)])
        return torch.mean(torch.abs(ideal - probs))

    if distance in {"pn", "manhattan_pn", "chebyshev_pn", "cosine_pn"}:
        pos = _distance(anchor, positive, distance)
        neg_anchor = _distance(anchor, negative, distance)
        neg_positive = _distance(positive, negative, distance)
        neg = torch.stack([neg_anchor, neg_positive], dim=0)
    else:
        pos = _distance(anchor, positive, distance)
        neg = _distance(anchor, negative, distance)
    return torch.relu(pos - neg + 1.0).mean()


def _init_linear(module: nn.Module):
    if isinstance(module, nn.Linear):
        fan_in = module.weight.shape[1]
        std = math.sqrt(1.0 / fan_in)
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class _BaseEncoder(nn.Module):
    def __init__(self, layer_sizes: list[int], input_dim: int):
        super().__init__()
        layers = []
        in_features = input_dim
        for index, size in enumerate(layer_sizes):
            layers.append(nn.Linear(in_features, size))
            layers.append(nn.SELU())
            if index < len(layer_sizes) - 1:
                layers.append(nn.AlphaDropout(0.1))
            in_features = size
        self.network = nn.Sequential(*layers)
        self.network.apply(_init_linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class _Encoder(nn.Module):
    def __init__(self, base: nn.Module, input_dim: int, embedding_dims: Optional[int]):
        super().__init__()
        self.base = base
        with torch.no_grad():
            sample = torch.zeros(1, input_dim)
            output = self.base(sample)
        base_dim = int(output.shape[-1])
        if embedding_dims is None or embedding_dims == base_dim:
            self.embedding = nn.Identity()
        else:
            self.embedding = nn.Linear(base_dim, embedding_dims)
            _init_linear(self.embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(self.base(x))


def _build_base_network(name: str, input_dim: int) -> nn.Module:
    if name == "szubert":
        return _BaseEncoder([128, 128, 128], input_dim)
    if name == "hinton":
        return _BaseEncoder([2000, 1000, 500], input_dim)
    if name == "maaten":
        return _BaseEncoder([500, 500, 2000], input_dim)
    raise NotImplementedError(f"Base network {name} is not implemented")


class _TripletNet(nn.Module):
    def __init__(self, encoder: _Encoder, supervised_head: Optional[nn.Module] = None):
        super().__init__()
        self.encoder = encoder
        self.supervised_head = supervised_head

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor):
        anchor_embedding = self.encoder(anchor)
        positive_embedding = self.encoder(positive)
        negative_embedding = self.encoder(negative)
        if self.supervised_head is None:
            return torch.stack([anchor_embedding, positive_embedding, negative_embedding], dim=0)
        return torch.stack([anchor_embedding, positive_embedding, negative_embedding], dim=0), self.supervised_head(anchor_embedding)


class _SoftmaxHead(nn.Module):
    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, n_classes)
        _init_linear(self.linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class Ivis:
    def __init__(
        self,
        embedding_dims: int = 2,
        *,
        k: int = 150,
        distance: str | Callable = "pn",
        batch_size: int = 128,
        epochs: int = 1000,
        n_epochs_without_progress: int = 20,
        n_trees: int = 50,
        ntrees=None,
        knn_distance_metric: str = "angular",
        search_k: int = -1,
        precompute: bool = True,
        model: str | nn.Module = "szubert",
        supervision_metric: str = "sparse_categorical_crossentropy",
        supervision_weight: float = 0.5,
        annoy_index_path: Optional[str] = None,
        callbacks=None,
        build_index_on_disk: bool = True,
        neighbour_matrix=None,
        verbose: int = 1,
        random_state: Optional[int] = 42,
        device: Optional[str] = None,
    ):
        self.embedding_dims = embedding_dims
        self.k = k
        self.distance = distance
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_epochs_without_progress = n_epochs_without_progress
        self.n_trees = n_trees
        self.ntrees = ntrees
        self.knn_distance_metric = knn_distance_metric
        self.search_k = search_k
        self.precompute = precompute
        self.model = model
        self.supervision_metric = supervision_metric
        self.supervision_weight = supervision_weight
        self.loss_history_ = []
        self.annoy_index_path = annoy_index_path
        self.callbacks = callbacks
        self.build_index_on_disk = build_index_on_disk
        self.neighbour_matrix = neighbour_matrix
        self.verbose = verbose
        self.random_state = random_state
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.callbacks_ = None
        self.encoder_ = None
        self.model_ = None
        self.n_classes_ = None
        self.neighbour_matrix_ = None
        self.supervised_model_ = None
        self.optimizer_ = None
        self.input_dim_ = None
        self._base_model_spec_ = None

    def _validate_parameters(self):
        self.callbacks_ = [] if self.callbacks is None else list(map(copy, self.callbacks))
        self.neighbour_matrix_ = self.neighbour_matrix
        if self.ntrees is not None:
            self.n_trees = self.ntrees
            self.ntrees = None

    def _get_serializable_dict(self):
        state = dict(self.__dict__)
        for key in ["model_", "encoder_", "supervised_model_", "optimizer_"]:
            state[key] = None
        state["callbacks"] = []
        state["callbacks_"] = []
        state["neighbour_matrix"] = None
        state["neighbour_matrix_"] = None
        state["device"] = str(self.device)
        state["_base_model_spec_"] = None
        if callable(state.get("distance")):
            state["distance"] = state["distance"].__name__
        if not isinstance(state.get("model"), str):
            state["model"] = None
        return state

    def __getstate__(self):
        state = self._get_serializable_dict()
        if self.encoder_ is not None:
            buffer = io.BytesIO()
            torch.save(self.encoder_.state_dict(), buffer)
            state["encoder_state_bytes"] = buffer.getvalue()
        if self.supervised_model_ is not None:
            buffer = io.BytesIO()
            torch.save(self.supervised_model_.state_dict(), buffer)
            state["supervised_state_bytes"] = buffer.getvalue()
        if self.optimizer_ is not None:
            buffer = io.BytesIO()
            torch.save(self.optimizer_.state_dict(), buffer)
            state["optimizer_state_bytes"] = buffer.getvalue()
        return state

    def __setstate__(self, state):
        encoder_state = state.pop("encoder_state_bytes", None)
        supervised_state = state.pop("supervised_state_bytes", None)
        optimizer_state = state.pop("optimizer_state_bytes", None)
        self.__dict__.update(state)
        if self.input_dim_ is not None:
            self._rebuild_model()
            if encoder_state is not None:
                self.encoder_.load_state_dict(torch.load(io.BytesIO(encoder_state), map_location=self.device, weights_only=False))
            if supervised_state is not None and self.supervised_model_ is not None:
                self.supervised_model_.load_state_dict(torch.load(io.BytesIO(supervised_state), map_location=self.device, weights_only=False))
            if optimizer_state is not None and self.optimizer_ is not None:
                self.optimizer_.load_state_dict(torch.load(io.BytesIO(optimizer_state), map_location=self.device, weights_only=False))

    def _build_neighbour_matrix(self, x: np.ndarray):
        if self.neighbour_matrix_ is not None:
            return self.neighbour_matrix_
        if self.neighbour_matrix is not None:
            self.neighbour_matrix_ = self.neighbour_matrix
        else:
            self.neighbour_matrix_ = _AnnoyKnnMatrix(
                x,
                k=self.k,
                metric=self.knn_distance_metric,
                n_trees=self.n_trees,
                search_k=self.search_k,
                precompute=self.precompute,
                index_path=self.annoy_index_path,
                build_index_on_disk=self.build_index_on_disk,
            )
        return self.neighbour_matrix_

    def _rebuild_model(self):
        if self.input_dim_ is None:
            return
        if isinstance(self.model, str):
            base = _build_base_network(self.model, self.input_dim_)
        else:
            base = self._base_model_spec_
        self.encoder_ = _Encoder(base.to(self.device), self.input_dim_, self.embedding_dims).to(self.device)
        with torch.no_grad():
            sample = torch.zeros(1, int(self.input_dim_), device=self.device)
            encoder_out_dim = int(self.encoder_(sample).shape[-1])
        if self.n_classes_ is not None:
            if self.supervision_metric == "sparse_categorical_crossentropy":
                self.supervised_model_ = _SoftmaxHead(encoder_out_dim, self.n_classes_).to(self.device)
            else:
                self.supervised_model_ = nn.Linear(encoder_out_dim, self.n_classes_).to(self.device)
                _init_linear(self.supervised_model_)
        else:
            self.supervised_model_ = None
        self.model_ = _TripletNet(self.encoder_, self.supervised_model_).to(self.device)
        self.optimizer_ = torch.optim.Adam(self.model_.parameters(), lr=1e-3)

    def _build_supervised_head(self, labels: np.ndarray):
        labels = np.asarray(labels)
        if getattr(self, "encoder_", None) is not None:
            with torch.no_grad():
                sample = torch.zeros(1, int(self.input_dim_), device=self.device)
                head_in_dim = int(self.encoder_(sample).shape[-1])
        else:
            head_in_dim = int(self.embedding_dims)

        if self.supervision_metric == "sparse_categorical_crossentropy":
            valid = labels[labels != -1]
            self.label_to_index_ = {label: index for index, label in enumerate(np.unique(valid))}
            self.n_classes_ = len(self.label_to_index_)
            return _SoftmaxHead(head_in_dim, self.n_classes_).to(self.device)
        if labels.ndim == 1:
            self.n_classes_ = 1
            head = nn.Linear(head_in_dim, 1).to(self.device)
            _init_linear(head)
            return head
        self.n_classes_ = labels.shape[-1]
        head = nn.Linear(head_in_dim, self.n_classes_).to(self.device)
        _init_linear(head)
        return head

    def _sample_triplets(self, x: np.ndarray, batch_indices: np.ndarray, rng: np.random.Generator):
        if hasattr(self.neighbour_matrix_, "get_batch"):
            neighbours = self.neighbour_matrix_.get_batch(batch_indices)
        elif isinstance(self.neighbour_matrix_, np.ndarray):
            neighbours = self.neighbour_matrix_[batch_indices]
        else:
            neighbours = [self.neighbour_matrix_[int(idx)] for idx in batch_indices]
        positive_indices = np.array([rng.choice(row) for row in neighbours], dtype=np.int64)
        negative_indices = np.empty_like(positive_indices)
        for row_index, row in enumerate(neighbours):
            row_set = set(np.asarray(row).tolist())
            while True:
                candidate = int(rng.integers(0, x.shape[0]))
                if candidate not in row_set:
                    negative_indices[row_index] = candidate
                    break
        return (
            x[batch_indices],
            x[positive_indices],
            x[negative_indices],
            batch_indices,
        )

    def _loss_with_supervision(self, anchor, positive, negative, labels=None):
        anchor_embedding = self.encoder_(anchor)
        positive_embedding = self.encoder_(positive)
        negative_embedding = self.encoder_(negative)
        triplet = _triplet_loss(anchor_embedding, positive_embedding, negative_embedding, self.distance)
        if labels is None or self.supervised_model_ is None:
            return triplet
        if self.supervision_metric == "sparse_categorical_crossentropy":
            mask = labels != -1
            if mask.any():
                logits = self.supervised_model_(anchor_embedding[mask])
                supervised = nn.functional.cross_entropy(logits, labels[mask].long())
                return (1.0 - self.supervision_weight) * triplet + self.supervision_weight * supervised
            return triplet
        outputs = self.supervised_model_(anchor_embedding)
        labels = labels.float().view(outputs.shape)
        supervised = nn.functional.mse_loss(outputs, labels)
        return (1.0 - self.supervision_weight) * triplet + self.supervision_weight * supervised

    def _fit(self, X, Y=None, shuffle_mode=True):
        self._validate_parameters()
        _seed_everything(self.random_state)
        x = _flatten(X)
        if self.batch_size > x.shape[0]:
            raise ValueError(
                f"batch_size value larger than num_rows in dataset (batch_size={self.batch_size}, rows={x.shape[0]})."
            )
        self.input_dim_ = x.shape[1]
        self._build_neighbour_matrix(x)

        if isinstance(self.model, str):
            base = _build_base_network(self.model, self.input_dim_)
            self._base_model_spec_ = None
        else:
            self._base_model_spec_ = self.model
            base = self.model
        self.encoder_ = _Encoder(base.to(self.device), self.input_dim_, self.embedding_dims).to(self.device)

        if Y is not None:
            self.supervised_model_ = self._build_supervised_head(np.asarray(Y))
            if self.supervision_metric == "sparse_categorical_crossentropy":
                Y = np.array([self.label_to_index_.get(label, -1) for label in np.asarray(Y)], dtype=np.int64)
        else:
            self.supervised_model_ = None

        self.model_ = _TripletNet(self.encoder_, self.supervised_model_).to(self.device)
        self.optimizer_ = torch.optim.Adam(self.model_.parameters(), lr=1e-3)

        best_loss = float("inf")
        best_state = None
        epochs_without_progress = 0
        self.loss_history_ = []
        self.model_.train()

        for epoch in range(self.epochs):
            indices = np.arange(x.shape[0])
            if shuffle_mode:
                np.random.shuffle(indices)
            batch_losses = []
            for batch_start in range(0, x.shape[0], self.batch_size):
                batch_indices = indices[batch_start: batch_start + self.batch_size]
                anchor, positive, negative, original_indices = self._sample_triplets(x, batch_indices, np.random.default_rng((self.random_state or 0) + epoch * 1000 + batch_start))
                anchor_t = torch.as_tensor(anchor, dtype=torch.float32, device=self.device)
                positive_t = torch.as_tensor(positive, dtype=torch.float32, device=self.device)
                negative_t = torch.as_tensor(negative, dtype=torch.float32, device=self.device)
                labels_t = None if Y is None else torch.as_tensor(np.asarray(Y)[original_indices], device=self.device)

                self.optimizer_.zero_grad(set_to_none=True)
                loss = self._loss_with_supervision(anchor_t, positive_t, negative_t, labels_t)
                loss.backward()
                self.optimizer_.step()
                batch_losses.append(float(loss.detach().cpu().item()))

            epoch_loss = float(np.mean(batch_losses)) if batch_losses else float("inf")
            self.loss_history_.append(epoch_loss)
            if self.verbose > 0:
                print(f"Epoch {epoch + 1}/{self.epochs}: loss={epoch_loss:.6f}")
            if epoch_loss + 1e-8 < best_loss:
                best_loss = epoch_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model_.state_dict().items()
                }
                epochs_without_progress = 0
            else:
                epochs_without_progress += 1
                if epochs_without_progress >= self.n_epochs_without_progress:
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def fit(self, X, Y=None, shuffle_mode=True):
        self._fit(X, Y, shuffle_mode)
        return self

    def fit_transform(self, X, Y=None, shuffle_mode=True):
        self.fit(X, Y, shuffle_mode)
        return self.transform(X)

    def transform(self, X):
        if self.encoder_ is None:
            raise NotFittedError("Model was not fitted yet. Call `fit` before calling `transform`.")
        x = _flatten(X)
        outputs = []
        self.encoder_.eval()
        with torch.no_grad():
            for batch_start in range(0, x.shape[0], self.batch_size):
                batch = torch.as_tensor(x[batch_start: batch_start + self.batch_size], dtype=torch.float32, device=self.device)
                outputs.append(self.encoder_(batch).cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def score_samples(self, X):
        if self.supervised_model_ is None:
            raise NotFittedError("Model was not trained in classification mode.")
        x = _flatten(X)
        outputs = []
        self.encoder_.eval()
        self.supervised_model_.eval()
        with torch.no_grad():
            for batch_start in range(0, x.shape[0], self.batch_size):
                batch = torch.as_tensor(x[batch_start: batch_start + self.batch_size], dtype=torch.float32, device=self.device)
                embedding = self.encoder_(batch)
                pred = self.supervised_model_(embedding)
                if self.supervision_metric == "sparse_categorical_crossentropy":
                    pred = torch.softmax(pred, dim=-1)
                outputs.append(pred.cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def save_model(self, folder_path, save_format="tf", overwrite=False):
        del save_format
        folder = Path(folder_path)
        if folder.exists():
            if not overwrite:
                raise FileExistsError(f"Folder '{folder}' already exists")
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)

        params = self._get_serializable_dict()
        params.update(
            {
                "input_dim_": self.input_dim_,
                "n_classes_": self.n_classes_,
                "loss_history_": self.loss_history_,
            }
        )
        (folder / "ivis_params.json").write_text(json.dumps(params, indent=2, sort_keys=True), encoding="utf-8")
        if self.encoder_ is not None:
            torch.save(self.encoder_.state_dict(), folder / "ivis_model.pt")
        if self.supervised_model_ is not None:
            torch.save(self.supervised_model_.state_dict(), folder / "supervised_model.pt")
        if self.optimizer_ is not None:
            torch.save(self.optimizer_.state_dict(), folder / "optimizer.pt")
        return self

    def load_model(self, folder_path):
        folder = Path(folder_path)
        params = json.loads((folder / "ivis_params.json").read_text(encoding="utf-8"))
        self.__dict__.update(params)
        self.device = torch.device(params.get("device", "cpu"))
        self.input_dim_ = params.get("input_dim_")
        self.n_classes_ = params.get("n_classes_")
        self.loss_history_ = params.get("loss_history_", [])
        self.neighbour_matrix_ = None
        self.callbacks_ = []

        if self.input_dim_ is not None:
            if self.model is None:
                raise NotImplementedError("Loading a custom base model requires the original module instance.")
            self._rebuild_model()
            model_state = torch.load(folder / "ivis_model.pt", map_location=self.device, weights_only=False)
            self.encoder_.load_state_dict(model_state)
            if self.n_classes_ is not None and (folder / "supervised_model.pt").exists():

                with torch.no_grad():
                    sample = torch.zeros(1, int(self.input_dim_), device=self.device)
                    enc_out = self.encoder_(sample)
                encoder_out_dim = int(enc_out.shape[-1])

            if self.n_classes_ is not None:
                if self.supervision_metric == "sparse_categorical_crossentropy":
                    self.supervised_model_ = _SoftmaxHead(encoder_out_dim, self.n_classes_).to(self.device)
                else:
                    self.supervised_model_ = nn.Linear(encoder_out_dim, self.n_classes_).to(self.device)
                    _init_linear(self.supervised_model_)
                supervised_state = torch.load(
                    folder / "supervised_model.pt",
                    map_location=self.device,
                    weights_only=False,
                )
                self.supervised_model_.load_state_dict(supervised_state)
            if (folder / "optimizer.pt").exists():
                optimizer_state = torch.load(
                    folder / "optimizer.pt",
                    map_location=self.device,
                    weights_only=False,
                )
                self.optimizer_.load_state_dict(optimizer_state)
        return self
