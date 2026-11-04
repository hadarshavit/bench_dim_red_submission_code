from functools import partial
import importlib
import inspect
from pathlib import Path
import sys


from lol import LOL
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import (
    PCA,
    FastICA,
    IncrementalPCA,
    KernelPCA,
    MiniBatchDictionaryLearning,
    MiniBatchNMF,
    MiniBatchSparsePCA,
    TruncatedSVD,
    FactorAnalysis,
    LatentDirichletAllocation,
)
from sklearn.manifold import MDS, TSNE, Isomap, LocallyLinearEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import MinMaxScaler
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.base import clone
from typing import TypeVar, Generic, List
import pickle
import numpy as np
import os
import torch
import pandas as pd
from scipy.sparse.linalg import ArpackNoConvergence
from scipy.stats import rankdata

T = TypeVar('T')


def ensure_local_slisemap_import():
    checkout = Path(__file__).resolve().parent / "slisemap"
    if checkout.exists():
        checkout_str = str(checkout)
        if sys.path[0] != checkout_str:
            sys.path.insert(0, checkout_str)


def ensure_local_parampacmap_import():
    checkout = Path(__file__).resolve().parent / "ParamRepulsor" / "src"
    if checkout.exists():
        checkout_str = str(checkout)
        if sys.path[0] != checkout_str:
            sys.path.insert(0, checkout_str)


def local_ivis_class():
    module_path = Path(__file__).resolve().parent / "ivis.py"
    spec = importlib.util.spec_from_file_location("benchmark_dim_red_ivis", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load local Ivis implementation from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Ivis


methods_sklearn = {
    "PCA_linear": partial(PCA, svd_solver="full", random_state=42),
    "PCA_rbf": partial(
        KernelPCA, kernel="rbf", random_state=42, n_jobs=1, fit_inverse_transform=False
    ),
    "PCA_sigmoid": partial(
        KernelPCA,
        kernel="sigmoid",
        random_state=42,
        n_jobs=1,
        fit_inverse_transform=False,
    ),
    "PCA_cosine": partial(
        KernelPCA,
        kernel="cosine",
        random_state=42,
        n_jobs=1,
        fit_inverse_transform=False,
    ),
    "PCA_poly": partial(
        KernelPCA, kernel="poly", random_state=42, n_jobs=1, fit_inverse_transform=False
    ),
    "IPCA": IncrementalPCA,
    "FastICA": partial(FastICA, random_state=42),
    "TruncatedSVD": partial(TruncatedSVD, random_state=42),
    "lolP": partial(LOL, random_state=42, svd_solver="full"),

    "lle": partial(LocallyLinearEmbedding, random_state=42),


    "pls_reg": partial(PLSRegression),
    "NMF": partial(MiniBatchNMF, random_state=42),
    "DictionaryLearning": partial(
        MiniBatchDictionaryLearning, random_state=42
    ),

    "FA": FactorAnalysis,
    "latentDA": LatentDirichletAllocation,
    "BernoulliRBM": partial(BernoulliRBM, random_state=42),
    "BernoulliRBMweighted": partial(BernoulliRBM, random_state=42),

    'GaussianRandomProjection': partial(GaussianRandomProjection, random_state=42),
    'SparseRandomProjection': partial(SparseRandomProjection, random_state=42),
}

with_weight = [
    "PCA_linear",
    "PCA_rbf",
    "PCA_sigmoid",
    "PCA_cosine",
    "PCA_poly",
    "IPCA",
    "FastICA",
    "TruncatedSVD",
    "isomap",
    "lle",
    "NMF",
    "DictionaryLearning",
    "sparsePCA",
    "FA",
    "latentDA",
    "BernoulliRBMweighted",
    "BernoulliRBMNorm",


    "UMAP_unsupervised",
    "denseMAP_unsupervised",
    "LocalMAP",
    "PaCMAP",
    "ParamPaCMAP",
    "parametric_UMAP_unsupervised",
    "Autoencoder",
    "Autoencoder_2layer",
    "DenoisingAutoencoder",
    "DenoisingAutoencoder_2layer",
    "VAE",
    "VAE_2layer",
    "paradime_ParametricTSNE",
    "paradime_ParametricTSNE_2layer",
    "paradime_ParametricUMAP",
    "paradime_ParametricUMAP_2layer",

    "random_trees",
    "isomap"
]


def scale_X(X, func_vals):
    X_mean = X.mean(axis=0)
    X_ = X - X_mean

    r = rankdata(func_vals)
    N = len(func_vals)
    w = np.log(N) - np.log(r)
    w /= np.sum(w)
    return X_ * w.reshape(-1, 1)


def as_numpy(X):
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        return X.values
    if hasattr(X, "detach"):
        return X.detach().cpu().numpy()
    return np.asarray(X)


def standardize_X(X):
    X_np = as_numpy(X).astype("float32", copy=False)
    X_np = np.nan_to_num(X_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    mean = X_np.mean(axis=0, keepdims=True)
    scale = X_np.std(axis=0, keepdims=True)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return (X_np - mean) / scale, mean, scale


def load_pickle(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def dump_pickle(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(obj, file)


class NearestNeighborTransformMixin:
    def _store_training_embedding(self, X, embedding):
        self.x_fit_ = as_numpy(X)
        self.embedding_ = as_numpy(embedding)
        n_neighbors = min(3, len(self.x_fit_))
        self.nn_ = NearestNeighbors(n_neighbors=n_neighbors)
        self.nn_.fit(self.x_fit_)

    def _nearest_neighbor_transform(self, X):
        X_np = as_numpy(X)
        if (
            hasattr(self, "x_fit_")
            and X_np.shape == self.x_fit_.shape
            and np.allclose(X_np, self.x_fit_)
        ):
            return self.embedding_

        distances, indices = self.nn_.kneighbors(X_np)
        weights = 1.0 / np.maximum(distances, 1e-12)
        weights /= weights.sum(axis=1, keepdims=True)
        return np.sum(self.embedding_[indices] * weights[..., None], axis=1)


class SklearnWrapper:
    def __init__(self, sklearn_class, n_components, weight, ml=False, **sklearn_params) -> None:
        self.sklearn_obj = sklearn_class(n_components=n_components, **sklearn_params)
        self.weight = weight
        self.x_mean = None
        self.ml = ml

    def fit(self, X, y):
        if self.weight and not self.ml:
            self.x_mean = X.mean(axis=0)
            x_scaled = scale_X(X, y)
        else:
            x_scaled = X

        try:
            self.sklearn_obj.fit(x_scaled, y)
        except ArpackNoConvergence:
            if not isinstance(self.sklearn_obj, KernelPCA):
                raise
            self.sklearn_obj = clone(self.sklearn_obj).set_params(
                eigen_solver="randomized"
            )
            self.sklearn_obj.fit(x_scaled, y)
        except ValueError as exc:
            if (
                self.sklearn_obj.__class__.__name__ == "LocallyLinearEmbedding"
                and "Error in determining null-space with ARPACK" in str(exc)
            ):
                self.sklearn_obj = clone(self.sklearn_obj).set_params(eigen_solver="dense")
                self.sklearn_obj.fit(x_scaled, y)
            else:
                raise

    def transform(self, X):
        if self.weight and not self.ml:
            x_ = as_numpy(X) - as_numpy(self.x_mean)
            x_red = self.sklearn_obj.transform(x_)
        else:
            x_red = self.sklearn_obj.transform(X)

        return x_red

    def save(self, path):
        dump_pickle(path, [self.sklearn_obj, self.x_mean])

    def load(self, path, numpy=False):
        self.sklearn_obj, self.x_mean = load_pickle(path)
        if self.weight and numpy and hasattr(self.x_mean, "values"):
            self.x_mean = self.x_mean.values.reshape(1, -1)


class NormalizedRBMWrapper(SklearnWrapper):
    def fit(self, X, y):
        X = as_numpy(X)
        if self.weight and not self.ml:
            self.x_mean = X.mean(axis=0)
            X = scale_X(X, as_numpy(y))
        self.scaler = MinMaxScaler(clip=True)
        self.sklearn_obj.fit(self.scaler.fit_transform(X), y)

    def transform(self, X):
        X = as_numpy(X)
        if self.weight and not self.ml:
            X = X - self.x_mean
        return self.sklearn_obj.transform(self.scaler.transform(X))

    def save(self, path):
        dump_pickle(path, self)

    def load(self, path, numpy=False):
        self.__dict__.update(load_pickle(path).__dict__)


class PositiveSklearnWrapper(SklearnWrapper):
    def __init__(self, sklearn_class, n_components, weight, ml=False, **sklearn_params) -> None:
        super().__init__(sklearn_class, n_components, weight, ml=ml, **sklearn_params)
        self.x_shift = None
        self.x_scaled_shift = None

    def _make_positive(self, X, fit=False):
        X_np = as_numpy(X).astype(float)
        if fit:
            mins = X_np.min(axis=0)
            self.x_shift = np.where(mins <= 0, -mins + 1e-9, 0.0)
        return np.maximum(X_np + self.x_shift, 1e-9)

    def _make_scaled_positive(self, X, fit=False):
        if fit:
            mins = X.min(axis=0)
            self.x_scaled_shift = np.where(mins <= 0, -mins + 1e-9, 0.0)
        return np.maximum(X + self.x_scaled_shift, 1e-9)

    def fit(self, X, y):
        X_pos = self._make_positive(X, fit=True)
        if self.weight and not self.ml:
            self.x_mean = X_pos.mean(axis=0)
            x_scaled = scale_X(pd.DataFrame(X_pos), y)
            x_scaled = self._make_scaled_positive(as_numpy(x_scaled), fit=True)
        else:
            x_scaled = X_pos
        self.sklearn_obj.fit(x_scaled, y)

    def transform(self, X):
        X_pos = self._make_positive(X)
        if self.weight and not self.ml:
            X_pos = X_pos - self.x_mean
            X_pos = self._make_scaled_positive(X_pos)
        return self.sklearn_obj.transform(X_pos)

    def save(self, path):
        dump_pickle(path, [self.sklearn_obj, self.x_mean, self.x_shift, self.x_scaled_shift])

    def load(self, path, numpy=False):
        self.sklearn_obj, self.x_mean, self.x_shift, self.x_scaled_shift = load_pickle(path)
        if self.weight and numpy and hasattr(self.x_mean, "values"):
            self.x_mean = self.x_mean.values.reshape(1, -1)


class RandomTreesWrapper:
    def __init__(self, n_components) -> None:
        self.sklearn_obj = RandomTreesEmbedding(n_estimators=n_components, random_state=42)

    def fit(self, X, y):
        self.sklearn_obj.fit(X, y)

    def transform(self, X):
        return self.sklearn_obj.transform(X)

    def save(self, path):
        dump_pickle(path, self.sklearn_obj)

    def load(self, path, numpy=False):
        self.sklearn_obj = load_pickle(path)


class IvisWrapper:
    def __init__(self, n_components, ml=False, k=10, batch_size=None) -> None:
        Ivis = local_ivis_class()

        if ml:
            self.sklearn_obj = Ivis(embedding_dims=n_components, verbose=0, supervision_metric='sparse_categorical_crossentropy', k=k, model='maaten', batch_size=batch_size or 128)
        else:
            self.sklearn_obj = Ivis(embedding_dims=n_components, verbose=0, supervision_metric='mse', k=k, model='maaten', batch_size=batch_size or 64)
        self.ml = ml

    def fit(self, X, y):
        X = as_numpy(X)
        self.sklearn_obj.batch_size = min(self.sklearn_obj.batch_size, len(X))
        if not self.ml:
            self.sklearn_obj.fit(X, as_numpy(y))
        else:
            self.sklearn_obj.fit(X, as_numpy(y))

    def transform(self, X):
        return self.sklearn_obj.transform(as_numpy(X))

    def save(self, path):
        dump_pickle(path, self.sklearn_obj)

    def load(self, path, numpy=False):
        self.sklearn_obj = load_pickle(path)


class UMAPWrapper(NearestNeighborTransformMixin):
    def __init__(self, n_components, densmap=False, supervised=True, weight=False, ml=False, n_neighbors=15, min_dist=0.1) -> None:
        from umap import UMAP

        target_metric = "categorical" if ml else "l2"
        self.sklearn_obj = UMAP(
            n_components=n_components,
            densmap=densmap,
            random_state=42,
            n_jobs=1,
            target_metric=target_metric,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
        )
        self.supervised = supervised
        self.weight = weight
        self.ml = ml
        self.densmap = densmap

    def fit(self, X, y):
        X_np = as_numpy(X)
        if self.weight and not self.ml:
            X_np = scale_X(X_np, as_numpy(y))
        if self.supervised:
            self.sklearn_obj.fit(X_np, as_numpy(y))
        else:
            self.sklearn_obj.fit(X_np)
        if self.densmap:
            self._store_training_embedding(X_np, self.sklearn_obj.embedding_)

    def transform(self, X):
        X_np = as_numpy(X)
        try:
            return self.sklearn_obj.transform(X_np)
        except NotImplementedError as exc:
            if self.densmap and "densMAP" in str(exc):
                return self._nearest_neighbor_transform(X_np)
            raise

    def save(self, path):
        dump_pickle(path, self.sklearn_obj)

    def load(self, path, numpy=False):
        self.sklearn_obj = load_pickle(path)


class LocalMAPWrapper:
    def __init__(self, n_components, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, weight=False, ml=False) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.MN_ratio = MN_ratio
        self.FP_ratio = FP_ratio
        self.weight = weight
        self.ml = ml
        self.localmap_obj = None

    def fit(self, X, y=None):
        from pacmap import LocalMAP

        X_np = as_numpy(X)
        if self.weight and not self.ml and y is not None:
            X_np = scale_X(X_np, as_numpy(y))
        self.localmap_obj = LocalMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            MN_ratio=self.MN_ratio,
            FP_ratio=self.FP_ratio,
            save_tree=True,
        )
        self.embedding_ = as_numpy(self.localmap_obj.fit_transform(X_np, init="pca"))

    def transform(self, X):
        return self.localmap_obj.transform(as_numpy(X))

    def save(self, path):
        dump_pickle(path, self)

    def load(self, path, numpy=False):
        ensure_local_parampacmap_import()
        other = load_pickle(path)
        self.__dict__.update(other.__dict__)


class PaCMAPWrapper:
    def __init__(self, n_components, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, weight=False, ml=False) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.MN_ratio = MN_ratio
        self.FP_ratio = FP_ratio
        self.weight = weight
        self.ml = ml
        self.pacmap_obj = None

    def fit(self, X, y=None):
        from pacmap import PaCMAP

        X_np = as_numpy(X)
        if self.weight and not self.ml and y is not None:
            X_np = scale_X(X_np, as_numpy(y))
        self.pacmap_obj = PaCMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            MN_ratio=self.MN_ratio,
            FP_ratio=self.FP_ratio,
            save_tree=True,
        )
        self.embedding_ = as_numpy(self.pacmap_obj.fit_transform(X_np, init="pca"))

    def transform(self, X):
        return self.pacmap_obj.transform(as_numpy(X))

    def save(self, path):
        dump_pickle(path, self)

    def load(self, path, numpy=False):
        other = load_pickle(path)
        self.__dict__.update(other.__dict__)


class ParamPaCMAPWrapper:
    def __init__(
        self,
        n_components,
        n_neighbors=10,
        n_FP=20,
        n_MN=5,
        lr=1e-3,
        batch_size=128,
        num_epochs=100,
        weight=True,
        ml=False,
    ) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.n_FP = n_FP
        self.n_MN = n_MN
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.weight = weight
        self.ml = ml
        self.x_weight_mean_ = None
        self.input_projector_ = None
        self.parampacmap_obj = None

    def fit(self, X, y=None):
        ensure_local_parampacmap_import()
        try:
            from parampacmap import ParamPaCMAP
        except ImportError as exc:
            raise ImportError(
                "ParamPaCMAP requires the optional 'parampacmap' package. "
                "Install it with: pip install parampacmap"
            ) from exc

        X_np = as_numpy(X).astype("float32", copy=True)
        if self.weight and not self.ml and y is not None:
            self.x_weight_mean_ = X_np.mean(axis=0, keepdims=True)
            X_np = scale_X(X_np, as_numpy(y))
        if X_np.shape[1] > 100 and len(X_np) < 100:
            self.input_projector_ = PCA(
                n_components=min(len(X_np), X_np.shape[1]),
                random_state=42,
            )
            X_np = self.input_projector_.fit_transform(X_np)
        self.parampacmap_obj = ParamPaCMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            n_FP=self.n_FP,
            n_MN=self.n_MN,
            lr=self.lr,
            batch_size=min(self.batch_size, len(X_np)),
            num_epochs=self.num_epochs,
            seed=42,
            verbose=False,
            num_workers=0,
            apply_pca=self.input_projector_ is None,
        )
        self.parampacmap_obj.fit(X_np)

    def transform(self, X):
        X_np = as_numpy(X).astype("float32", copy=True)
        if self.weight and not self.ml and self.x_weight_mean_ is not None:
            X_np = X_np - self.x_weight_mean_
        if self.input_projector_ is not None:
            X_np = self.input_projector_.transform(X_np)
        return self.parampacmap_obj.transform(X_np)

    def save(self, path):
        dump_pickle(path, self)

    def load(self, path, numpy=False):
        ensure_local_parampacmap_import()
        other = load_pickle(path)
        self.__dict__.update(other.__dict__)


class PUMAPWrapper:
    def __init__(self, n_components, supervised=True, weight=False, ml=False, n_neighbors=15, min_dist=0.1) -> None:
        from parametric_umap import ParametricUMAP

        target_metric = "categorical" if ml else "l2"
        self.sklearn_obj = ParametricUMAP(
            n_components=n_components,
            random_state=42,
            target_metric=target_metric,
            n_training_epochs=10,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
        )
        self.supervised = supervised
        self.weight = weight
        self.ml = ml

    def fit(self, X, y):
        X_np = as_numpy(X)
        if self.weight and not self.ml:
            X_np = scale_X(X_np, as_numpy(y))
        if self.supervised:
            self.sklearn_obj.fit(X_np, as_numpy(y))
        else:
            self.sklearn_obj.fit(X_np)

    def transform(self, X):
        return self.sklearn_obj.transform(as_numpy(X))

    def save(self, path):
        dump_pickle(path, self.sklearn_obj)

    def load(self, path, numpy=False):
        self.sklearn_obj = load_pickle(path)


class SlisemapWrapper(NearestNeighborTransformMixin):
    def __init__(self, n_components, ml=False) -> None:
        self.n_components = n_components
        self.ml = ml

    def fit(self, X, y):
        ensure_local_slisemap_import()
        from slisemap import Slisemap

        self.sklearn_obj = Slisemap(X=as_numpy(X), y=as_numpy(y), d=self.n_components, random_state=42)
        self.sklearn_obj.optimise()
        self._store_training_embedding(X, self.sklearn_obj.get_Z())

    def transform(self, X):
        return self._nearest_neighbor_transform(X)

    def save(self, path):
        dump_pickle(path, {
            "n_components": self.n_components,
            "x_fit": self.x_fit_,
            "embedding": self.embedding_,
            "nn": self.nn_,
        })

    def load(self, path, numpy=False):
        state = load_pickle(path)
        self.n_components = state["n_components"]
        self.x_fit_ = state["x_fit"]
        self.embedding_ = state["embedding"]
        self.nn_ = state["nn"]


class SlipmapWrapper(NearestNeighborTransformMixin):
    def __init__(self, n_components, prototypes=0.25, lasso=1e-4, ridge=1e-4, ml=False) -> None:
        self.n_components = n_components
        self.prototypes = prototypes
        self.lasso = lasso
        self.ridge = ridge
        self.ml = ml

    def fit(self, X, y):
        ensure_local_slisemap_import()
        from slisemap import Slipmap

        y_np = as_numpy(y)
        y_std = np.std(y_np)
        y_scaled = (y_np - np.mean(y_np)) / (y_std if y_std > 0 else 1.0)
        configs = [
            (self.prototypes, self.lasso, self.ridge, y_np, torch.float32),
            (self.prototypes, self.lasso, self.ridge, y_scaled, torch.float64),
        ]
        last_error = None
        for prototypes, lasso, ridge, y_fit, dtype in configs:
            try:
                self.sklearn_obj = self._fit_slipmap(Slipmap, X, y_fit, prototypes, lasso, ridge, dtype)
                self._store_training_embedding(X, self.sklearn_obj.get_Z())
                return
            except RuntimeError as exc:
                if "value cannot be converted to type float without overflow" not in str(exc):
                    raise
                last_error = exc
        raise last_error

    def _fit_slipmap(self, Slipmap, X, y, prototypes, lasso, ridge, dtype):
        self.sklearn_obj = Slipmap(
            X=as_numpy(X),
            y=as_numpy(y),
            d=self.n_components,
            prototypes=prototypes,
            lasso=lasso,
            ridge=ridge,
            jit=False,
            dtype=dtype,
            device=torch.device("cpu"),
        )
        self.sklearn_obj.optimise()
        return self.sklearn_obj

    def transform(self, X):
        return self._nearest_neighbor_transform(X)

    def save(self, path):
        dump_pickle(path, {
            "n_components": self.n_components,
            "x_fit": self.x_fit_,
            "embedding": self.embedding_,
            "nn": self.nn_,
        })

    def load(self, path, numpy=False):
        state = load_pickle(path)
        self.n_components = state["n_components"]
        self.x_fit_ = state["x_fit"]
        self.embedding_ = state["embedding"]
        self.nn_ = state["nn"]


class TorchDRWrapper:
    def __init__(self, torchdr_class, n_components, **kwargs) -> None:
        torchdr = importlib.import_module("torchdr")
        cls = getattr(torchdr, torchdr_class)
        if "transform" not in cls.__dict__:
            raise NotImplementedError(
                f"torchdr.{torchdr_class} does not implement native out-of-sample transform"
            )
        params = dict(n_components=n_components, random_state=42, device="cpu")
        params.update(kwargs)
        signature = inspect.signature(cls)
        if "lr" in signature.parameters and "lr" not in kwargs:
            params["lr"] = 0.1
        params = {k: v for k, v in params.items() if k in signature.parameters}
        self.torchdr_obj = cls(**params)
        self.x_mean_ = None
        self.x_scale_ = None

    def fit(self, X, y=None):
        X_np, self.x_mean_, self.x_scale_ = standardize_X(X)
        if hasattr(self.torchdr_obj, "fit_transform"):
            embedding = self.torchdr_obj.fit_transform(X_np)
        else:
            self.torchdr_obj.fit(X_np)
            embedding = self._embedding_from_estimator()
        self.embedding_ = as_numpy(embedding)

    def _embedding_from_estimator(self):
        if hasattr(self.torchdr_obj, "embedding_"):
            return self.torchdr_obj.embedding_
        if hasattr(self.torchdr_obj, "Z"):
            return self.torchdr_obj.Z
        raise AttributeError("TorchDR estimator did not expose an embedding.")

    def transform(self, X):
        X_np = as_numpy(X).astype("float32", copy=False)
        X_np = np.nan_to_num(X_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if self.x_mean_ is not None and self.x_scale_ is not None:
            X_np = (X_np - self.x_mean_) / self.x_scale_
        return as_numpy(self.torchdr_obj.transform(X_np))

    def save(self, path):
        dump_pickle(path, self)

    def load(self, path, numpy=False):
        other = load_pickle(path)
        self.__dict__.update(other.__dict__)


class ContrastivePCAWrapper:
    def __init__(self, n_components, algorithm, standardize=True, ml=False) -> None:
        self.n_components = n_components
        self.algorithm = algorithm
        self.standardize = standardize
        self.ml = ml
        self.contrastive_obj = None

    def _foreground_background(self, X, y):
        X_np = as_numpy(X)
        y_np = as_numpy(y).reshape(-1)

        if self.ml:
            labels, counts = np.unique(y_np, return_counts=True)
            fg_label = labels[np.argmax(counts)]
            fg_mask = y_np == fg_label
        else:
            fg_mask = y_np <= np.median(y_np)

        if fg_mask.all() or (~fg_mask).sum() == 0:
            fg_mask = np.zeros(len(y_np), dtype=bool)
            fg_mask[: max(1, len(y_np) // 2)] = True

        return X_np[fg_mask], X_np[~fg_mask]

    def fit(self, X, y):
        module = importlib.import_module(self.algorithm.lower())
        cls = getattr(module, self.algorithm)
        self.contrastive_obj = cls(
            n_components=self.n_components,
            standardize=self.standardize,
        )
        foreground, background = self._foreground_background(X, y)
        if self.algorithm == "CPCA":
            self.contrastive_obj.fit(fg=foreground, bg=background)
        else:
            self.contrastive_obj.fit(K=foreground, R=background)

    def transform(self, X):
        return self.contrastive_obj.transform(as_numpy(X))

    def save(self, path):
        dump_pickle(path, self)

    def load(self, path, numpy=False):
        other = load_pickle(path)
        self.__dict__.update(other.__dict__)


def _hidden_dims_for_autoencoder(in_dim, n_components, hidden_layers):
    first_hidden = min(256, max(32, min(in_dim, max(4 * n_components, in_dim // 2))))
    if hidden_layers == 1:
        return [first_hidden]
    second_hidden = min(first_hidden, max(16, max(2 * n_components, first_hidden // 2)))
    return [first_hidden, second_hidden]


def _make_mlp(dims, final_activation=False):
    layers = []
    for idx, (in_features, out_features) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(torch.nn.Linear(in_features, out_features))
        if idx < len(dims) - 2 or final_activation:
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


class _AutoencoderNet(torch.nn.Module):
    def __init__(self, in_dim, n_components, hidden_dims):
        super().__init__()
        self.encoder = _make_mlp([in_dim, *hidden_dims, n_components])
        self.decoder = _make_mlp([n_components, *reversed(hidden_dims), in_dim])

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)


class _VariationalAutoencoderNet(torch.nn.Module):
    def __init__(self, in_dim, n_components, hidden_dims):
        super().__init__()
        self.encoder = _make_mlp([in_dim, *hidden_dims], final_activation=True)
        last_hidden = hidden_dims[-1] if hidden_dims else in_dim
        self.mu = torch.nn.Linear(last_hidden, n_components)
        self.logvar = torch.nn.Linear(last_hidden, n_components)
        self.decoder = _make_mlp([n_components, *reversed(hidden_dims), in_dim])

    def encode_params(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h).clamp(-8.0, 8.0)

    def encode(self, x):
        mu, _ = self.encode_params(x)
        return mu

    def forward(self, x):
        mu, logvar = self.encode_params(x)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        return self.decoder(z), mu, logvar


class TorchAutoencoderWrapper:
    def __init__(
        self,
        n_components,
        variant="autoencoder",
        hidden_layers=1,
        epochs=80,
        batch_size=128,
        learning_rate=1e-3,
        noise_std=0.05,
        beta=0.01,
        weight=False,
        ml=False,
    ) -> None:
        self.n_components = n_components
        self.variant = variant
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.noise_std = noise_std
        self.beta = beta
        self.weight = weight
        self.ml = ml
        self.x_weight_mean_ = None
        self.hidden_dims_ = None
        self.x_mean_ = None
        self.x_scale_ = None
        self.model_ = None

    def _build_model(self, in_dim):
        self.hidden_dims_ = _hidden_dims_for_autoencoder(
            in_dim, self.n_components, self.hidden_layers
        )
        if self.variant == "vae":
            return _VariationalAutoencoderNet(in_dim, self.n_components, self.hidden_dims_)
        return _AutoencoderNet(in_dim, self.n_components, self.hidden_dims_)

    def fit(self, X, y=None):
        torch.manual_seed(42)
        X_np = as_numpy(X).astype("float32", copy=True)
        X_np = np.nan_to_num(X_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if self.weight and not self.ml:
            self.x_weight_mean_ = X_np.mean(axis=0, keepdims=True)
            X_np = scale_X(X_np, as_numpy(y))
        X_np, self.x_mean_, self.x_scale_ = standardize_X(X_np)
        tensor = torch.as_tensor(X_np, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(tensor)
        generator = torch.Generator().manual_seed(42)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
            generator=generator,
        )
        self.model_ = self._build_model(X_np.shape[1])
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        self.model_.train()
        for _ in range(self.epochs):
            for (batch,) in loader:
                optimizer.zero_grad()
                if self.variant == "vae":
                    recon, mu, logvar = self.model_(batch)
                    recon_loss = torch.nn.functional.mse_loss(recon, batch)
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + self.beta * kl_loss
                else:
                    model_input = batch
                    if self.variant == "denoising":
                        model_input = batch + self.noise_std * torch.randn_like(batch)
                    recon = self.model_(model_input)
                    loss = torch.nn.functional.mse_loss(recon, batch)
                loss.backward()
                optimizer.step()

    def transform(self, X):
        X_np = as_numpy(X).astype("float32", copy=True)
        X_np = np.nan_to_num(X_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if self.weight and not self.ml:
            X_np = X_np - self.x_weight_mean_
        X_np = (X_np - self.x_mean_) / self.x_scale_
        self.model_.eval()
        with torch.no_grad():
            z = self.model_.encode(torch.as_tensor(X_np, dtype=torch.float32))
        return z.cpu().numpy()

    def save(self, path):
        dump_pickle(path, {
            "n_components": self.n_components,
            "variant": self.variant,
            "hidden_layers": self.hidden_layers,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "noise_std": self.noise_std,
            "beta": self.beta,
            "weight": self.weight,
            "ml": self.ml,
            "x_weight_mean": self.x_weight_mean_,
            "hidden_dims": self.hidden_dims_,
            "x_mean": self.x_mean_,
            "x_scale": self.x_scale_,
            "state_dict": self.model_.state_dict(),
        })

    def load(self, path, numpy=False):
        state = load_pickle(path)
        self.n_components = state["n_components"]
        self.variant = state["variant"]
        self.hidden_layers = state["hidden_layers"]
        self.epochs = state["epochs"]
        self.batch_size = state["batch_size"]
        self.learning_rate = state["learning_rate"]
        self.noise_std = state["noise_std"]
        self.beta = state["beta"]
        self.weight = state.get("weight", False)
        self.ml = state.get("ml", False)
        self.x_weight_mean_ = state.get("x_weight_mean")
        self.hidden_dims_ = state["hidden_dims"]
        self.x_mean_ = state["x_mean"]
        self.x_scale_ = state["x_scale"]
        if numpy:
            if self.x_weight_mean_ is not None:
                self.x_weight_mean_ = as_numpy(self.x_weight_mean_)
            self.x_mean_ = as_numpy(self.x_mean_)
            self.x_scale_ = as_numpy(self.x_scale_)
        in_dim = int(as_numpy(self.x_mean_).shape[-1])
        if self.variant == "vae":
            self.model_ = _VariationalAutoencoderNet(in_dim, self.n_components, self.hidden_dims_)
        else:
            self.model_ = _AutoencoderNet(in_dim, self.n_components, self.hidden_dims_)
        self.model_.load_state_dict(state["state_dict"])
        self.model_.eval()


class ParadimeWrapper:
    def __init__(
        self,
        n_components,
        routine,
        hidden_layers=1,
        epochs=10,
        init_epochs=0,
        batch_size=32,
        weight=False,
        ml=False,
    ) -> None:
        self.n_components = n_components
        self.routine = routine
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.init_epochs = init_epochs
        self.batch_size = batch_size
        self.weight = weight
        self.ml = ml
        self.x_weight_mean_ = None
        self.x_mean_ = None
        self.x_scale_ = None
        self.paradime_obj = None

    def _routine_class(self):
        try:
            routines = importlib.import_module("paradime.routines")
        except ImportError as exc:
            raise ImportError(
                "ParaDime methods require the optional 'paradime' package. "
                "Install it with: pip install paradime"
            ) from exc
        return getattr(routines, self.routine)

    def _routine_kwargs(self, cls, X_np, hidden_dims):
        candidate_kwargs = {
            "dataset": X_np,
            "data": X_np,
            "in_dim": X_np.shape[1],
            "out_dim": self.n_components,
            "emb_dim": self.n_components,
            "n_components": self.n_components,
            "hidden_dims": hidden_dims,
            "epochs": self.epochs,
            "init_epochs": self.init_epochs,
            "batch_size": min(self.batch_size, len(X_np)),
            "init_batch_size": min(self.batch_size, len(X_np)),
            "use_cuda": False,
            "verbose": False,
        }
        if self.routine == "ParametricTSNE":
            candidate_kwargs["perplexity"] = min(5.0, max(2.0, (len(X_np) - 1) / 10))
            candidate_kwargs["initialization"] = None
        if self.routine == "ParametricUMAP":
            candidate_kwargs["n_neighbors"] = min(30, max(2, len(X_np) - 1))

        try:
            signature = inspect.signature(cls)
        except (TypeError, ValueError):
            signature = inspect.signature(cls.__init__)
        params = signature.parameters
        kwargs = {k: v for k, v in candidate_kwargs.items() if k in params}
        if kwargs:
            return kwargs
        return {
            "dataset": X_np,
            "hidden_dims": hidden_dims,
            "use_cuda": False,
            "verbose": False,
        }

    def fit(self, X, y=None):
        X_np = as_numpy(X).astype("float32", copy=True)
        X_np = np.nan_to_num(X_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if self.weight and not self.ml:
            self.x_weight_mean_ = X_np.mean(axis=0, keepdims=True)
            X_np = scale_X(X_np, as_numpy(y))
        X_np, self.x_mean_, self.x_scale_ = standardize_X(X_np)
        hidden_dims = _hidden_dims_for_autoencoder(
            X_np.shape[1], self.n_components, self.hidden_layers
        )
        cls = self._routine_class()
        self.paradime_obj = cls(**self._routine_kwargs(cls, X_np, hidden_dims))
        self._relax_perplexity_search()
        self.paradime_obj.train()

    def _relax_perplexity_search(self):
        for relation in getattr(self.paradime_obj, "global_relations", {}).values():
            transforms = getattr(relation, "transform", ())
            if not isinstance(transforms, (list, tuple)):
                transforms = (transforms,)
            for transform in transforms:
                if transform.__class__.__name__ == "PerplexityBasedRescale":
                    transform.kwargs = {"bracket": [1e-8, 1e8]}

    def transform(self, X):
        X_np = as_numpy(X).astype("float32", copy=True)
        X_np = np.nan_to_num(X_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if self.weight and not self.ml:
            X_np = X_np - self.x_weight_mean_
        X_np = (X_np - self.x_mean_) / self.x_scale_
        return as_numpy(self.paradime_obj.apply(X_np))

    def save(self, path):
        dump_pickle(path, self)

    def load(self, path, numpy=False):
        other = load_pickle(path)
        self.__dict__.update(other.__dict__)


methods_sklearn = {k: partial(SklearnWrapper, sklearn_class=v, weight=k in with_weight)
                   for k, v in methods_sklearn.items()}
methods_sklearn["NMF"] = partial(
    PositiveSklearnWrapper,
    sklearn_class=partial(MiniBatchNMF, random_state=42),
    weight="NMF" in with_weight,
)
methods_sklearn["latentDA"] = partial(
    PositiveSklearnWrapper,
    sklearn_class=LatentDirichletAllocation,
    weight="latentDA" in with_weight,
)

methods_extra = {
    "BernoulliRBMNorm": partial(
        NormalizedRBMWrapper,
        sklearn_class=partial(BernoulliRBM, random_state=42),
        weight=True,
    ),
    "UMAP": partial(UMAPWrapper, supervised=True, weight=False),
    "UMAP_unsupervised": partial(UMAPWrapper, supervised=False, weight=True),


    "LocalMAP": partial(LocalMAPWrapper, weight=True),
    "PaCMAP": partial(PaCMAPWrapper, weight=True),
    "ParamPaCMAP": partial(ParamPaCMAPWrapper, weight=True),
    "parametric_UMAP": partial(PUMAPWrapper, supervised=True, weight=False),
    "parametric_UMAP_unsupervised": partial(PUMAPWrapper, supervised=False, weight=True),

    "slipmap": SlipmapWrapper,
    "ivis": IvisWrapper,
    "Autoencoder": partial(TorchAutoencoderWrapper, variant="autoencoder", hidden_layers=1, weight=True),
    "Autoencoder_2layer": partial(TorchAutoencoderWrapper, variant="autoencoder", hidden_layers=2, weight=True),
    "DenoisingAutoencoder": partial(TorchAutoencoderWrapper, variant="denoising", hidden_layers=1, weight=True),
    "DenoisingAutoencoder_2layer": partial(TorchAutoencoderWrapper, variant="denoising", hidden_layers=2, weight=True),
    "VAE": partial(TorchAutoencoderWrapper, variant="vae", hidden_layers=1, weight=True),
    "VAE_2layer": partial(TorchAutoencoderWrapper, variant="vae", hidden_layers=2, weight=True),


}

methods = {
    **methods_sklearn,
    **methods_extra
}
