from functools import partial

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
from sklearn.neural_network import BernoulliRBM
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from umap import UMAP
from typing import TypeVar, Generic, List
import lzma
import pickle
import numpy as np
import os
from ivis import Ivis
import pandas as pd
from scipy.stats import rankdata

T = TypeVar('T')


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
    "DictionaryLearning": partial(
        MiniBatchDictionaryLearning, random_state=42
    ),
    "FA": FactorAnalysis,
    "BernoulliRBM": partial(BernoulliRBM, random_state=42),
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
    "NMF",
    "DictionaryLearning",
    "FA",
    "latentDA",
    "random_trees",
]


def scale_X(X, func_vals):
    X_mean = X.mean(axis=0)
    X_ = X - X_mean

    r = rankdata(func_vals)
    N = len(func_vals)
    w = np.log(N) - np.log(r)
    w /= np.sum(w)
    return X_ * w.reshape(-1, 1)


class SklearnWrapper:
    def __init__(self, sklearn_class, n_components, weight) -> None:
        self.sklearn_obj = sklearn_class(n_components=n_components)
        self.weight = weight
        self.x_mean = None

    def fit(self, X, y):
        if self.weight:
            self.x_mean = X.mean(axis=0)
            x_scaled = scale_X(X, y)
        else:
            x_scaled = X

        self.sklearn_obj.fit(x_scaled, y)
    
    def transform(self, X):
        if self.weight:
            x_ = X - self.x_mean
            x_red = self.sklearn_obj.transform(x_)
        else:
            x_red = self.sklearn_obj.transform(X)

        return x_red
    
    def save(self, path):
        with lzma.open(
            path,
            "wb",
        ) as file:
            pickle.dump([self.sklearn_obj, self.x_mean], file)

    def load(self, path, numpy=False):
        with lzma.open(
            path,
            "rb",
        ) as file:
            self.sklearn_obj, self.x_mean = pickle.load(file)
            if self.weight and numpy: 
                self.x_mean = self.x_mean.values.reshape(1, -1)


class RandomTreesWrapper:
    def __init__(self, n_components) -> None:
        self.sklearn_obj = RandomTreesEmbedding(n_estimators=n_components, random_state=42)

    def fit(self, X, y):
        self.sklearn_obj.fit(X, y)
    
    def transform(self, X):
        return self.sklearn_obj.transform(X)
    
    def save(self, path):
        with lzma.open(
            path,
            "wb",
        ) as file:
            pickle.dump(self.sklearn_obj, file)


class IvisWrapper:
    def __init__(self, n_components, ml=False) -> None:
        if ml:
            self.sklearn_obj = Ivis(embedding_dims=n_components, verbose=0, supervision_metric='sparse_categorical_crossentropy', k=10, model='maaten', batch_size=128)
        else:
            self.sklearn_obj = Ivis(embedding_dims=n_components, verbose=0, supervision_metric='mse', k=10, model='maaten', batch_size=64)
        self.ml = ml

    def fit(self, X, y):
        if not self.ml:
            self.sklearn_obj.fit(X.values, y)
        else:
            self.sklearn_obj.fit(X, y.values)
    
    def transform(self, X):
        if not self.ml:
            if isinstance(X, pd.DataFrame):
                return self.sklearn_obj.transform(X.values)
            else:
                return self.sklearn_obj.transform(X)
        else:
            return self.sklearn_obj.transform(X)
    
    def save(self, path):
        self.sklearn_obj.save_model(path)
    
    def load(self, path, numpy=False):
        self.sklearn_obj.load_model(path)


class PUMAPWrapper:
    def __init__(self, n_components) -> None:
        self.sklearn_obj = ParametricUMAP(n_components=n_components)

    def fit(self, X, y):
        self.sklearn_obj.fit(X, y)
    
    def transform(self, X):
        return self.sklearn_obj.transform(X)
    
    def save(self, path):
        os.makedirs(path, exist_ok=False)
        self.sklearn_obj.save(path, verbose=False)

    def load(self, path):
        self.sklearn_obj = load_ParametricUMAP(path, verbose=False)


class SlisemapWrapper:
    def __init__(self, n_components) -> None:
        self.n_components = n_components

    def fit(self, X, y):
        self.sklearn_obj = Slisemap(X=X, y=y, d=self.n_components, random_state=42)
        self.sklearn_obj.optimise()
    
    def transform(self, X):
        return self.sklearn_obj.predict(X)
    
    def save(self, path):
        self.sklearn_obj.save(path, compress=True, any_extension=True)


class SlipmapWrapper:
    def __init__(self, n_components) -> None:
        self.n_components = n_components

    def fit(self, X, y):
        self.sklearn_obj = Slipmap(X=X, y=y, d=self.n_components)
        self.sklearn_obj.optimise()
    
    def transform(self, X):
        return self.sklearn_obj.predict(X)
    
    def save(self, path):
        self.sklearn_obj.save(path, compress=True, any_extension=True)


methods_sklearn = {k: partial(SklearnWrapper, sklearn_class=v, weight=k in with_weight)
                   for k, v in methods_sklearn.items()}

methods = {
    **methods_sklearn
}

