from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter
import numpy as np

from methods import methods


def get_configspace(method_name: str, inpt_dim: int, red_dim: int, sample: int, ml: bool = False) -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=42)
    if method_name in {"PCA_rbf", "PCA_sigmoid"}:
        cs.add(UniformFloatHyperparameter("gamma", 1e-4, 1.0, log=True))
    elif method_name == "PCA_poly":
        cs.add(UniformIntegerHyperparameter("degree", 2, 5))
        cs.add(UniformFloatHyperparameter("gamma", 1e-4, 1.0, log=True))
    elif method_name == "FastICA":
        cs.add(UniformFloatHyperparameter("tol", 1e-5, 1e-2, log=True))
        cs.add(UniformIntegerHyperparameter("max_iter", 100, 1000, log=True))
    elif method_name == "lle":
        cs.add(UniformIntegerHyperparameter("n_neighbors", max(red_dim + 1, 5), min(sample - 1, 50)))
        cs.add(UniformFloatHyperparameter("reg", 1e-4, 1e-1, log=True))
    elif method_name == "NMF":
        cs.add(UniformFloatHyperparameter("alpha_W", 1e-5, 1e-1, log=True))
        cs.add(UniformFloatHyperparameter("alpha_H", 1e-5, 1e-1, log=True))
    elif method_name == "DictionaryLearning":
        cs.add(UniformFloatHyperparameter("alpha", 1e-4, 10.0, log=True))
        cs.add(UniformIntegerHyperparameter("max_iter", 100, 1000, log=True))
    elif method_name == "latentDA":
        cs.add(UniformFloatHyperparameter("learning_decay", 0.5, 0.9))
        cs.add(UniformIntegerHyperparameter("max_iter", 10, 100, log=True))
    elif method_name in {"BernoulliRBM", "BernoulliRBMweighted", "BernoulliRBMNorm"}:
        cs.add(UniformFloatHyperparameter("learning_rate", 1e-3, 1e-1, log=True))
        cs.add(UniformIntegerHyperparameter("n_iter", 10, 100, log=True))
        cs.add(UniformIntegerHyperparameter("batch_size", 10, min(sample, 128), log=True))
    elif method_name in {"UMAP", "UMAP_unsupervised", "parametric_UMAP", "parametric_UMAP_unsupervised"}:
        cs.add(UniformIntegerHyperparameter("n_neighbors", 5, min(sample - 1, 50)))
        cs.add(UniformFloatHyperparameter("min_dist", 0.0, 0.8))
    elif method_name in {"LocalMAP", "PaCMAP"}:
        cs.add(UniformIntegerHyperparameter("n_neighbors", 5, min(sample - 1, 50)))
        cs.add(UniformFloatHyperparameter("MN_ratio", 0.1, 1.0))
        cs.add(UniformFloatHyperparameter("FP_ratio", 1.0, 4.0))
    elif method_name == "ParamPaCMAP":
        cs.add(UniformIntegerHyperparameter("n_neighbors", 5, min(sample - 1, 50)))
        cs.add(UniformIntegerHyperparameter("n_MN", 2, min(sample - 1, 20)))
        cs.add(UniformIntegerHyperparameter("n_FP", 5, min(sample - 1, 50)))
        cs.add(UniformFloatHyperparameter("lr", 1e-4, 1e-2, log=True))
        cs.add(CategoricalHyperparameter("batch_size", [32, 64, 128]))
        cs.add(UniformIntegerHyperparameter("num_epochs", 50, 250, log=True))
    elif method_name == "slipmap":
        cs.add(UniformFloatHyperparameter("prototypes", 0.05, 0.5))
        cs.add(UniformFloatHyperparameter("lasso", 1e-5, 1e-2, log=True))
        cs.add(UniformFloatHyperparameter("ridge", 1e-5, 1e-2, log=True))
    elif method_name == "ivis":
        cs.add(UniformIntegerHyperparameter("k", 5, min(sample - 1, 30)))
        cs.add(CategoricalHyperparameter("batch_size", [32, 64, 128]))
    elif method_name.startswith(("Autoencoder", "DenoisingAutoencoder", "VAE")):
        cs.add(UniformIntegerHyperparameter("epochs", 20, 120, log=True))
        cs.add(CategoricalHyperparameter("batch_size", [32, 64, 128]))
        cs.add(UniformFloatHyperparameter("learning_rate", 1e-4, 1e-2, log=True))
        if method_name.startswith("DenoisingAutoencoder"):
            cs.add(UniformFloatHyperparameter("noise_std", 0.01, 0.2))
        if method_name.startswith("VAE"):
            cs.add(UniformFloatHyperparameter("beta", 1e-4, 1e-1, log=True))
    return cs


def make_method(method_name: str, n_components: int, config=None, ml: bool = False):
    params = {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in dict(config or {}).items()
    }
    return methods[method_name](n_components=n_components, ml=ml, **params)
