from pathlib import Path
from experiment_settings import ML_RED_DIMS, method_names, ml_cluster_path, ml_evaluation_task_configs, ml_method_allowed_for_config, ml_reduced_path
import os

import openml
from methods import methods
import time
import pickle
import submitit
from ml_reduce import read_reduced
from experiment_settings import ml_cluster_path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.metrics.cluster import adjusted_rand_score, \
    adjusted_mutual_info_score, homogeneity_completeness_v_measure, fowlkes_mallows_score, \
        silhouette_score, calinski_harabasz_score


def reduce(task_id, cv, method_name, n_components):
    print(task_id, cv, method_name, n_components, flush=True)
    result_path = ml_cluster_path(task_id, cv, method_name, n_components)
    if result_path.exists():
        return
    result_path.parent.mkdir(parents=True, exist_ok=True)
    df = read_reduced(task_id, cv, method_name, n_components)

    task = openml.tasks.get_task(task_id, download_splits=False)
    train_indices, test_indices = task.get_train_test_split_indices(
                repeat=0,
                fold=cv,
                sample=0,
            )

    X = df.values
    Y = task.get_dataset().get_data(target=task.target_name)[1]

    Xtrain = X[train_indices]
    Ytrain = Y[train_indices]
    Xtest = X[test_indices]
    Ytest = Y[test_indices]

    dbscan = HDBSCAN()
    labels = dbscan.fit_predict(Xtest)

    ari = adjusted_rand_score(Ytest, labels)
    ami = adjusted_mutual_info_score(Ytest, labels)
    hcv = homogeneity_completeness_v_measure(Ytest, labels)
    fms = fowlkes_mallows_score(Ytest, labels)


    temporary_path = result_path.with_name(f".{result_path.name}.{os.getpid()}.tmp")
    try:
        Path(temporary_path).parent.mkdir(parents=True, exist_ok=True)
        with open(temporary_path, "wb") as file:
            pickle.dump([ari, ami, hcv, fms], file)
        os.replace(temporary_path, result_path)
    finally:
        temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    executor = submitit.AutoExecutor("logs/ml_cluster", cluster="slurm")
    executor.update_parameters(timeout_min=24 * 60, cpus_per_task=1, mem_gb=2.8)
    with executor.batch():
        (method_names_, _) = method_names()
        for (task_id, folds, n_features, n_instances) in ml_evaluation_task_configs():
            for cv in range(folds):
                for method_name in method_names_:
                    for red_dim in ML_RED_DIMS:
                        if n_features <= red_dim or not ml_method_allowed_for_config(method_name, n_instances, red_dim):
                            continue
                        reduced = ml_reduced_path(task_id, cv, method_name, red_dim)
                        if not os.path.exists(ml_cluster_path(task_id, cv, method_name, red_dim)) and os.path.exists(reduced):
                            executor.submit(reduce, task_id, cv, method_name, red_dim)
