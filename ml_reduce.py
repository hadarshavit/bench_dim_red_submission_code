from pathlib import Path
from experiment_settings import ML_RED_DIMS, method_names, ml_evaluation_task_configs, ml_method_allowed_for_config, ml_reduce_time_path
import os

import openml
from methods import methods
import time
import pickle
import submitit
from experiment_settings import ml_reduced_path, ml_reduce_time_path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from scipy.sparse import issparse


def get_preprocessor(categorical_ind):
    categorical_ind = np.array(categorical_ind)
    (cat_idx,) = np.where(categorical_ind)
    (cont_idx,) = np.where(~categorical_ind)

    preprocessor = make_pipeline(
                ColumnTransformer(
                    [
                        (
                            "cat",
                            make_pipeline(
                                SimpleImputer(strategy="most_frequent"),
                                OrdinalEncoder(

                                    handle_unknown="use_encoded_value",
                                    encoded_missing_value=-1,
                                    unknown_value=-1,
                                ),
                            ),
                            cat_idx.tolist(),
                        ),
                        (
                            "cont",
                            make_pipeline(
                                SimpleImputer(strategy="median"), StandardScaler()
                            ),
                            cont_idx.tolist(),
                        ),
                    ]
                )
            )
    return preprocessor


def load_ml_data(task_id, cv):
    task = openml.tasks.get_task(task_id, download_splits=False)
    data, targets, categorical, _ = task.get_dataset().get_data(target=task.target_name)
    train_indices, test_indices = task.get_train_test_split_indices(
                repeat=0,
                fold=cv,
                sample=0,
            )
    preprocessor = get_preprocessor(categorical)
    preprocessor.fit(data.iloc[train_indices].to_numpy())
    data = preprocessor.transform(data.to_numpy())


    le = LabelEncoder()
    targets = pd.Series(le.fit_transform(targets))
    return data, targets, train_indices, test_indices


def reduce(task_id, cv, method_name, n_components, ml=True):
    result_path = ml_reduced_path(task_id, cv, method_name, n_components)
    time_path = ml_reduce_time_path(task_id, cv, method_name, n_components)
    if result_path.exists() and time_path.exists():
        return
    result_path.parent.mkdir(parents=True, exist_ok=True)
    data, targets, train_indices, _ = load_ml_data(task_id, cv)
    if data.shape[1] <= n_components:
        return

    reducer = methods[method_name](n_components=n_components, ml=ml)

    try:
        start_fit = time.time()
        start_cpu_fit = time.perf_counter()
        reducer.fit(data[train_indices, :], targets.iloc[train_indices])
        end_cpu_fit = time.perf_counter()
        end_fit = time.time()
    except Exception:
        if method_name != 'lle':
            raise
        reducer.sklearn_obj.eigen_solver = 'dense'
        start_fit = time.time()
        start_cpu_fit = time.perf_counter()
        reducer.fit(data[train_indices, :], targets.iloc[train_indices])
        end_cpu_fit = time.perf_counter()
        end_fit = time.time()

    start_transform = time.time()
    start_cpu_transform = time.perf_counter()
    data = reducer.transform(data)
    end_cpu_transform = time.perf_counter()
    end_transform = time.time()

    pd.DataFrame(data).to_parquet(
        result_path,
        compression='lz4')
    Path(time_path).parent.mkdir(parents=True, exist_ok=True)
    with open(
        time_path,
        "wb",
    ) as file:
        pickle.dump([start_fit, end_fit, start_cpu_fit,
                     end_cpu_fit, start_transform, end_transform,
                     start_cpu_transform, end_cpu_transform], file)


def read_reduced(task_id, cv, method_name, n_components):
    path = ml_reduced_path(task_id, cv, method_name, n_components)
    return pd.read_parquet(path)


if __name__ == "__main__":
    executor = submitit.AutoExecutor("logs/ml_reduce", cluster="slurm")
    executor.update_parameters(timeout_min=24 * 60, cpus_per_task=1, mem_gb=2.8)
    with executor.batch():
        (method_names_, _) = method_names()
        for (task_id, folds, n_features, n_instances) in ml_evaluation_task_configs():
            for cv in range(folds):
                for method_name in method_names_:
                    for red_dim in ML_RED_DIMS:
                        if n_features <= red_dim or not ml_method_allowed_for_config(method_name, n_instances, red_dim):
                            continue
                        if not os.path.exists(ml_reduce_time_path(task_id, cv, method_name, red_dim)):
                            executor.submit(reduce, task_id, cv, method_name, red_dim)
