import openml
from methods import methods
import time
import lzma
import pickle
import os
import submitit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import hdbscan
from sklearn.metrics.cluster import adjusted_rand_score, \
    adjusted_mutual_info_score, homogeneity_completeness_v_measure, fowlkes_mallows_score, \
        silhouette_score, calinski_harabasz_score


def reduce(task_id, cv, method_name, n_components):
    print(task_id, cv, method_name, n_components, flush=True)
    df = pd.read_parquet(f"benchmark_dim_red/ml_data_reduced/{task_id}_{cv}_{method_name}_{n_components}.parquet.xz")

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

    dbscan = hdbscan.HDBSCAN()
    labels = dbscan.fit_predict(Xtest)

    ari = adjusted_rand_score(Ytest, labels)
    ami = adjusted_mutual_info_score(Ytest, labels)
    hcv = homogeneity_completeness_v_measure(Ytest, labels)
    fms = fowlkes_mallows_score(Ytest, labels)

    with lzma.open(f"benchmark_dim_red/ml_data_reduced/{task_id}_{cv}_{method_name}_{n_components}_cluster.pkl.xz", "wb") as file:
        pickle.dump([ari, ami, hcv, fms], file)


def parallel_executor(task_id, cv):
     for method_name in methods.keys():
        for n_components in [5, 10, 20, 40]:
            if os.path.exists(f"benchmark_dim_red/ml_data_reduced/{task_id}_{cv}_{method_name}_{n_components}_cluster.pkl.xz"):
                continue
            try:
                reduce(task_id, cv, method_name, n_components)
            except Exception as e:
                print(e)
                pass


if __name__ == '__main__':
    executor = submitit.AutoExecutor('benchmark_dim_red/logs_ml', 'slurm')
    executor.update_parameters(
        timeout_min=60*48,
        slurm_partition="PARTITION" ,
        slurm_array_parallelism=1024,
        cpus_per_task=1,
        mem_gb=2.8 * 1, 
        tasks_per_node=1,
        slurm_job_name="cluster",
        slurm_setup=["module -q load GCCcore/.12.2.0", "module -q load Python/3.10.8", 
                    "export PYTHONPATH=/benchmark_dim_red",
                    "export OMP_NUM_THREADS=1",
                    "export OPENBLAS_NUM_THREADS=1",
                    "export MKL_NUM_THREADS=1",
                    "export BLIS_NUM_THREADS=1",
                    "source /venvs/bench_dim_red/bin/activate"],
        slurm_additional_parameters={"account": "ACCOUNT"},
    )

    with executor.batch():
        suite = openml.study.get_suite(99)
        print(suite)
        cnt = 0
        for task_id in suite.tasks:
            task = openml.tasks.get_task(task_id, download_splits=False)
            data, targets, categorical, attribute_names = task.get_dataset().get_data(target=task.target_name)
            if data.shape[1] <= 20:
                continue    
            for cv in range(1, 10):
                
                for method_name in methods.keys():
                    for n_components in [5, 10, 20, 40]:
                        if data.shape[1] <= n_components:
                            continue
                        if data.shape[0] > 10_000:
                            continue
                        if os.path.exists(f"benchmark_dim_red/ml_data_reduced/{task_id}_{cv}_{method_name}_{n_components}_cluster.pkl.xz"):
                            continue
                        print(f'benchmark_dim_red/ml_data_reduced/{task_id}_{cv}_{method_name}_{n_components}.parquet.xz')
                        executor.submit(reduce, task_id, cv, method_name, n_components)