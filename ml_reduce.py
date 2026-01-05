import openml
from methods import methods
import time
import lzma
import pickle
import os
import submitit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd




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

def reduce(task_id, cv, method_name, n_components):
    if os.path.exists(f"benchmark_dim_red/ml_data_reduced/{task_id}_{cv}_{method_name}_{n_components}.parquet.xz"):
        return
    task = openml.tasks.get_task(task_id, download_splits=False)
    data, targets, categorical, attribute_names = task.get_dataset().get_data(target=task.target_name)
    if data.shape[1] <= n_components:
        return
    train_indices, test_indices = task.get_train_test_split_indices(
                repeat=0,
                fold=cv,
                sample=0,
            )
    preprocessor = get_preprocessor(categorical)
    preprocessor.fit(data.iloc[train_indices])
    data = preprocessor.transform(data)

    if method_name == 'lolP' or method_name == 'ivis' or method_name == 'pls_reg':
        le = LabelEncoder()
        targets = pd.Series(le.fit_transform(targets))

    if method_name == 'ivis':
        reducer = methods[method_name](n_components=n_components, ml=True)
    else:
        reducer = methods[method_name](n_components=n_components)
    
    try:
        start_fit = time.time()
        start_cpu_fit = time.perf_counter()
        reducer.fit(data[train_indices, :], targets.iloc[train_indices])
        end_cpu_fit = time.perf_counter()
        end_fit = time.time()
    except:
        if method_name == 'lle':
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
        f"benchmark_dim_red/ml_data_reduced/{task_id}_{cv}_{method_name}_{n_components}.parquet.xz", 
        compression='lz4')
    with lzma.open(
        f"benchmark_dim_red/ml_data_reduced/{task_id}_{cv}_{method_name}_{n_components}_time.pkl.xz",
        "wb",
    ) as file:
        pickle.dump([start_fit, end_fit, start_cpu_fit, 
                     end_cpu_fit, start_transform, end_transform, 
                     start_cpu_transform, end_cpu_transform], file)


def parallel_executor(task_id, cv):
     for method_name in methods.keys():
        for n_components in [5, 10, 20, 40]:
            if os.path.exists(f"benchmark_dim_red/ml_data_reduced/{task_id}_{cv}_{method_name}_{n_components}.parquet.xz"):
                continue
            try:
                reduce(task_id, cv, method_name, n_components)
            except Exception as e:
                print(e)
                continue


if __name__ == '__main__':
    executor = submitit.AutoExecutor('benchmark_dim_red/logs_ml', 'slurm')
    executor.update_parameters(
        timeout_min=60*48,
        slurm_partition="CLUSTER" ,
        slurm_array_parallelism=1024,
        cpus_per_task=1,
        mem_gb=5.2 * 1, 
        tasks_per_node=1,
        slurm_job_name="m_dim_red",
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
                        if os.path.exists(f"benchmark_dim_red/ml_data_reduced/{task_id}_{cv}_{method_name}_{n_components}.parquet.xz"):
                            continue
                        print(f'benchmark_dim_red/ml_data_reduced/{task_id}_{cv}_{method_name}_{n_components}.parquet.xz')
                        executor.submit(reduce, task_id, cv, method_name, n_components)
