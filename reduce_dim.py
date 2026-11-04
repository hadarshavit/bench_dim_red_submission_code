from pathlib import Path
from experiment_settings import expected_synthetic_reducers, method_weight
import argparse
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from ioh import get_problem, ProblemClass
from sampling import create_initial_sample
import numpy as np
import submitit
from scipy.stats import rankdata
import time
from methods import methods, with_weight


def execute(fid, method_name, instance, red_dim, inpt_dim, sample):

    np.random.seed(42)
    print(method_name, flush=True)
    method_class = methods[method_name]
    weight = method_name in with_weight
    if os.path.exists(f"results/{sample}/data_reduced/{fid}/{inpt_dim}/{method_name}_{red_dim}_{instance}_{weight}_time.pkl"):
        print('Already done')
        print(f"results/{sample}/data_reduced/{fid}/{inpt_dim}/{method_name}_{red_dim}_{instance}_{weight}_time.pkl")
        reducer = method_class(n_components=red_dim)
        try:
            reducer.load(f"results/{sample}/data_reduced/{fid}/{inpt_dim}/{method_name}_{red_dim}_{instance}_{weight}_reducer.pkl")
            print('Loaded')
            return
        except:
            print('Error loading')
            Path(f'results/{sample}/data_reduced/{fid}/{inpt_dim}/{method_name}_{red_dim}_{instance}_{weight}_time.pkl').unlink(missing_ok=True)
            Path(f'results/{sample}/data_reduced/{fid}/{inpt_dim}/{method_name}_{red_dim}_{instance}_{weight}_reducer.pkl').unlink(missing_ok=True)


    f = get_problem(fid=fid, dimension=inpt_dim, instance=instance, problem_class=ProblemClass.BBOB)
    x = create_initial_sample(inpt_dim, sample_type = 'sobol', n=sample, lower_bound=-5, upper_bound=5, seed=42)

    y = x.apply(lambda u: f(u), axis = 1)
    reducer = method_class(n_components=red_dim)
    start = time.time()
    start_cpu = time.perf_counter()
    reducer.fit(x, y)
    end_cpu = time.perf_counter()
    end = time.time()


    reducer.save(f"results/{sample}/data_reduced/{fid}/{inpt_dim}/{method_name}_{red_dim}_{instance}_{weight}_reducer.pkl")
    Path(f'results/{sample}/data_reduced/{fid}/{inpt_dim}/{method_name}_{red_dim}_{instance}_{weight}_time.pkl').parent.mkdir(parents=True, exist_ok=True)
    with open(
        f"results/{sample}/data_reduced/{fid}/{inpt_dim}/{method_name}_{red_dim}_{instance}_{weight}_time.pkl",
        "wb",
    ) as file:
        pickle.dump([start, end, start_cpu, end_cpu], file)


if __name__ == "__main__":
    executor = submitit.AutoExecutor("logs/reduce_dim", cluster="slurm")
    executor.update_parameters(timeout_min=24 * 60, cpus_per_task=1, mem_gb=2.8)
    with executor.batch():
        for (sample, fid, inpt_dim, method_name, red_dim, instance, reducer) in expected_synthetic_reducers():
            weight = method_weight(method_name)
            result = reducer.with_name(f'{method_name}_{red_dim}_{instance}_{weight}_time.pkl')
            if not os.path.exists(result):
                executor.submit(execute, fid, method_name, instance, red_dim, inpt_dim, sample)
