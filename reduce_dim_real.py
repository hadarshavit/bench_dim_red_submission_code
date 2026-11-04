from pathlib import Path
from experiment_settings import expected_real_reducers, method_weight, real_data_path
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
import torch
from methods import methods, with_weight
from functools import partial
from real_world_benchs.real_world_benchs import *


benchs = {
    "Rover": Rover,
    "SVM": SVM,
    "Mujoco_ant": partial(Mujoco, 'ant'),
    "Mujoco_half_cheetah": partial(Mujoco, 'half_cheetah'),
    "Mujoco_hopper": partial(Mujoco, 'hopper'),
    "Mujoco_humanoid": partial(Mujoco, 'humanoid'),
    "Mujoco_swimmer": partial(Mujoco, 'swimmer'),
    "Mujoco_walker_2d": partial(Mujoco, 'walker_2d'),
    "Mopta": Mopta,
    "LassoReal_diabetes": partial(LassoReal, 'diabetes'),
    "LassoReal_breast_cancer": partial(LassoReal, 'breast_cancer'),
    "LassoReal_leukemia": partial(LassoReal, 'leukemia'),
    "LassoReal_dna": partial(LassoReal, 'dna'),

    "LassoSynthetic_synt_simple": partial(LassoSynthetic, 'synt_simple'),
    "LassoSynthetic_synt_medium": partial(LassoSynthetic, 'synt_medium'),
    "LassoSynthetic_synt_high": partial(LassoSynthetic, 'synt_high'),
    "LassoSynthetic_synt_hard": partial(LassoSynthetic, 'synt_hard')
}

def execute(fid, method_name, seed, red_dim, sample):
    print(f"Executing {fid} {method_name} {seed} {red_dim} {sample}")
    np.random.seed(42)
    print(method_name, flush=True)
    method_class = methods[method_name]
    weight = method_name in with_weight

    bench = benchs[fid]()
    lb, ub = bench.get_bounds()

    if len(lb) < red_dim:
        return
    x = create_initial_sample(len(lb), sample_type = 'sobol', n=sample, lower_bound=lb, upper_bound=ub, seed=seed)

    y = x.apply(lambda u: bench(torch.from_numpy(u.values))[0], axis = 1)

    reducer = method_class(n_components=red_dim)
    start = time.time()
    start_cpu = time.perf_counter()

    reducer.fit(x, y)
    end_cpu = time.perf_counter()
    end = time.time()

    reducer.save(f"results/{sample}/real/data_reduced/{fid}/{method_name}_{red_dim}_{seed}_{weight}_reducer.pkl")
    print(f"results/{sample}/real/data_reduced/{fid}/{method_name}_{red_dim}_{seed}_{weight}_time.pkl")
    Path(f'results/{sample}/real/data_reduced/{fid}/{method_name}_{red_dim}_{seed}_{weight}_time.pkl').parent.mkdir(parents=True, exist_ok=True)
    with open(
        f"results/{sample}/real/data_reduced/{fid}/{method_name}_{red_dim}_{seed}_{weight}_time.pkl",
        "wb",
    ) as file:
        pickle.dump([start, end, start_cpu, end_cpu], file)


if __name__ == "__main__":
    executor = submitit.AutoExecutor("logs/reduce_dim_real", cluster="slurm")
    executor.update_parameters(timeout_min=24 * 60, cpus_per_task=1, mem_gb=2.8)
    with executor.batch():
        for (sample, fid, method_name, red_dim, seed, reducer) in expected_real_reducers():
            weight = method_weight(method_name)
            result = reducer.with_name(f'{method_name}_{red_dim}_{seed}_{weight}_time.pkl')
            if not os.path.exists(result) and os.path.exists(real_data_path(fid, seed)):
                executor.submit(execute, fid, method_name, seed, red_dim, sample)
