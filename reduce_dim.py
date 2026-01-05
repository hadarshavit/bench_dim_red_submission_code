import argparse
import lzma
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from ioh import get_problem, ProblemClass
from pflacco.sampling import create_initial_sample
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
    if os.path.exists(f"benchmark_dim_red/{sample}/data_reduced/{fid}/{inpt_dim}/{method_name}_{red_dim}_{instance}_{weight}_time.pkl.xz"):
        print('Already done')
        print(f"benchmark_dim_red/{sample}/data_reduced/{fid}/{inpt_dim}/{method_name}_{red_dim}_{instance}_{weight}_time.pkl.xz")
        reducer = method_class(n_components=red_dim)
        try:
            reducer.load(f"benchmark_dim_red/{sample}/data_reduced/{fid}/{inpt_dim}/{method_name}_{red_dim}_{instance}_{weight}_reducer.pkl.xz")
            print('Loaded')
            return
        except:
            print('Error loading')
            os.system(f'rm -rf benchmark_dim_red/{sample}/data_reduced/{fid}/{inpt_dim}/{method_name}_{red_dim}_{instance}_{weight}_time.pkl.xz')
            os.system(f'rm -rf benchmark_dim_red/{sample}/data_reduced/{fid}/{inpt_dim}/{method_name}_{red_dim}_{instance}_{weight}_reducer.pkl.xz')
        
    f = get_problem(fid=fid, dimension=inpt_dim, instance=instance, problem_class=ProblemClass.BBOB)
    x = create_initial_sample(inpt_dim, sample_type = 'sobol', n=sample, lower_bound=-5, upper_bound=5, seed=42)

    y = x.apply(lambda u: f(u), axis = 1)
    reducer = method_class(n_components=red_dim)
    start = time.time()
    start_cpu = time.perf_counter()
    reducer.fit(x, y)
    end_cpu = time.perf_counter()
    end = time.time()
    with lzma.open(
        f"benchmark_dim_red/{sample}/data_reduced/{fid}/{inpt_dim}/{method_name}_{red_dim}_{instance}_{weight}_time.pkl.xz",
        "wb",
    ) as file:
        pickle.dump([start, end, start_cpu, end_cpu], file)
    reducer.save(f"benchmark_dim_red/{sample}/data_reduced/{fid}/{inpt_dim}/{method_name}_{red_dim}_{instance}_{weight}_reducer.pkl.xz")
    


def parallel_executor(fid, method_name, sample):
    for inpt_dim in [20, 40, 60, 120, 240, 500, 1000]:
        os.makedirs(
            f"benchmark_dim_red/{sample}/data_reduced/{fid}/{inpt_dim}", exist_ok=True
        )
        for instance in range(0, 31):
            for red_dim in [5, 10, 20, 40]:
                if inpt_dim > red_dim:
                    weight = method_name in with_weight
                    if os.path.exists(f"benchmark_dim_red/{sample}/data_reduced/{fid}/{inpt_dim}/{method_name}_{red_dim}_{instance}_{weight}_time.pkl.xz"):
                        continue
                    try:
                        execute(fid, method_name, instance, red_dim, inpt_dim, sample)
                    except Exception as e:
                        print(e)
                        continue
                    


if __name__ == "__main__":
    execute(15, 'isomap', 26, 10, 500, 1000)
    exit()

    

    executor = submitit.AutoExecutor('benchmark_dim_red/logs', 'slurm')
    executor.update_parameters(
        timeout_min=60*48,
        slurm_partition="CLUSTER" ,
        slurm_array_parallelism=1024,
        cpus_per_task=1,
        mem_gb=5,
        tasks_per_node=1,
        slurm_job_name="dim_red",
        slurm_setup=["module -q load GCCcore/.12.2.0", "module -q load Python/3.10.8", 
                    "export PYTHONPATH=benchmark_dim_red",
                    "export OMP_NUM_THREADS=1",
                    "export OPENBLAS_NUM_THREADS=1",
                    "export MKL_NUM_THREADS=1",
                    "export BLIS_NUM_THREADS=1",
                    "source venvs/bench_dim_red/bin/activate"],
        slurm_additional_parameters={"account": "ACCOUNT"},
    )
    from methods import IvisWrapper
    methods = {'ivis': IvisWrapper}

    cnt = 0
    for sample in [100, 200, 500, 1000]:
        for fid in range(1, 25):
            for method_name, method_class in methods.items():
                for inpt_dim in [20, 40, 60, 120, 240, 500, 1000]:
                    for instance in range(0, 31):
                        for red_dim in [5, 10, 20, 40]:
                            if inpt_dim > red_dim:
                                weight = method_name in with_weight
                                if os.path.exists(f"benchmark_dim_red/{sample}/data_reduced/{fid}/{inpt_dim}/{method_name}_{red_dim}_{instance}_{weight}_time.pkl.xz"):
                                    continue
                                executor.submit(execute, fid, method_name, instance, red_dim, inpt_dim, sample)
                                cnt += 1
                                if cnt % 100 == 0:
                                    print(cnt)
                                    import subprocess
                                    import time
                                    cont_sleep = True
                                    while cont_sleep:
                                        out = subprocess.check_output(['squeue', '--me']).decode('utf-8').split('\n')
                                        if len(out) < 800:
                                            cont_sleep = False
                                            break
                                        time.sleep(60)
                                        
                        
