import plotly.express as px
import lzma
import pickle
from methods import methods, with_weight
import numpy as np
import pandas as pd
from ioh import get_problem, ProblemType, ProblemClass
import os
from concurrent.futures import ProcessPoolExecutor
from boptimise_reduced import bo
import submitit


def run(sample, red_dim, inpt_dim, fid):
    if os.path.exists(f'/benchmark_dim_red/agg_res/bop/{sample}_{inpt_dim}_{red_dim}_{fid}.parquet.xz'):
        return
    
    with open('/benchmark_dim_red/analyse/optima.pkl', 'rb') as file:
        fs = pickle.load(file)

    print(f"Running {sample} {inpt_dim} {red_dim} {fid}")
    ys_methods = []
    for method_name, method_class in methods.items():
        all_ys = []
        weight = method_name in with_weight
        for instance in range(0, 31):
            print(f'{instance} {method_name}', flush=True)
            f = fs[(fid, instance, inpt_dim)]
            try:
                print('start open', flush=True)
                with lzma.open(
                        f"benchmark_dim_red/{sample}/bop_reduced/{fid}/{inpt_dim}/{instance}_{method_name}_{red_dim}_{weight}.pkl.xz",
                        "rb",
                    ) as file:
                    print('opened', f"benchmark_dim_red/{sample}/bop_reduced/{fid}/{inpt_dim}/{instance}_{method_name}_{red_dim}_{weight}.pkl.xz",flush=True)
                    Xs_reduced, Xs, Ys, start, end, status = pickle.load(file)
                    Ys = np.minimum.accumulate(Ys.flatten()) - f
                    all_ys.append(Ys)
            except:
                print(f"benchmark_dim_red/{sample}/bop_reduced/{fid}/{inpt_dim}/{instance}_{method_name}_{red_dim}_{weight}.pkl.xz")
                print(f"Error with {method_name} {sample} {inpt_dim} {red_dim} {fid} {instance}")
                bo(fid, instance, 30, 300, inpt_dim, method_name, red_dim, sample)
                with lzma.open(
                        f"benchmark_dim_red/{sample}/bop_reduced/{fid}/{inpt_dim}/{instance}_{method_name}_{red_dim}_{weight}.pkl.xz",
                        "rb",
                    ) as file:
                    print('opened', f"benchmark_dim_red/{sample}/bop_reduced/{fid}/{inpt_dim}/{instance}_{method_name}_{red_dim}_{weight}.pkl.xz",flush=True)
                    Xs_reduced, Xs, Ys, start, end, status = pickle.load(file)
                    Ys = np.minimum.accumulate(Ys.flatten()) - f
                    all_ys.append(Ys)
        all_ys = np.array(all_ys)
        all_ys = all_ys.mean(axis=0)
        all_ys = pd.Series(all_ys).reset_index()
        all_ys['method_name'] = method_name
        all_ys['instance'] = instance
        all_ys['fid'] = fid
        all_ys['inpt_dim'] = inpt_dim
        all_ys['red_dim'] = red_dim
        all_ys['sample'] = sample
        ys_methods.append(all_ys)
    ys_methods = pd.concat(ys_methods, axis=0)
    ys_methods['rank'] = ys_methods.groupby(['index'])[0].rank(method='min')
    ys_methods.to_parquet(f'/benchmark_dim_red/agg_res/bop/{sample}_{inpt_dim}_{red_dim}_{fid}.parquet.xz', compression='lz4')
    print(f"Done with {sample} {inpt_dim} {red_dim} {fid}", flush=True)


if __name__ == '__main__':
    executor = submitit.AutoExecutor('benchmark_dim_red/logs_bop', 'slurm')
    executor.update_parameters(
        timeout_min=60*48,
        slurm_partition="PARTITION" ,
        slurm_array_parallelism=1024,
        cpus_per_task=1,
        mem_gb=2.8 * 1, 
        tasks_per_node=1,
        slurm_job_name="bop",
        slurm_setup=["module -q load GCCcore/.12.2.0", "module -q load Python/3.10.8", 
                    "export PYTHONPATH=benchmark_dim_red",
                    "export OMP_NUM_THREADS=1",
                    "export OPENBLAS_NUM_THREADS=1",
                    "export MKL_NUM_THREADS=1",
                    "export BLIS_NUM_THREADS=1",
                    "source venvs/bench_dim_red/bin/activate"],
        slurm_additional_parameters={"account": "ACCOUNT"},
    )
    for sample in [100, 200, 500, 1000]:
        for red_dim in [5, 10, 20, 40]:
            for inpt_dim in [20, 40, 60, 120, 240, 500, 1000]:
                    if inpt_dim > red_dim:
                        for fid in range(1, 25):
                            if os.path.exists(f'/benchmark_dim_red/agg_res/bop/{sample}_{inpt_dim}_{red_dim}_{fid}.parquet.xz'):
                                continue
                            executor.submit(run, sample, red_dim, inpt_dim, fid)

        
