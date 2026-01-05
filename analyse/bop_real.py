import plotly.express as px
import lzma
import pickle
from methods import methods, with_weight
import numpy as np
import pandas as pd
from ioh import get_problem, ProblemType, ProblemClass
import os
from concurrent.futures import ProcessPoolExecutor
from real_world_benchs.real_world_benchs import *
from functools import partial


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
    "LassoReal_leukemia": partial(LassoReal, 'leukemia'), 
    "LassoReal_dna": partial(LassoReal, 'dna'),
    "LassoSynthetic_synt_simple": partial(LassoSynthetic, 'synt_simple'),
    "LassoSynthetic_synt_medium": partial(LassoSynthetic, 'synt_medium'), 
    "LassoSynthetic_synt_high": partial(LassoSynthetic, 'synt_high'),
    "LassoSynthetic_synt_hard": partial(LassoSynthetic, 'synt_hard')
}

def run(sample, red_dim, fid):
    
    ys_methods = []
    for method_name, method_class in methods.items():
        all_ys = []
        weight = method_name in with_weight
        for instance in range(0, 31):
            with lzma.open(
                    f"/bench_dim_red_claix/new_bench_dim_red/benchmark_dim_red/{sample}/real/bop_reduced/{fid}/{method_name}_{red_dim}_{instance}_{weight}.pkl.xz",
                    "rb",
                ) as file:
                Xs_reduced, Xs, Ys, start, end, status = pickle.load(file)
                Ys = np.minimum.accumulate(Ys.flatten())
                all_ys.append(Ys)
        all_ys = np.array(all_ys)
        all_ys = all_ys.mean(axis=0)
        all_ys = pd.Series(all_ys).reset_index()
        all_ys['method_name'] = method_name
        all_ys['instance'] = instance
        all_ys['fid'] = fid
        all_ys['red_dim'] = red_dim
        all_ys['sample'] = sample
        ys_methods.append(all_ys)
    ys_methods = pd.concat(ys_methods, axis=0)
    ys_methods['rank'] = ys_methods.groupby(['index'])[0].rank(method='min')
    ys_methods.to_parquet(f'benchmark_dim_red/agg_res/bop/{sample}_{red_dim}_{fid}.parquet.xz', compression='lz4')
    print(f"Done with {sample} {red_dim} {fid}", flush=True)


if __name__ == '__main__':
    failed = []
    with ProcessPoolExecutor(max_workers=32) as executor:
        for sample in [100, 200, 500, 1000]:
            for red_dim in [5, 10, 20, 40]:
                for fid in benchs.keys():
                    bench = benchs[fid]()
                    lb, ub = bench.get_bounds()
                    print(red_dim, len(lb), flush=True)
                    if len(lb) > red_dim:
                        try:
                            run(sample, red_dim, fid)
                        except Exception as e:
                            print(f"Error with {sample} {red_dim} {fid}: {e}", flush=True)
                            failed.append((sample, red_dim, fid))

        executor.shutdown(wait=True)
        print(failed)
        
