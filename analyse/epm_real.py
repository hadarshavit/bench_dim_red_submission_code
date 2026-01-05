import plotly.express as px
import lzma
import pickle
from methods import methods, with_weight
import numpy as np
import pandas as pd
from ioh import get_problem, ProblemType, ProblemClass
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error
from pflacco.sampling import create_initial_sample
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ProcessPoolExecutor
import os

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


def run(sample, fid, inpt_dim):
    if os.path.exists(f'/benchmark_dim_red/agg_res/epm/{sample}_{fid}_{inpt_dim}.parquet.xz'):
        return
    print(f"Running {sample} {fid} {inpt_dim}")

    for sample in [100, 200, 500, 1000]:
        for fid in benchs.keys():
            if os.path.exists(f'/benchmark_dim_red/agg_res/epm/{sample}_{fid}.parquet.xz'):
                continue
            data = []

            bench = benchs[fid]()
            lb, ub = bench.get_bounds()
            for instance in range(0, 31):
                for red_dim in [5, 10, 20, 40]:
                    print(f"Running {sample} {fid} {inpt_dim} {instance} {red_dim}")
                    if len(lb) < red_dim:
                        continue
                    for method_name, method_class in methods.items():
                        weight = method_name in with_weight
                        for cv in range(5):

                            with lzma.open(
                                f'benchmark_dim_red/{sample}/real/epm/res/{fid}/{method_name}_{red_dim}_{instance}_{weight}_{cv}.pkl.xz',
                                "rb",
                            ) as file:
                                d = pickle.load(file) 
                                data.append([sample, fid, inpt_dim, instance, method_name, red_dim, weight, cv,  d['mse']])
            df = pd.DataFrame(data, columns=['sample', 'fid', 'inpt_dim', 'instance', 'method_name', 'red_dim', 'weight', 'cv', 'mse'])
            df.to_parquet(f'/benchmark_dim_red/agg_res/epm/{sample}_{fid}.parquet.xz', compression='lz4')


if __name__ == '__main__':
    run(None, None, None)
                    
