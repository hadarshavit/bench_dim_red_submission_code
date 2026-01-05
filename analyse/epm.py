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

def run(sample, fid, inpt_dim):
    if os.path.exists(f'/benchmark_dim_red/agg_res/epm/{sample}_{fid}_{inpt_dim}.parquet.xz'):
        return
    print(f"Running {sample} {fid} {inpt_dim}")

    data = []
    for instance in range(0, 31):

        for red_dim in [5, 10, 20, 40]:
            if inpt_dim > red_dim:
                for method_name, method_class in methods.items():
                    weight = method_name in with_weight
                    for cv in range(5):

                        with lzma.open(
                            f'benchmark_dim_red/{sample}/epm/res/{fid}/{inpt_dim}/{instance}_{method_name}_{red_dim}_{weight}_{cv}.pkl.xz',
                            "rb",
                        ) as file:
                            d = pickle.load(file) 
                            data.append([sample, fid, inpt_dim, instance, method_name, red_dim, weight, cv,  d['mse']])
    df = pd.DataFrame(data, columns=['sample', 'fid', 'inpt_dim', 'instance', 'method_name', 'red_dim', 'weight', 'cv', 'mse'])
    df.to_parquet(f'/benchmark_dim_red/agg_res/epm/{sample}_{fid}_{inpt_dim}.parquet.xz', compression='lz4')


if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=10) as executor:
        for sample in [100, 200, 500, 1000]:
            for fid in range(1, 25):
                for inpt_dim in [20, 40, 60, 120, 240, 500, 1000]:
                    
                        j = executor.submit(run, sample, fid, inpt_dim)
        executor.shutdown(wait=True)