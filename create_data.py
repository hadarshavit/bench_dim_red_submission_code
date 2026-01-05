import numpy as np
import os
from ioh import get_problem, ProblemClass
from pflacco.sampling import create_initial_sample
import lzma
import pickle
from concurrent.futures import ProcessPoolExecutor


def execute(fid, inpt_dim, instance, sample):
    print(f'fid: {fid}, inpt_dim: {inpt_dim}, instance: {instance}')
    f = get_problem(fid=fid, dimension=inpt_dim, instance=instance, problem_class=ProblemClass.BBOB)
    x = create_initial_sample(inpt_dim, sample_type = 'sobol', n=sample, lower_bound=-5, upper_bound=5, seed=42)

    y = x.apply(lambda u: f(u), axis = 1)

    with lzma.open(f'/benchmark_dim_red/{sample}/data_orig/{fid}/{inpt_dim}/{instance}.pkl.xz', 'wb') as file:
        pickle.dump((x, y), file)

if __name__ == '__main__':
    sample = 100
    with ProcessPoolExecutor(max_workers=52) as executor:
        for fid in range(1, 25):
            for inpt_dim in [20, 40, 60, 120, 240, 500, 1000]:
                for instance in range(0, 31):
                    os.makedirs(f'/benchmark_dim_red/{sample}/data_orig/{fid}/{inpt_dim}', exist_ok=True)
                    executor.submit(execute, fid, inpt_dim, instance, sample)
        executor.shutdown(wait=True)
