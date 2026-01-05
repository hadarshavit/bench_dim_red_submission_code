import math
import torch

from botorch.fit import fit_gpytorch_mll_torch, fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from ioh import get_problem, ProblemType, ProblemClass
from botorch.utils.sampling import draw_sobol_samples
from evotorch import Problem
from evotorch.algorithms import GeneticAlgorithm
from evotorch.operators import OnePointCrossOver, GaussianMutation
import warnings
import logging
import lzma
import time
import pickle
import submitit
import os
from methods import methods, with_weight
from botorch.models.transforms.input import Normalize
import botorch
warnings.filterwarnings("ignore")
logging.getLogger('evotorch.core').setLevel(logging.ERROR)
torch.set_num_interop_threads(1)
torch.set_num_threads(1)

def bo(fid:int, iid: int, init_sample: int, total_budget: int, dim: int, method_name: str, red_dim: int, sample: int): 
    
    logging.getLogger('evotorch.core').setLevel(logging.ERROR)

    f = get_problem(fid, iid, dim)
    method_class = methods[method_name]
    reducer = method_class(n_components=red_dim)

    weight = method_name in with_weight
    reducer.load(f"benchmark_dim_red/{sample}/data_reduced/{fid}/{dim}/{method_name}_{red_dim}_{iid}_{weight}_reducer.pkl.xz", numpy=True)
    print(f"Starting", flush=True)
    start = time.time()
    
    bounds = torch.tensor([[-5] * dim, [5] * dim], dtype=torch.float32)
    Xs = draw_sobol_samples(bounds=bounds, n=init_sample, q=1, seed=111).squeeze(1)

    Xs_reduced = torch.tensor(reducer.transform(Xs.numpy())).float()
    Ys = torch.tensor(f(Xs.cpu().numpy()), dtype=torch.float32).unsqueeze(-1)
    sources = []
    for i in range(total_budget - init_sample):
        torch.manual_seed(i)
        gp = SingleTaskGP(Xs_reduced, Ys, input_transform=Normalize(red_dim))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        try:
            fit_gpytorch_mll(mll)

            EI = ExpectedImprovement(gp, torch.min(Ys), maximize=False)
            with torch.no_grad():
                def obj(X: torch.Tensor):
                    with torch.no_grad():
                        cnds = torch.tensor(reducer.transform(X.numpy())).unsqueeze(1).float()
                        return EI(cnds)
                problem = Problem(objective_sense='max', 
                                objective_func=obj,
                                vectorized=True,
                                num_actors=0,
                                solution_length=dim,
                                bounds=bounds,
                                device='cpu',
                                store_solution_stats=True
                                    )

                ga = GeneticAlgorithm(problem=problem,
                                    operators=[
                                            OnePointCrossOver(problem, tournament_size=4),
                                            GaussianMutation(problem, stdev=1.0),
                                        ],
                                        popsize=100)
                ga.run(100)
                best = ga.status["best"] 

                Xs_reduced = torch.cat([Xs_reduced, torch.tensor(reducer.transform(best.values.cpu().unsqueeze(0).numpy())).float()])
                Xs = torch.cat([Xs, best.values.clone().unsqueeze(0)])
                Ys = torch.cat([Ys, torch.tensor([f(best.values.cpu().numpy())], device='cpu').unsqueeze(0)])
                sources.append(True)
                print(f"iteration {i + 1}/{total_budget - init_sample} done", Ys[-1].item(),  time.time() - start, flush=False)

        except botorch.exceptions.ModelFittingError:
            best = torch.rand(dim).float() * 10 - 5
            Xs_reduced = torch.cat([Xs_reduced, torch.tensor(reducer.transform(best.cpu().unsqueeze(0).numpy())).float()])
            print(Xs_reduced.shape)
            Xs = torch.cat([Xs, best.clone().unsqueeze(0)])
            Ys = torch.cat([Ys, torch.tensor([f(best.cpu().numpy())], device='cpu').unsqueeze(0)])
            sources.append(False)
    end = time.time()
    with lzma.open(
        f"benchmark_dim_red/{sample}/bop_reduced/{fid}/{dim}/{iid}_{method_name}_{red_dim}_{weight}.pkl.xz",
        "wb",
    ) as file:
        pickle.dump([Xs_reduced.cpu().numpy(), Xs.cpu().numpy(), Ys.cpu().numpy(), start, end, sources], 
            file)

def execute(fid, inpt_dim, red_dim, sample, total_budget):
    import time
    start = time.time()
    for method_name, method_class in methods.items():
        for instance in range(0, 31):
            os.makedirs(f"benchmark_dim_red/{sample}/bop_reduced/{fid}/{inpt_dim}", exist_ok=True)
            if time.time() - start > 45 * 3600:
                return
            weight = method_name in with_weight
            if os.path.exists(f"benchmark_dim_red/{sample}/bop_reduced/{fid}/{inpt_dim}/{instance}_{method_name}_{red_dim}_{weight}.pkl.xz"):
                continue
            try:
                bo(fid, instance, 30, total_budget, inpt_dim, method_name, red_dim, sample)
            except Exception as e:
                print(e)
                continue


if __name__ == '__main__':
    bo(12, 30, 30, 300, 120, 'SparseRandomProjection', 5, 100)
    exit()
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

    cnt = 0
    all = 0
    for sample in [100, 200, 500, 1000]:
        for fid in range(1, 25):
            for method_name, method_class in methods.items():
                for inpt_dim in [20, 40, 60, 120, 240, 500, 1000]:
                    for instance in range(0, 31):
                        for red_dim in [5, 10, 20, 40]:
                            if inpt_dim > red_dim:

                                all += 1
                                weight = method_name in with_weight
                                if not os.path.exists(f"benchmark_dim_red/{sample}/bop_reduced/{fid}/{inpt_dim}/{instance}_{method_name}_{red_dim}_{weight}.pkl.xz"):
                                    print(f"Missing benchmark_dim_red/{sample}/bop_reduced/{fid}/{inpt_dim}/{instance}_{method_name}_{red_dim}_{weight}.pkl.xz")
                                    cnt += 1
                                    executor.submit(bo, fid, instance, 30, 300, inpt_dim, method_name, red_dim, sample)

