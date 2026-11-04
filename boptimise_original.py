from pathlib import Path
import math
import torch

from botorch.fit import fit_gpytorch_mll_torch, fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import LogExpectedImprovement
from ioh import get_problem, ProblemType, ProblemClass
from botorch.utils.sampling import draw_sobol_samples
from evotorch import Problem
from evotorch.algorithms import GeneticAlgorithm
from evotorch.operators import OnePointCrossOver, GaussianMutation
import warnings
import logging
import time
import pickle
import submitit
import os
from botorch.models.transforms.input import Normalize
warnings.filterwarnings("ignore")
logging.getLogger('evotorch.core').setLevel(logging.ERROR)


def bo(fid:int, iid: int, init_sample: int, total_budget: int, dim: int):
    f = get_problem(fid, iid, dim)
    start = time.time()

    bounds = torch.tensor([[-5] * dim, [5] * dim], dtype=torch.float32)
    Xs = draw_sobol_samples(bounds=bounds, n=init_sample, q=1, seed=111).squeeze(1)
    Ys = torch.tensor(f(Xs.cpu().numpy()), dtype=torch.float32).unsqueeze(-1)

    for i in range(total_budget - init_sample):
        torch.manual_seed(i)
        gp = SingleTaskGP(Xs, Ys, input_transform=Normalize(dim))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        EI = LogExpectedImprovement(gp, torch.min(Ys), maximize=False)

        with torch.no_grad():
            def obj(X: torch.Tensor):
                return EI(X.unsqueeze(1).clone())
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

            Xs = torch.cat([Xs, best.values.clone().unsqueeze(0)])
            Ys = torch.cat([Ys, torch.tensor([f(best.values.cpu().numpy())], device='cpu').unsqueeze(0)])
            print(f"iteration {i + 1}/{total_budget - init_sample} done", Ys[-1].item(),  time.time() - start, flush=True)
    end = time.time()
    Path(f'results/bo_orig/{fid}/{dim}/{iid}.pkl').parent.mkdir(parents=True, exist_ok=True)
    with open(
        f"results/bo_orig/{fid}/{dim}/{iid}.pkl",
        "wb",
    ) as file:
        pickle.dump([Xs.cpu().numpy(), Ys.cpu().numpy(), start, end],
            file)


if __name__ == "__main__":
    from experiment_settings import SYNTHETIC_FIDS, INSTANCES, INPUT_DIMS, BOP_INIT_SAMPLE, BOP_TOTAL_BUDGET

    executor = submitit.AutoExecutor("logs/boptimise_original", cluster="slurm")
    executor.update_parameters(timeout_min=24 * 60, cpus_per_task=1, mem_gb=2.8)
    with executor.batch():
        for fid in SYNTHETIC_FIDS:
            for iid in INSTANCES:
                for dim in INPUT_DIMS:
                    if not os.path.exists(f"results/bo_orig/{fid}/{dim}/{iid}.pkl"):
                        executor.submit(bo, fid, iid, BOP_INIT_SAMPLE, BOP_TOTAL_BUDGET, dim)
