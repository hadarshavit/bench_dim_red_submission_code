from pathlib import Path
from experiment_settings import BASE_DIR, BOP_INIT_SAMPLE, BOP_TOTAL_BUDGET, expected_real_reducers, method_weight
import math
import torch

from botorch.fit import fit_gpytorch_mll_torch, fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from ioh import get_problem, ProblemType, ProblemClass
from botorch.utils.sampling import draw_sobol_samples, sample_hypersphere
from evotorch import Problem
from evotorch.algorithms import GeneticAlgorithm
from evotorch.operators import OnePointCrossOver, GaussianMutation
import warnings
import logging
import time
import pickle
import submitit
import os
from methods import methods, with_weight
from botorch.models.transforms.input import Normalize
from real_world_benchs.real_world_benchs import *
from functools import partial
import botorch
warnings.filterwarnings("ignore")
logging.getLogger('evotorch.core').setLevel(logging.ERROR)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
torch.set_num_threads(1)


def ga_settings(method_name: str) -> tuple[int, int]:
    if method_name in {"UMAP", "UMAP_unsupervised", "parametricUMAP", "parametricUMAP_unsupervised", "ParamPaCMAP"}:
        return 30, 30
    return 100, 100


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
    "LassoReal_rcv1": partial(LassoReal, 'rcv1'),
    "LassoSynthetic_synt_simple": partial(LassoSynthetic, 'synt_simple'),
    "LassoSynthetic_synt_medium": partial(LassoSynthetic, 'synt_medium'),
    "LassoSynthetic_synt_high": partial(LassoSynthetic, 'synt_high'),
    "LassoSynthetic_synt_hard": partial(LassoSynthetic, 'synt_hard')
}


def bo(fid: str, seed: int, init_sample: int, total_budget: int, method_name: str, red_dim: int, sample: int):
    print(f"Starting {fid} {seed} {method_name} {red_dim} {sample}", flush=True)
    logging.getLogger('evotorch.core').setLevel(logging.ERROR)

    bench = benchs[fid]()
    lb, ub = bench.get_bounds()

    print(f"lb: {len(lb)}, ub: {len(ub)}", flush=True)
    if len(lb) < red_dim:
        return

    method_class = methods[method_name]
    reducer = method_class(n_components=red_dim)

    weight = method_name in with_weight
    if os.path.exists(f"results/{sample}/real/bop_reduced/{fid}/{method_name}_{red_dim}_{seed}_{weight}.pkl"):
        return
    reducer.load(f"results/{sample}/real/data_reduced/{fid}/{method_name}_{red_dim}_{seed}_{weight}_reducer.pkl", numpy=True)
    print(f"Starting", flush=True)
    start = time.time()
    bounds = torch.tensor([lb, ub], dtype=torch.float32)
    Xs = draw_sobol_samples(bounds=bounds, n=init_sample, q=1, seed=seed + 5000).squeeze(1)

    Xs_reduced = torch.tensor(reducer.transform(Xs.numpy())).float()
    Ys = torch.stack([
        torch.from_numpy(bench(x_i)).float() for i, x_i in enumerate(torch.unbind(Xs, dim=0), 0)
    ], dim=0).squeeze(0)
    print(Ys.shape, 'a')


    sources = []
    ga_popsize, ga_generations = ga_settings(method_name)
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
                                solution_length=len(lb),
                                bounds=bounds,
                                device='cpu',
                                store_solution_stats=True
                                    )

                ga = GeneticAlgorithm(problem=problem,
                                    operators=[
                                            OnePointCrossOver(problem, tournament_size=4),
                                            GaussianMutation(problem, stdev=1.0),
                                        ],
                                        popsize=ga_popsize)
                ga.run(ga_generations)
                best = ga.status["best"]

                Xs_reduced = torch.cat([Xs_reduced, torch.tensor(reducer.transform(best.values.cpu().unsqueeze(0).numpy())).float()])
                Xs = torch.cat([Xs, best.values.clone().unsqueeze(0)])

                Ys = torch.cat([Ys, torch.from_numpy(bench(best.values.clone())).float().unsqueeze(0)])

                sources.append(True)
                print(f"iteration {i + 1}/{total_budget - init_sample} done", Ys[-1].item(),  time.time() - start, flush=False)

        except botorch.exceptions.ModelFittingError:
            best = draw_sobol_samples(bounds=bounds, n=1, q=1, seed=seed + 5000 + i).squeeze(0).squeeze(0)
            Xs_reduced = torch.cat([Xs_reduced, torch.tensor(reducer.transform(best.cpu().unsqueeze(0).numpy())).float()])
            print(Xs_reduced.shape)
            Xs = torch.cat([Xs, best.clone().unsqueeze(0)])
            Ys = torch.cat([Ys, torch.from_numpy(bench(best.cpu())).float().unsqueeze(0)])

            sources.append(False)
            print(f"iteration {i + 1}/{total_budget - init_sample} done", Ys[-1].item(),  time.time() - start, flush=False)
    end = time.time()
    Path(f'results/{sample}/real/bop_reduced/{fid}/{method_name}_{red_dim}_{seed}_{weight}.pkl').parent.mkdir(parents=True, exist_ok=True)
    with open(
        f"results/{sample}/real/bop_reduced/{fid}/{method_name}_{red_dim}_{seed}_{weight}.pkl",
        "wb",
    ) as file:
        pickle.dump([Xs_reduced.cpu().numpy(), Xs.cpu().numpy(), Ys.cpu().numpy(), start, end, sources],
            file)

def execute(fid, inpt_dim, red_dim, sample, total_budget):
    import time
    start = time.time()
    for method_name, method_class in methods.items():
        for instance in range(0, 31):
            os.makedirs(f"results/{sample}/bop_reduced/{fid}/{inpt_dim}", exist_ok=True)
            if time.time() - start > 45 * 3600:
                return
            weight = method_name in with_weight
            if os.path.exists(f"results/{sample}/bop_reduced/{fid}/{inpt_dim}/{instance}_{method_name}_{red_dim}_{weight}.pkl"):
                continue
            try:
                bo(fid, instance, 30, total_budget, inpt_dim, method_name, red_dim, sample)
            except Exception as e:
                print(e)
                continue


if __name__ == "__main__":
    executor = submitit.AutoExecutor("logs/boptimise_reduced_real", cluster="slurm")
    executor.update_parameters(timeout_min=24 * 60, cpus_per_task=1, mem_gb=2.8)
    with executor.batch():
        for (sample, fid, method_name, red_dim, seed, reducer) in expected_real_reducers():
            weight = method_weight(method_name)
            result = BASE_DIR / str(sample) / 'real' / 'bop_reduced' / fid / f'{method_name}_{red_dim}_{seed}_{weight}.pkl'
            if not os.path.exists(result) and os.path.exists(reducer):
                executor.submit(bo, fid, seed, BOP_INIT_SAMPLE, BOP_TOTAL_BUDGET, method_name, red_dim, sample)
