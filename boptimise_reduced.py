from pathlib import Path
from experiment_settings import BASE_DIR, BOP_INITIAL_INPUT_DIMS, BOP_INITIAL_RED_DIM, BOP_INITIAL_SAMPLE, BOP_INIT_SAMPLE, BOP_TOTAL_BUDGET, expected_synthetic_reducers, method_weight
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
import time
import pickle
import submitit
import os
from methods import methods, with_weight
from botorch.models.transforms.input import Normalize
import botorch
warnings.filterwarnings("ignore")
logging.getLogger('evotorch.core').setLevel(logging.ERROR)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
torch.set_num_threads(1)


def ga_settings(method_name: str) -> tuple[int, int]:
    if method_name in {"UMAP", "UMAP_unsupervised", "parametricUMAP", "parametricUMAP_unsupervised", "LocalMAP", "PaCMAP", "ParamPaCMAP"}:
        return 30, 30
    return 100, 100


def bo(
    fid: int,
    iid: int,
    init_sample: int,
    total_budget: int,
    dim: int,
    method_name: str,
    red_dim: int,
    sample: int,
    use_training_initial_sample: bool = False,
):

    logging.getLogger('evotorch.core').setLevel(logging.ERROR)

    f = get_problem(fid, iid, dim)
    method_class = methods[method_name]
    reducer = method_class(n_components=red_dim)

    weight = method_name in with_weight


    reducer.load(f"results/{sample}/data_reduced/{fid}/{dim}/{method_name}_{red_dim}_{iid}_{weight}_reducer.pkl", numpy=True)
    print(f"Starting", flush=True)
    start = time.time()

    bounds = torch.tensor([[-5] * dim, [5] * dim], dtype=torch.float32)
    if use_training_initial_sample:
        from sampling import create_initial_sample
        Xs = torch.tensor(
            create_initial_sample(
                dim, sample_type="sobol", n=sample, lower_bound=-5, upper_bound=5, seed=42
            ).to_numpy(),
            dtype=torch.float32,
        )
    else:
        Xs = draw_sobol_samples(bounds=bounds, n=init_sample, q=1, seed=111).squeeze(1)

    Xs_reduced = torch.tensor(reducer.transform(Xs.numpy())).float()
    Ys = torch.tensor(f(Xs.cpu().numpy()), dtype=torch.float32).unsqueeze(-1)
    sources = []
    ga_popsize, ga_generations = ga_settings(method_name)
    for i in range(total_budget - len(Xs)):
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
                                        popsize=ga_popsize)
                ga.run(ga_generations)
                best = ga.status["best"]

                Xs_reduced = torch.cat([Xs_reduced, torch.tensor(reducer.transform(best.values.cpu().unsqueeze(0).numpy())).float()])
                Xs = torch.cat([Xs, best.values.clone().unsqueeze(0)])
                Ys = torch.cat([Ys, torch.tensor([f(best.values.cpu().numpy())], device='cpu').unsqueeze(0)])
                sources.append(True)
                print(f"iteration {i + 1}/{total_budget - len(Xs)} done", Ys[-1].item(),  time.time() - start, flush=False)

        except botorch.exceptions.ModelFittingError:
            best = torch.rand(dim).float() * 10 - 5
            Xs_reduced = torch.cat([Xs_reduced, torch.tensor(reducer.transform(best.cpu().unsqueeze(0).numpy())).float()])
            print(Xs_reduced.shape)
            Xs = torch.cat([Xs, best.clone().unsqueeze(0)])
            Ys = torch.cat([Ys, torch.tensor([f(best.cpu().numpy())], device='cpu').unsqueeze(0)])
            sources.append(False)
    end = time.time()
    result_dir = f"results/{sample}/{'bop_initial_sample' if use_training_initial_sample else 'bop_reduced'}/{fid}/{dim}"
    os.makedirs(result_dir, exist_ok=True)
    Path(f'{result_dir}/{iid}_{method_name}_{red_dim}_{weight}.pkl').parent.mkdir(parents=True, exist_ok=True)
    with open(
        f"{result_dir}/{iid}_{method_name}_{red_dim}_{weight}.pkl",
        "wb",
    ) as file:
        pickle.dump([Xs_reduced.cpu().numpy(), Xs.cpu().numpy(), Ys.cpu().numpy(), start, end, sources],
            file)


def bo_with_initial_sample(fid: int, iid: int, total_budget: int, dim: int, method_name: str, red_dim: int, sample: int):
    bo(fid, iid, sample, total_budget, dim, method_name, red_dim, sample, use_training_initial_sample=True)

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
    executor = submitit.AutoExecutor("logs/boptimise_reduced", cluster="slurm")
    executor.update_parameters(timeout_min=24 * 60, cpus_per_task=1, mem_gb=2.8)
    with executor.batch():
        for (sample, fid, inpt_dim, method_name, red_dim, instance, reducer) in expected_synthetic_reducers():
            weight = method_weight(method_name)
            result = BASE_DIR / str(sample) / 'bop_reduced' / str(fid) / str(inpt_dim) / f'{instance}_{method_name}_{red_dim}_{weight}.pkl'
            if not os.path.exists(result) and os.path.exists(reducer):
                executor.submit(bo, fid, instance, BOP_INIT_SAMPLE, BOP_TOTAL_BUDGET, inpt_dim, method_name, red_dim, sample)
        for (sample, fid, inpt_dim, method_name, red_dim, instance, reducer) in expected_synthetic_reducers():
            if sample != BOP_INITIAL_SAMPLE or inpt_dim not in BOP_INITIAL_INPUT_DIMS or red_dim != BOP_INITIAL_RED_DIM:
                continue
            weight = method_weight(method_name)
            result = BASE_DIR / str(sample) / 'bop_initial_sample' / str(fid) / str(inpt_dim) / f'{instance}_{method_name}_{red_dim}_{weight}.pkl'
            if not os.path.exists(result) and os.path.exists(reducer):
                executor.submit(bo_with_initial_sample, fid, instance, BOP_TOTAL_BUDGET, inpt_dim, method_name, red_dim, sample)
