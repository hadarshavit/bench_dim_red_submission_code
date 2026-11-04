from pathlib import Path
import submitit
from experiment_settings import BASE_DIR, INSTANCES, SYNTHETIC_FIDS, bop_tuning_config_path, method_weight, tuning_configs
import logging
import math
import os
import pickle
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import botorch
import numpy as np
import torch
from botorch.acquisition import ExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.utils.sampling import draw_sobol_samples
from evotorch import Problem
from evotorch.algorithms import GeneticAlgorithm
from evotorch.operators import GaussianMutation, OnePointCrossOver
from gpytorch.mlls import ExactMarginalLogLikelihood
from ioh import ProblemClass, get_problem
from smac import HyperparameterOptimizationFacade, Scenario

from methods import with_weight
from methods_cs import get_configspace, make_method
from sampling import create_initial_sample

logging.getLogger("evotorch.core").setLevel(logging.ERROR)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
torch.set_num_threads(1)

TUNING_FIDS = range(1, 25)
TUNING_INSTANCE = 100
TUNING_N_TRIALS = 30
TUNING_WORKERS = 24
BOP_INIT_SAMPLE = 30
BOP_TOTAL_BUDGET = 300


def tuning_workers() -> int:
    try:
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", TUNING_WORKERS))
    except ValueError:
        cpus = TUNING_WORKERS
    return max(1, min(TUNING_WORKERS, cpus))


def ga_settings(method_name: str) -> tuple[int, int]:
    if method_name in {"UMAP", "UMAP_unsupervised", "parametricUMAP", "parametricUMAP_unsupervised", "LocalMAP", "PaCMAP", "ParamPaCMAP"}:
        return 30, 30
    return 100, 100


def config_path(sample, inpt_dim, method_name, red_dim, weight):
    return f"results/{sample}/bop_tuning/configs/{inpt_dim}/{method_name}_{red_dim}_{weight}.pkl"


def train_reducer(fid, instance, inpt_dim, method_name, red_dim, sample, config):
    f = get_problem(fid=fid, dimension=inpt_dim, instance=instance, problem_class=ProblemClass.BBOB)
    x = create_initial_sample(inpt_dim, sample_type="sobol", n=sample, lower_bound=-5, upper_bound=5, seed=42)
    y = x.apply(lambda u: f(u), axis=1)
    reducer = make_method(method_name, red_dim, config)
    reducer.fit(x, y)
    return reducer


def run_bo(fid, instance, inpt_dim, method_name, red_dim, sample, config):
    f = get_problem(fid=fid, dimension=inpt_dim, instance=instance, problem_class=ProblemClass.BBOB)
    reducer = train_reducer(fid, instance, inpt_dim, method_name, red_dim, sample, config)
    bounds = torch.tensor([[-5] * inpt_dim, [5] * inpt_dim], dtype=torch.float32)
    Xs = draw_sobol_samples(bounds=bounds, n=BOP_INIT_SAMPLE, q=1, seed=111).squeeze(1)
    Xs_reduced = torch.tensor(reducer.transform(Xs.numpy())).float()
    Ys = torch.tensor(f(Xs.cpu().numpy()), dtype=torch.float32).unsqueeze(-1)
    sources = []
    ga_popsize, ga_generations = ga_settings(method_name)
    start = time.time()
    steps = BOP_TOTAL_BUDGET - BOP_INIT_SAMPLE
    for i in range(steps):
        torch.manual_seed(i)
        gp = SingleTaskGP(Xs_reduced, Ys, input_transform=Normalize(red_dim))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        try:
            fit_gpytorch_mll(mll)
            ei = ExpectedImprovement(gp, torch.min(Ys), maximize=False)
            with torch.no_grad():
                def obj(X: torch.Tensor):
                    cnds = torch.tensor(reducer.transform(X.numpy())).unsqueeze(1).float()
                    return ei(cnds)
                problem = Problem(
                    objective_sense="max",
                    objective_func=obj,
                    vectorized=True,
                    num_actors=0,
                    solution_length=inpt_dim,
                    bounds=bounds,
                    device="cpu",
                    store_solution_stats=True,
                )
                ga = GeneticAlgorithm(
                    problem=problem,
                    operators=[OnePointCrossOver(problem, tournament_size=4), GaussianMutation(problem, stdev=1.0)],
                    popsize=ga_popsize,
                )
                ga.run(ga_generations)
                best = ga.status["best"].values.clone()
                sources.append(True)
        except botorch.exceptions.ModelFittingError:
            best = torch.rand(inpt_dim).float() * 10 - 5
            sources.append(False)
        Xs_reduced = torch.cat([Xs_reduced, torch.tensor(reducer.transform(best.cpu().unsqueeze(0).numpy())).float()])
        Xs = torch.cat([Xs, best.cpu().unsqueeze(0)])
        Ys = torch.cat([Ys, torch.tensor([f(best.cpu().numpy())], device="cpu").unsqueeze(0)])
        print(f"iteration {i + 1}/{steps} done", Ys[-1].item(), time.time() - start, flush=False)
    return Xs_reduced.cpu().numpy(), Xs.cpu().numpy(), Ys.cpu().numpy(), sources


def log_regret(fid, inpt_dim, ys):
    f = get_problem(fid=fid, dimension=inpt_dim, instance=TUNING_INSTANCE, problem_class=ProblemClass.BBOB)
    optimum = float(getattr(getattr(f, "optimum", None), "y", 0.0) or 0.0)
    best = float(np.min(ys))
    return math.log10(max(best - optimum, 1e-12))


def score_config(fid, inpt_dim, method_name, red_dim, sample, config):
    _, _, ys, _ = run_bo(fid, TUNING_INSTANCE, inpt_dim, method_name, red_dim, sample, config)
    return log_regret(fid, inpt_dim, ys)


def tune_config(method_name, inpt_dim, red_dim, sample):
    weight = method_name in with_weight
    out = config_path(sample, inpt_dim, method_name, red_dim, weight)
    if os.path.exists(out):
        return
    cs = get_configspace(method_name, inpt_dim, red_dim, sample)
    if len(cs) == 0:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "wb") as file:
            pickle.dump({}, file)
        return

    def train(config, seed):
        cfg = dict(config)
        with ProcessPoolExecutor(max_workers=tuning_workers()) as executor:
            futures = [
                executor.submit(score_config, fid, inpt_dim, method_name, red_dim, sample, cfg)
                for fid in TUNING_FIDS
            ]
            try:
                losses = [future.result() for future in futures]
            except Exception:
                return 1e12
        return float(np.mean(losses))

    with tempfile.TemporaryDirectory(prefix="bop_tuning_") as smac_tmp:
        scenario = Scenario(configspace=cs, output_directory=smac_tmp, deterministic=True, n_trials=TUNING_N_TRIALS, seed=0)
        inc = HyperparameterOptimizationFacade(scenario=scenario, target_function=train, overwrite=True).optimize()

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as file:
        pickle.dump(dict(inc), file)


def bo(fid: int, iid: int, total_budget: int, inpt_dim: int, method_name: str, red_dim: int, sample: int):
    weight = method_name in with_weight
    result = f"results/{sample}/bop_reduced_tuning/{fid}/{inpt_dim}/{iid}_{method_name}_{red_dim}_{weight}.pkl"
    if os.path.exists(result):
        return
    with open(config_path(sample, inpt_dim, method_name, red_dim, weight), "rb") as file:
        config = pickle.load(file)
    global BOP_TOTAL_BUDGET
    old_budget = BOP_TOTAL_BUDGET
    BOP_TOTAL_BUDGET = total_budget
    try:
        start = time.time()
        xs_reduced, xs, ys, sources = run_bo(fid, iid, inpt_dim, method_name, red_dim, sample, config)
        end = time.time()
    finally:
        BOP_TOTAL_BUDGET = old_budget
    os.makedirs(os.path.dirname(result), exist_ok=True)
    with open(result, "wb") as file:
        pickle.dump([xs_reduced, xs, ys, start, end, sources, config], file)


if __name__ == "__main__":
    executor = submitit.AutoExecutor("logs/boptimise_reduced_tuning", cluster="slurm")
    executor.update_parameters(timeout_min=24 * 60, cpus_per_task=1, mem_gb=2.8)
    with executor.batch():
        for (sample, inpt_dim, method_name, red_dim) in tuning_configs():
            if not os.path.exists(bop_tuning_config_path(sample, inpt_dim, method_name, red_dim)):
                executor.submit(tune_config, method_name, inpt_dim, red_dim, sample)
        for (sample, inpt_dim, method_name, red_dim) in tuning_configs():
            weight = method_weight(method_name)
            config = bop_tuning_config_path(sample, inpt_dim, method_name, red_dim)
            for fid in SYNTHETIC_FIDS:
                for instance in INSTANCES:
                    result = BASE_DIR / str(sample) / 'bop_reduced_tuning' / str(fid) / str(inpt_dim) / f'{instance}_{method_name}_{red_dim}_{weight}.pkl'
                    if not os.path.exists(result) and os.path.exists(config):
                        executor.submit(bo, fid, instance, BOP_TOTAL_BUDGET, inpt_dim, method_name, red_dim, sample)
