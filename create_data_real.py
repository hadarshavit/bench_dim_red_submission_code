import argparse
import os
import pickle
from pathlib import Path

from methods import methods, with_weight
import submitit
from pynisher import limit
from ioh import get_problem, ProblemClass
from sampling import create_initial_sample
from real_world_benchs.real_world_benchs import *
from functools import partial
import torch
import warnings
warnings.filterwarnings("ignore")


REPO_DIR = Path(__file__).resolve().parent
BASE_DIR = Path("results")
DEFAULT_LOG_DIR = Path("logs/logs_real_data")


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


def real_data_path(fid, seed):
    return BASE_DIR / "real_data" / fid / f"{seed}_data.pkl"


def create_data(fid, seed, overwrite=False):
    path = real_data_path(fid, seed)
    if path.exists() and not overwrite:
        return f"exists: {path}"

    os.makedirs(path.parent, exist_ok=True)

    bench = benchs[fid]()
    lb, ub = bench.get_bounds()

    x_orig = create_initial_sample(len(lb), sample_type = 'sobol', n=min(50 * len(ub), 5000), lower_bound=lb, upper_bound=ub, seed=(seed + 1000))

    y = x_orig.apply(lambda u: bench(torch.from_numpy(u.values))[0], axis = 1).values

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(
        path,
        "wb",
    ) as file:
        pickle.dump([x_orig, y], file)

    return f"created: {path}"


def make_executor(args):
    executor = submitit.AutoExecutor(str(args.log_dir), "slurm")
    executor.update_parameters(
        timeout_min=args.timeout_min,
        slurm_partition=args.partition,
        slurm_array_parallelism=args.array_size,
        cpus_per_task=args.cpus,
        mem_gb=args.mem_gb,
        tasks_per_node=1,
        slurm_job_name=args.job_name,
        slurm_setup=[


            f"export PYTHONPATH={REPO_DIR}",
            "export OMP_NUM_THREADS=1",
            "export OPENBLAS_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            "export BLIS_NUM_THREADS=1",
        ],
        slurm_additional_parameters={"account": args.account} if args.account else {},
    )
    return executor


def parse_args():
    parser = argparse.ArgumentParser(description="Create real benchmark data with one submitit job per fid/seed.")
    parser.add_argument("--fids", nargs="+", default=list(benchs), choices=list(benchs))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(31)))
    parser.add_argument("--force", action="store_true", help="Submit and overwrite even if the output file already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print jobs that would be submitted without submitting them.")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--job-name", default="real_data")
    parser.add_argument("--partition", default=None)
    parser.add_argument("--account", default="")
    parser.add_argument("--array-size", type=int, default=800, help="Maximum number of simultaneously running array elements.")
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--mem-gb", type=float, default=2.8)
    parser.add_argument("--timeout-min", type=int, default=24 * 60)
    return parser.parse_args()


def missing_jobs(fids, seeds, force=False):
    jobs = []
    for fid in fids:
        if fid not in benchs:
            raise ValueError(f"Unknown fid: {fid}")
        for seed in seeds:
            path = real_data_path(fid, seed)
            if force or not os.path.exists(path):
                jobs.append((fid, seed))
    return jobs


if __name__ == '__main__':
    args = parse_args()
    jobs_to_submit = missing_jobs(args.fids, args.seeds, args.force)
    if not jobs_to_submit:
        print("No missing real-data jobs.")
        raise SystemExit(0)

    print(f"Real-data jobs to submit: {len(jobs_to_submit)}")
    if args.dry_run:
        for fid, seed in jobs_to_submit:
            print(f"{fid} seed={seed} -> {real_data_path(fid, seed)}")
        raise SystemExit(0)

    executor = make_executor(args)
    fids, seeds = zip(*jobs_to_submit)
    submitted = executor.map_array(create_data, fids, seeds, [args.force] * len(jobs_to_submit))
    job_ids = [job.job_id for job in submitted]
    print(f"Submitted {len(job_ids)} real-data array elements.")
    print(f"First job id: {job_ids[0]}")
    print(f"Last job id: {job_ids[-1]}")
