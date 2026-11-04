import ast
import os
import random
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Iterable


BASE_DIR = Path("results")


REPO_DIR = Path(__file__).resolve().parent


SAMPLES = (100, 200, 500, 1000)


SYNTHETIC_FIDS = range(1, 25)


INPUT_DIMS = (20, 40, 60, 120, 240, 500, 1000)


RED_DIMS = (5, 10, 20, 40)


INSTANCES = range(0, 31)


CV_FOLDS = range(5)


BOP_TOTAL_BUDGET = 300


BOP_INIT_SAMPLE = 30


SLISEMAP_MAX_SAMPLE = 100


SLIPMAP_MAX_RED_DIM = 5


BOP_INITIAL_SAMPLE = 100


BOP_INITIAL_INPUT_DIMS = (120, 240)


BOP_INITIAL_RED_DIM = 10


TUNING_SAMPLE = 100


TUNING_INPUT_DIMS = (120, 240)


TUNING_RED_DIM = 10


REAL_FID_DIMS = {
    "Rover": 100,
    "SVM": 388,
    "Mujoco_ant": 888,
    "Mujoco_half_cheetah": 102,
    "Mujoco_hopper": 33,
    "Mujoco_humanoid": 6392,
    "Mujoco_swimmer": 16,
    "Mujoco_walker_2d": 102,
    "Mopta": 124,

    "LassoReal_leukemia": 7129,
    "LassoReal_dna": 180,
    "LassoSynthetic_synt_simple": 1000,
    "LassoSynthetic_synt_medium": 1000,
    "LassoSynthetic_synt_high": 1000,
    "LassoSynthetic_synt_hard": 1000,
}


ML_SUITE_IDS = (271, 99)


ML_TUNING_TASK_IDS = (12, 28, 43)


ML_RED_DIMS = (5, 10, 20, 40)


ML_ALGORITHMS = ("rf", "gb", "mlp")


ML_CLUSTER_TUNING_METRICS = ("homogeneity", "completeness")


ML_MAX_INSTANCES = 20_000


ML_SLIPMAP_MAX_INSTANCES = 1_000


ML_TUNED_EVALUATION_DATASETS = 10


ML_TUNED_EVALUATION_SEED = 0


def method_weight(method_name: str) -> bool:
    return method_name in method_names()[1]


@lru_cache(maxsize=1)
def method_names() -> tuple[tuple[str, ...], tuple[str, ...]]:
    tree = ast.parse((REPO_DIR / "methods.py").read_text())
    method_names = []
    weighted = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if "with_weight" in names:
            weighted = ast.literal_eval(node.value)
        if (
            ("methods_sklearn" in names or "methods_extra" in names)
            and isinstance(node.value, ast.Dict)
        ):
            method_names.extend(
                key.value for key in node.value.keys if isinstance(key, ast.Constant)
            )
    return tuple(method_names), tuple(weighted)


def synthetic_reducer_path(
    sample: int, fid: int, inpt_dim: int, method_name: str, red_dim: int, instance: int
) -> Path:
    weight = method_weight(method_name)
    return (
        BASE_DIR
        / str(sample)
        / "data_reduced"
        / str(fid)
        / str(inpt_dim)
        / f"{method_name}_{red_dim}_{instance}_{weight}_reducer.pkl"
    )


def real_reducer_path(sample: int, fid: str, method_name: str, red_dim: int, seed: int) -> Path:
    weight = method_weight(method_name)
    return (
        BASE_DIR
        / str(sample)
        / "real"
        / "data_reduced"
        / fid
        / f"{method_name}_{red_dim}_{seed}_{weight}_reducer.pkl"
    )


def real_data_path(fid: str, seed: int) -> Path:
    return BASE_DIR / "real_data" / fid / f"{seed}_data.pkl"


def epm_tuning_config_path(sample: int, inpt_dim: int, method_name: str, red_dim: int) -> Path:
    weight = method_weight(method_name)
    return BASE_DIR / str(sample) / "epm_tuning" / "configs" / str(inpt_dim) / f"{method_name}_{red_dim}_{weight}.pkl"


def bop_tuning_config_path(sample: int, inpt_dim: int, method_name: str, red_dim: int) -> Path:
    weight = method_weight(method_name)
    return BASE_DIR / str(sample) / "bop_tuning" / "configs" / str(inpt_dim) / f"{method_name}_{red_dim}_{weight}.pkl"


def ml_reduced_path(task_id: int, cv: int, method_name: str, red_dim: int) -> Path:
    return BASE_DIR / "ml_data_reduced" / f"{task_id}_{cv}_{method_name}_{red_dim}.parquet"


def ml_reduce_time_path(task_id: int, cv: int, method_name: str, red_dim: int) -> Path:
    return BASE_DIR / "ml_data_reduced" / f"{task_id}_{cv}_{method_name}_{red_dim}_time.pkl"


def ml_cluster_path(task_id: int, cv: int, method_name: str, red_dim: int) -> Path:
    return BASE_DIR / "ml_data_reduced" / f"{task_id}_{cv}_{method_name}_{red_dim}_cluster.pkl"


def ml_pred_path(task_id: int, cv: int, method_name: str, red_dim: int, algorithm: str) -> Path:
    return BASE_DIR / "ml_data_reduced" / f"{task_id}_{cv}_{method_name}_{red_dim}_{algorithm}_mlpred.pkl"


def ml_pred_tuning_config_path(method_name: str, red_dim: int, algorithm: str) -> Path:
    return BASE_DIR / "ml_tuning" / "configs" / "pred" / f"{method_name}_{red_dim}_{algorithm}.pkl"


def ml_pred_tuning_result_path(
    task_id: int, cv: int, method_name: str, red_dim: int, algorithm: str
) -> Path:
    return BASE_DIR / "ml_tuning" / "pred" / f"{task_id}_{cv}_{method_name}_{red_dim}_{algorithm}_mlpred.pkl"


def ml_cluster_tuning_config_path(method_name: str, red_dim: int, metric: str) -> Path:
    return BASE_DIR / "ml_tuning" / "configs" / "cluster" / f"{method_name}_{red_dim}_{metric}.pkl"


def ml_cluster_tuning_result_path(
    task_id: int, cv: int, method_name: str, red_dim: int, metric: str
) -> Path:
    return BASE_DIR / "ml_tuning" / "cluster" / metric / f"{task_id}_{cv}_{method_name}_{red_dim}_cluster.pkl"


@lru_cache(maxsize=1)
def ml_task_configs() -> tuple[tuple[int, int, int, int], ...]:
    import openml
    from requests.exceptions import RequestException

    def fetch(label, operation):
        delay = 2.0
        for attempt in range(1, 9):
            try:
                return operation()
            except (openml.exceptions.OpenMLServerError, RequestException) as exc:
                if attempt == 8:
                    raise
                print(
                    f"OpenML {label} failed ({type(exc).__name__}); "
                    f"retrying in {delay:.0f}s [{attempt}/8]",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
                delay = min(delay * 2, 30.0)

    task_ids = sorted(
        {
            task_id
            for suite_id in ML_SUITE_IDS
            for task_id in fetch(
                f"suite {suite_id}",
                lambda suite_id=suite_id: openml.study.get_suite(suite_id),
            ).tasks
        }
    )
    configs = []
    for task_id in task_ids:
        task = fetch(
            f"task {task_id}",
            lambda task_id=task_id: openml.tasks.get_task(
                task_id, download_splits=False
            ),
        )
        dataset = fetch(
            f"dataset qualities for task {task_id}",
            lambda task=task: task.get_dataset(
                download_data=False, download_qualities=True
            ),
        )

        n_features = int(dataset.qualities["NumberOfFeatures"]) - 1
        n_instances = int(dataset.qualities["NumberOfInstances"])
        if n_features <= 20 or n_instances > ML_MAX_INSTANCES:
            continue
        _, folds, _ = fetch(
            f"splits for task {task_id}",
            task.get_split_dimensions,
        )
        configs.append((task_id, folds, n_features, n_instances))
    return tuple(configs)


def ml_evaluation_task_configs() -> Iterable[tuple[int, int, int, int]]:
    return (
        config
        for config in ml_task_configs()
        if config[0] not in ML_TUNING_TASK_IDS
    )


@lru_cache(maxsize=1)
def ml_tuned_evaluation_task_configs() -> tuple[tuple[int, int, int, int], ...]:
    configs = tuple(ml_evaluation_task_configs())
    if len(configs) < ML_TUNED_EVALUATION_DATASETS:
        raise RuntimeError(
            f"Need {ML_TUNED_EVALUATION_DATASETS} ML evaluation datasets, "
            f"but only {len(configs)} are eligible"
        )
    return tuple(
        random.Random(ML_TUNED_EVALUATION_SEED).sample(
            configs, ML_TUNED_EVALUATION_DATASETS
        )
    )


def ml_tuning_task_configs() -> tuple[tuple[int, int, int, int], ...]:
    configs = tuple(
        config
        for config in ml_task_configs()
        if config[0] in ML_TUNING_TASK_IDS
    )
    found = {task_id for task_id, _, _, _ in configs}
    missing = set(ML_TUNING_TASK_IDS) - found
    if missing:
        raise RuntimeError(f"Missing configured ML tuning tasks from OpenML suites: {sorted(missing)}")
    return configs


def expected_synthetic_reducers() -> Iterable[tuple[int, int, int, str, int, int, Path]]:
    methods, _ = method_names()
    for sample in SAMPLES:
        for fid in SYNTHETIC_FIDS:
            for inpt_dim in INPUT_DIMS:
                for method_name in methods:
                    for instance in INSTANCES:
                        for red_dim in RED_DIMS:
                            if not method_allowed_for_config(method_name, sample, red_dim):
                                continue
                            if inpt_dim <= red_dim:
                                continue
                            yield (
                                sample,
                                fid,
                                inpt_dim,
                                method_name,
                                red_dim,
                                instance,
                                synthetic_reducer_path(
                                    sample, fid, inpt_dim, method_name, red_dim, instance
                                ),
                            )


def expected_real_reducers() -> Iterable[tuple[int, str, str, int, int, Path]]:
    methods, _ = method_names()
    for sample in SAMPLES:
        for fid in real_fids():
            fid_dim = REAL_FID_DIMS[fid]
            for method_name in methods:
                for seed in INSTANCES:
                    for red_dim in RED_DIMS:
                        if not method_allowed_for_config(method_name, sample, red_dim):
                            continue
                        if fid_dim < red_dim:
                            continue
                        yield (
                            sample,
                            fid,
                            method_name,
                            red_dim,
                            seed,
                            real_reducer_path(sample, fid, method_name, red_dim, seed),
                        )


def method_allowed_for_sample(method_name: str, sample: int) -> bool:
    return method_name != "slisemap" or sample <= SLISEMAP_MAX_SAMPLE


def method_allowed_for_config(method_name: str, sample: int, red_dim: int) -> bool:
    if not method_allowed_for_sample(method_name, sample):
        return False
    if method_name == "slipmap" and red_dim > SLIPMAP_MAX_RED_DIM:
        return False
    return True


def ml_method_allowed_for_config(
    method_name: str, n_instances: int, red_dim: int
) -> bool:
    if method_name == "slisemap":
        return False
    if method_name == "slipmap":
        return (
            n_instances <= ML_SLIPMAP_MAX_INSTANCES
            and red_dim <= SLIPMAP_MAX_RED_DIM
        )
    return True


@lru_cache(maxsize=None)
def ml_method_allowed_for_tuning(method_name: str, red_dim: int) -> bool:
    configs = [
        config
        for config in ml_tuning_task_configs()
        if config[2] > red_dim
    ]
    return bool(configs) and all(
        ml_method_allowed_for_config(method_name, n_instances, red_dim)
        for _, _, _, n_instances in configs
    )


def tuning_configs() -> Iterable[tuple[int, int, str, int]]:
    methods, _ = method_names()
    for inpt_dim in TUNING_INPUT_DIMS:
        for method_name in methods:
            if not method_allowed_for_config(method_name, TUNING_SAMPLE, TUNING_RED_DIM):
                continue
            yield TUNING_SAMPLE, inpt_dim, method_name, TUNING_RED_DIM


def real_fids() -> tuple[str, ...]:
    return (
        "Rover",
        "SVM",
        "Mujoco_ant",
        "Mujoco_half_cheetah",
        "Mujoco_hopper",
        "Mujoco_humanoid",
        "Mujoco_swimmer",
        "Mujoco_walker_2d",
        "Mopta",
        "LassoReal_leukemia",
        "LassoReal_dna",
        "LassoSynthetic_synt_simple",
        "LassoSynthetic_synt_medium",
        "LassoSynthetic_synt_high",
        "LassoSynthetic_synt_hard",
    )
