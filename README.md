# Benchmarking Dimensionality Reduction Methods

Experiment and analysis code for the JAIR study of dimensionality reduction for
Bayesian optimization, empirical performance modeling, classification, and
clustering. The benchmarks use BBOB functions, real-world optimization problems,
and OpenML datasets.

## Installation

Use Python 3.10 or 3.11 on Linux. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The dependency versions match the experiment environment. Slisemap and
ParamRepulsor sources are bundled, including the Slisemap changes used in the
experiments. [ivis.py](ivis.py) and [parametric_umap.py](parametric_umap.py) contain
the local PyTorch implementations. Install `requirements-real.txt` for the
real-world benchmarks. OpenML requires network access to download datasets and
predefined splits.

Static figures require Chrome or Chromium for Kaleido.

## Code layout

| Files | Purpose |
| --- | --- |
| [methods.py](methods.py), [methods_cs.py](methods_cs.py) | Reducers, weighting, inverse mappings, persistence, and tuning spaces |
| [experiment_settings.py](experiment_settings.py), [sampling.py](sampling.py) | Shared experiment grids, dataset selection, and initial designs |
| [reduce_dim.py](reduce_dim.py), [reduce_dim_real.py](reduce_dim_real.py) | Fit optimization reducers |
| [boptimise_reduced.py](boptimise_reduced.py), [boptimise_reduced_real.py](boptimise_reduced_real.py) | Bayesian optimization in reduced spaces |
| [epm.py](epm.py), [epm_real.py](epm_real.py) | Random forest performance models |
| [ml_reduce.py](ml_reduce.py), [ml_pred.py](ml_pred.py), [ml_cluster.py](ml_cluster.py) | OpenML reduction, classification, and clustering |
| `*_tuning.py` | Reducer tuning and evaluation on held-out problems or datasets |
| [real_world_benchs](real_world_benchs) | Real-world benchmark wrappers and policy files |
| [aggregation](aggregation) | Aggregate results into DuckDB databases |
| [analyse](analyse) | The analysis and figure-generation code used in the paper |

## Run experiments

Run commands from the repository root. Files use relative directories:
`results/` for experiment outputs, `logs/` for submission logs,
`results/aggregation/` for aggregate databases, and `figures/` for figures.
Results and fitted reducers are ordinary pickle files (`.pkl`); reduced OpenML
data uses Parquet files (`.parquet`).

Each experiment file contains its own SLURM submission code and loops over the
corresponding configurations. Activate the environment before running it.
Configure the `executor.update_parameters(...)` call in that file for your
cluster's partition and resource limits. For example:

```bash
python reduce_dim.py
```

Wait for the reducer jobs to finish before submitting downstream tasks:

```bash
python epm.py
python boptimise_reduced.py
```

For OpenML, run `ml_reduce.py` first, then `ml_pred.py` and `ml_cluster.py`.
The loops skip existing output files and submit downstream configurations only
when their required inputs exist. Tuning scripts first submit missing reducer
configurations. Run each tuning script again after those jobs finish to submit
the evaluations using the saved configurations.

A single experiment can also run directly without SLURM:

```bash
python -c "from reduce_dim import execute; execute(fid=1, method_name='PCA_linear', instance=0, red_dim=5, inpt_dim=20, sample=100)"
python -c "from boptimise_reduced import bo; bo(fid=1, iid=0, init_sample=30, total_budget=300, dim=20, method_name='PCA_linear', red_dim=5, sample=100)"
```

The synthetic grid covers BBOB functions 1–24, instances 0–30, input dimensions
20, 40, 60, 120, 240, 500, and 1000, reduced dimensions 5, 10, 20, and 40, and
training samples 100, 200, 500, and 1000. Only dimensions smaller than the input
are evaluated. Default Bayesian optimization uses 30 initial evaluations and
300 total evaluations. Its initial-sample variant initializes optimization from
the reducer's training design. OpenML suites are 271 and 99, with tuning tasks
and method restrictions defined in [experiment_settings.py](experiment_settings.py).

## Real-world inputs

Generate evaluation data with [create_data_real.py](create_data_real.py) before
running `reduce_dim_real.py`, `epm_real.py`, or `boptimise_reduced_real.py`.
Use `python create_data_real.py --help` for benchmark and seed selection.
Additional benchmark inputs are distributed separately:

- SVM uses `CT_slice_X.npy` and `CT_slice_y.npy` in `real_world_benchs/`,
  containing the input matrix and targets of the CT slice localization dataset.
- MOPTA08 needs the executable for the current architecture beside
  [mopta.py](real_world_benchs/mopta.py), with execute permission. Set `TMP` to a
  writable temporary directory before running it.
- MuJoCo uses Gymnasium's MuJoCo environments and the supplied
  [policy files](real_world_benchs/mujuco_policies).
- LassoBench supplies the Lasso datasets.

## Aggregate and analyze

After the corresponding experiments complete, aggregate their results and
produce figures. For example:

```bash
python -c "from aggregation.ml_pred import run_aggregation; run_aggregation()"
python -m analyse.ml_pred
```

Other aggregation modules cover BO, EPM, clustering, timing, tuned reducers,
and the initial-sample BO variant. They also provide SLURM entry points;
configure [aggregation/common.py](aggregation/common.py) and
[aggregation/bop.py](aggregation/bop.py) for your cluster. BO aggregation uses
[analyse/optima.pkl](analyse/optima.pkl).

Generate individual figure families with `python -m analyse.bop`,
`python -m analyse.epm`, `python -m analyse.ml_cluster`, and
`python -m analyse.ml_time`. Figure scripts accept `--aggregation-dir` and
`--output-dir`. Current summaries
use the normalized `BernoulliRBMNorm` variant, displayed as BernoulliRBM.

## Third-party code

[Slisemap](slisemap) and [ParamRepulsor](ParamRepulsor) retain their upstream
licenses. Benchmark module documentation identifies original sources.
