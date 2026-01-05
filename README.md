# Benchmarking Dimensionality Reduction Methods

This repository contains the code for benchmarking various dimensionality reduction techniques across different optimization and machine learning tasks, as described in the accompanying JAIR submission.

## Overview

This framework evaluates dimensionality reduction methods on:
- **Synthetic benchmark functions** (BBOB suite)
- **Real-world optimization problems** (Rover, SVM, Mujoco tasks, etc.)
- **Machine learning datasets** (OpenML suite)

The evaluation covers both the quality of reduced representations and their computational efficiency.

## Repository Structure

### Core Files

- [`methods.py`](code/methods.py) - Defines all dimensionality reduction methods and wrapper classes
- [`create_data.py`](code/create_data.py) - Generates initial datasets for benchmarking
- [`reduce_dim.py`](code/reduce_dim.py) - Applies dimensionality reduction to datasets

### Optimization Tasks

- [`boptimise_original.py`](code/boptimise_original.py) - Bayesian optimization on original high-dimensional spaces
- [`boptimise_reduced.py`](code/boptimise_reduced.py) - Bayesian optimization on reduced-dimensional spaces  
- [`boptimise_reduced_real.py`](code/boptimise_reduced_real.py) - Bayesian optimization on real-world problems

### Machine Learning Tasks

- [`ml_reduce.py`](code/ml_reduce.py) - Dimensionality reduction for ML datasets
- [`ml_pred.py`](code/ml_pred.py) - Supervised learning performance evaluation
- [`ml_cluster.py`](code/ml_cluster.py) - Clustering performance evaluation

### Surrogate Modeling

- [`epm.py`](code/epm.py) - Empirical Performance Models using Random Forests

### Analysis

- [`analyse/`](code/analyse/) - Contains Jupyter notebooks and scripts for result analysis

