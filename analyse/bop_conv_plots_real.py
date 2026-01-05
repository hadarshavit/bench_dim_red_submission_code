import plotly.express as px
import lzma
import pickle
from methods import methods, with_weight
import numpy as np
import pandas as pd
from ioh import get_problem, ProblemType, ProblemClass
from methods import methods, with_weight
from real_world_benchs.real_world_benchs import *
from functools import partial

benchs = {
    "Rover": Rover,
    "SVM": SVM,
    "Mujoco_ant": partial(Mujoco, "ant"),
    "Mujoco_half_cheetah": partial(Mujoco, "half_cheetah"),
    "Mujoco_hopper": partial(Mujoco, "hopper"),
    "Mujoco_humanoid": partial(Mujoco, "humanoid"),
    "Mujoco_swimmer": partial(Mujoco, "swimmer"),
    "Mujoco_walker_2d": partial(Mujoco, "walker_2d"),
    "Mopta": Mopta,
    "LassoReal_leukemia": partial(LassoReal, "leukemia"),
    "LassoReal_dna": partial(LassoReal, "dna"),
    "LassoSynthetic_synt_simple": partial(LassoSynthetic, "synt_simple"),
    "LassoSynthetic_synt_medium": partial(LassoSynthetic, "synt_medium"),
    "LassoSynthetic_synt_high": partial(LassoSynthetic, "synt_high"),
    "LassoSynthetic_synt_hard": partial(LassoSynthetic, "synt_hard"),
}


for sample in [100, 200, 500, 1000]:
    for red_dim in [5, 10, 20, 40]:
        for fid in benchs.keys():
            if fid == "SVM" and  sample == 200 and red_dim == 10:
                continue
            bench = benchs[fid]()
            lb, ub = bench.get_bounds()
            if len(lb) <= red_dim:
                continue
            df = pd.read_parquet(
                f"benchmark_dim_red/agg_res/bop/{sample}_{red_dim}_{fid}.parquet.xz"
            )
            df = df[df["index"] >= 30]
            df["index"] = df["index"] + 1
            df["0"] = df["0"] - df["0"].min()
            fig = px.line(
                df,
                log_y=False,
                x="index",
                y="0",
                color="method_name",
                color_discrete_map={
                    "PCA_linear": px.colors.qualitative.Plotly[0],
                    "PCA_rbf": px.colors.qualitative.Dark24[5],
                    "PCA_sigmoid": px.colors.qualitative.Safe[10],
                    "PCA_cosine": px.colors.qualitative.Safe[9],
                    "PCA_poly": px.colors.qualitative.Safe[3],
                    "IPCA": px.colors.qualitative.Safe[5],
                    "FastICA": px.colors.qualitative.Plotly[1],
                    "TruncatedSVD": px.colors.qualitative.Plotly[2],
                    "lolP": px.colors.qualitative.Plotly[3],
                    "isomap": px.colors.qualitative.Plotly[4],
                    "lle": px.colors.qualitative.Plotly[5],
                    "pls_reg": px.colors.qualitative.Plotly[6],
                    "DictionaryLearning": px.colors.qualitative.Plotly[7],
                    "sparsePCA": px.colors.qualitative.Plotly[8],
                    "FA": px.colors.qualitative.Plotly[9],
                    "BernoulliRBM": px.colors.qualitative.G10[4],
                    "GaussianRandomProjection": px.colors.qualitative.G10[8],
                    "SparseRandomProjection": px.colors.qualitative.G10[9],
                },
            )
            fig.update_layout(
                title_text=None,
                template="plotly_white",
                showlegend=False,
                legend_title=None,
                width=200,
                height=120,
                margin=dict(l=0, r=0, t=0, b=0),
                font=dict(
                    family="Times New Roman",
                    size=10,
                    color="Black",
                ),
                title_font=dict(
                    family="Times New Roman",
                    size=10,
                    color="Black",
                ),
                xaxis_title="# Evaluations",
                yaxis_title="f - f*",
            )
            fig.update_xaxes(title_font_size=10)
            fig.update_yaxes(title_font_size=10, type="log")
            fig.update_traces(line={"width": 1})
            fig.write_image(
                f"benchmark_dim_red/figures/bop/{sample}_{red_dim}_{fid}.pdf"
            )
