from pathlib import Path
import sys
import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.append(str(REPO_DIR))

from aggregation.common import LEGACY_RBM_METHODS, aggregation_methods


AGGREGATION_DIR = Path("results/aggregation")
FIGURE_DIR = REPO_DIR / "figures"
METHODS, WEIGHTED_METHODS = aggregation_methods()
WEIGHTED_METHODS = set(WEIGHTED_METHODS)
UNSUPERVISED_METHODS = (
    WEIGHTED_METHODS | {"BernoulliRBM", "GaussianRandomProjection", "SparseRandomProjection"}
) & set(METHODS)
SUPERVISED_METHODS = set(METHODS) - UNSUPERVISED_METHODS
PCA_METHODS = {method for method in METHODS if method.startswith("PCA_")} | {"IPCA"}

DISPLAY_NAMES = {
    "BernoulliRBMNorm": "BernoulliRBM",

    "PCA_linear": "PCA (Linear)",
    "PCA_rbf": "PCA (RBF)",
    "PCA_sigmoid": "PCA (Sigmoid)",
    "PCA_cosine": "PCA (Cosine)",
    "PCA_poly": "PCA (Polynomial)",
    "TruncatedSVD": "Truncated SVD",
    "lolP": "LoLP",
    "lle": "LLE",
    "pls_reg": "PLS Regression",
    "GaussianRandomProjection": "Gaussian Random Projection",
    "SparseRandomProjection": "Sparse Random Projection",
    "DictionaryLearning": "Dictionary Learning",
    "latentDA": "Latent DA",
    "UMAP_unsupervised": "UMAP (Unsupervised)",
    "parametric_UMAP": "Parametric UMAP",
    "parametric_UMAP_unsupervised": "Parametric UMAP (Unsupervised)",
    "Autoencoder_2layer": "Autoencoder (2-layer)",
    "DenoisingAutoencoder": "Denoising Autoencoder",
    "DenoisingAutoencoder_2layer": "Denoising Autoencoder (2-layer)",
    "VAE_2layer": "VAE (2-layer)",
    "ivis": "Ivis",
    "slipmap": "Slipmap",
}


COLORS = {
    "PCA_linear": "#636EFA",
    "PCA_rbf": "#222A2A",
    "PCA_sigmoid": "rgb(136, 136, 136)",
    "PCA_cosine": "rgb(102, 17, 0)",
    "PCA_poly": "rgb(17, 119, 51)",
    "IPCA": "rgb(170, 68, 153)",
    "FastICA": "#EF553B",
    "TruncatedSVD": "#00CC96",
    "lle": "#19D3F3",
    "NMF": "#AA0DFE",
    "DictionaryLearning": "#B6E880",
    "FA": "#FECB52",
    "latentDA": "#3283FE",


    "GaussianRandomProjection": "#B82E2E",
    "SparseRandomProjection": "#316395",
    "BernoulliRBMNorm": "#565656",
    "UMAP_unsupervised": "#16FF32",
    "LocalMAP": "#F7E1A0",
    "PaCMAP": "#E2E2E2",
    "ParamPaCMAP": "#1CBE4F",
    "parametric_UMAP_unsupervised": "#DEA0FD",
    "Autoencoder": "#FEAF16",
    "Autoencoder_2layer": "#F8A19F",
    "DenoisingAutoencoder": "#90AD1C",
    "DenoisingAutoencoder_2layer": "#F6222E",
    "VAE": "#1CFFCE",
    "VAE_2layer": "#2ED9FF",
    "lolP": "#AB63FA",
    "pls_reg": "#FF6692",
    "UMAP": "#1C8356",
    "parametric_UMAP": "#C4451C",
    "slipmap": "#FE00FA",
    "ivis": "#325A9B",
}
METHOD_ORDER = tuple(COLORS)


def display_name(method: str) -> str:
    return DISPLAY_NAMES.get(method, method)


def prepare_methods(
    df: pd.DataFrame,
    column: str = "method_name",
    unsupervised_first: bool = False,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    df = df[~df[column].isin(LEGACY_RBM_METHODS)]
    if df.empty:
        raise RuntimeError("The aggregate database query returned no rows")
    df = df.copy()
    present = set(df[column])
    order = METHOD_ORDER if unsupervised_first else METHODS
    ordered = [method for method in order if method in present]
    ordered.extend(sorted(present - set(ordered)))
    df["method"] = df[column].map(display_name)
    labels = [display_name(method) for method in ordered]
    colors = {display_name(method): COLORS.get(method, "#777777") for method in ordered}
    return df, labels, colors


def style_figure(fig, y_title: str, width: int = 300, height: int = 250, show_legend: bool = False) -> None:
    fig.update_layout(
        title_text=None,
        template="plotly_white",
        showlegend=show_legend,
        legend_title=None,
        width=width,
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        font=dict(size=9, color="Black"),
        title_font=dict(size=9, color="Black"),
        xaxis_title=None,
        yaxis_title=y_title,
    )


def box_figure(
    df: pd.DataFrame,
    value: str,
    y_title: str,
    log_y: bool = False,
):
    method_column = "method_name" if "method_name" in df.columns else "method"
    df, order, colors = prepare_methods(
        df,
        column=method_column,
        unsupervised_first=True,
    )
    fig = px.box(
        df,
        x="method",
        y=value,
        color="method",
        log_y=log_y,
        category_orders={"method": order},
        color_discrete_map=colors,
    )
    style_figure(fig, y_title)
    unsupervised_labels = {display_name(method) for method in UNSUPERVISED_METHODS}
    n_unsupervised = sum(method in unsupervised_labels for method in order)
    fig.add_vline(
        x=n_unsupervised - 0.5,
        line_width=1,
        line_dash="dash",
        line_color="#aaaaaa",
    )
    fig.update_layout(
        width=550,
        height=240,
        margin=dict(l=45, r=10, t=25, b=100),
    )
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=order,
        tickmode="array",
        tickvals=order,
        ticktext=order,
        tickangle=-90,
        title=None,
        showticklabels=True,
        automargin=True,
    )
    return fig


def summary_heatmap_figure(
    df: pd.DataFrame,
    scenario: str,
    value: str,
    value_title: str,
    higher_is_better: bool,
):
    method_column = "method_name" if "method_name" in df.columns else "method"
    if df.empty:
        raise RuntimeError("The aggregate database query returned no rows")
    df = df.copy()
    if "rank" not in df.columns:
        df["rank"] = df.groupby(scenario)[value].rank(
            method="average", ascending=not higher_is_better
        )
    summary = (
        df.groupby([method_column, scenario], observed=True)
        .agg(mean=(value, "mean"), mean_rank=("rank", "mean"))
        .reset_index()
    )
    if isinstance(df[scenario].dtype, pd.CategoricalDtype):
        scenarios = [value for value in df[scenario].cat.categories if value in set(df[scenario])]
    else:
        scenarios = list(dict.fromkeys(df[scenario].tolist()))
    summary, labels, _ = prepare_methods(
        summary, column=method_column, unsupervised_first=True
    )
    mean = (
        summary.pivot(index=scenario, columns="method", values="mean")
        .reindex(index=scenarios, columns=labels)
    )

    fig = go.Figure(
        go.Heatmap(
            x=labels,
            y=scenarios,
            z=mean.values,
            coloraxis="coloraxis",
            hovertemplate=(
                "%{x}<br>%{y}<br>Mean " + value_title + " %{z:.4g}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title_text=None,
        template="plotly_white",
        width=1100,
        height=max(380, 38 * len(scenarios) + 235),
        margin=dict(l=75, r=125, t=15, b=220),
        font=dict(size=12, color="Black"),
        coloraxis=dict(
            colorscale="Cividis" if higher_is_better else "Cividis_r",
            cmin=mean.min().min(),
            cmax=mean.max().max(),
            colorbar=dict(
                title=f"Mean {value_title}", thickness=10, len=0.85, tickfont_size=11
            ),
        ),
    )
    fig.update_xaxes(
        categoryorder="array", categoryarray=labels,
        type="category", tickmode="array", tickvals=labels,
        tickangle=-90, automargin=True, tickfont_size=14,
    )
    fig.update_yaxes(
        type="category", autorange="reversed", automargin=True, tickfont_size=12
    )
    return fig


def line_figure(df: pd.DataFrame, x: str, y: str, y_title: str):
    df, order, colors = prepare_methods(df)
    fig = px.line(
        df,
        x=x,
        y=y,
        color="method",
        category_orders={"method": order},
        color_discrete_map=colors,
        render_mode="svg",
    )
    style_figure(fig, y_title, width=200, height=200)
    fig.update_layout(
        xaxis_title="# Evaluations",
        margin=dict(l=30, r=8, t=8, b=25),
    )
    x_min, x_max = float(df[x].min()), float(df[x].max())
    rank_min, rank_max = 1, len(order)
    x_ticks = endpoint_ticks(x_min, x_max)
    rank_ticks = endpoint_ticks(rank_min, rank_max)
    x_padding = max((x_max - x_min) * 0.01, 0.5)
    fig.update_xaxes(
        title_font_size=9,
        range=[x_min - x_padding, x_max + x_padding],
        tickmode="array",
        tickvals=x_ticks,
        ticktext=[format_tick(value) for value in x_ticks],
        automargin=True,
    )
    fig.update_yaxes(
        title_font_size=9,
        range=[rank_min - 0.5, rank_max + 0.5],
        tickmode="array",
        tickvals=rank_ticks,
        ticktext=[format_tick(value) for value in rank_ticks],
        automargin=True,
    )
    by_label = {display_name(method): method for method in METHODS}
    for trace in fig.data:
        method = by_label[trace.name]
        if method in PCA_METHODS:
            symbol, marker_size, width, opacity = "star", 7, 2, 1
        elif method in UNSUPERVISED_METHODS:
            symbol, marker_size, width, opacity = "diamond", 4, 1, 0.8
        else:
            symbol, marker_size, width, opacity = "circle-open", 4, 1, 0.8
        trace.update(
            mode="lines+markers",
            line={"dash": "solid", "width": width},
            marker={"symbol": symbol, "size": marker_size, "maxdisplayed": 9},
            opacity=opacity,
        )
    return fig


def heatmap_figure(df: pd.DataFrame, x: str, value: str):
    df, _, _ = prepare_methods(df)
    x_values = sorted(df[x].unique())
    final = df[df[x] == x_values[-1]].sort_values([value, "method"])
    order = final["method"].tolist()
    matrix = df.pivot(index="method", columns=x, values=value).reindex(order)
    rank_max = len(order)
    display_values = rank_max + 1 - matrix.values
    x_min, x_max = float(x_values[0]), float(x_values[-1])
    x_ticks = endpoint_ticks(x_min, x_max)
    fig = go.Figure(go.Heatmap(
        x=matrix.columns,
        y=matrix.index,
        z=display_values,
        customdata=matrix.values,
        zmin=1,
        zmax=rank_max,
        colorscale="Cividis",
        colorbar=dict(
            title="Mean Rank",
            tickmode="array",
            tickvals=list(range(1, rank_max + 1)),
            ticktext=[str(rank) for rank in range(rank_max, 0, -1)],
            tickfont=dict(size=7),
            thickness=12,
            len=0.95,
        ),
        hovertemplate="%{y}<br>Evaluation %{x}<br>Mean rank %{customdata:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title_text=None,
        template="plotly_white",
        width=420,
        height=500,
        margin=dict(l=150, r=15, t=10, b=35),
        font=dict(size=9, color="Black"),
        xaxis_title="# Evaluations",
        yaxis_title=None,
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=x_ticks,
        ticktext=[format_tick(value) for value in x_ticks],
        automargin=True,
    )
    fig.update_yaxes(autorange="reversed", automargin=True, tickfont_size=8)
    return fig


def endpoint_ticks(minimum: float, maximum: float) -> list[float]:
    if minimum == maximum:
        return [minimum]
    target_step = (maximum - minimum) / 4
    magnitude = 10 ** math.floor(math.log10(target_step))
    step = next(scale * magnitude for scale in (1, 2, 5, 10) if scale * magnitude >= target_step)
    first = math.ceil(minimum / step) * step
    ticks = [minimum]
    value = first
    while value < maximum:
        if value > minimum:
            ticks.append(value)
        value += step
    ticks.append(maximum)
    return ticks


def format_tick(value: float) -> str:
    return f"{value:g}"


def write_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(path)
    print(path, flush=True)


def write_legend(path: Path) -> None:
    fig = go.Figure()
    for name, symbol, width in (
        ("Supervised", "circle-open", 1),
        ("Unsupervised", "diamond", 1),
        ("PCA family", "star", 2),
    ):
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode="lines+markers",
            name=name,
            line={"color": "#555555", "dash": "solid", "width": width},
            marker={"color": "#555555", "symbol": symbol, "size": 6},
            showlegend=True,
            legendrank=0,
        ))
    for method in METHOD_ORDER:
        fig.add_trace(go.Box(
            y=[None],
            name=display_name(method),
            marker_color=COLORS[method],
            line_color=COLORS[method],
            showlegend=True,
            legendrank=1,
        ))
    fig.update_layout(
        title_text=None,
        template="plotly_white",
        width=1000,
        height=200,
        margin=dict(l=0, r=0, t=0, b=0),
        font=dict(size=9, color="Black"),
        showlegend=True,
        legend_title=None,
        legend=dict(
            orientation="h",
            maxheight=1,
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.4,
            font_size=9,
        ),
        legend_tracegroupgap=1,
        legend_indentation=0,
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    write_figure(fig, path)
