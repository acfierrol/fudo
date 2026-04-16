"""Standard EDA visualization functions."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_missing_matrix(df: pd.DataFrame, save_path: Path | None = None) -> plt.Figure:
    """Heatmap of missing values."""
    fig, ax = plt.subplots(figsize=(12, max(4, len(df.columns) * 0.3)))
    sns.heatmap(df.isnull().T, cbar=False, cmap="Reds", ax=ax)
    ax.set_title("Missing Values Matrix")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_distributions(df: pd.DataFrame, save_path: Path | None = None) -> plt.Figure:
    """Histograms for all numeric columns."""
    numeric = df.select_dtypes(include="number")
    n_cols = len(numeric.columns)
    if n_cols == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No numeric columns", ha="center", va="center")
        return fig

    ncols = min(3, n_cols)
    nrows = (n_cols + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = [axes] if n_cols == 1 else axes.flat

    for i, col in enumerate(numeric.columns):
        numeric[col].dropna().hist(bins=30, ax=axes[i], edgecolor="white")
        axes[i].set_title(col)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Numeric Distributions", y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, save_path: Path | None = None) -> plt.Figure:
    """Correlation heatmap for numeric columns."""
    corr = df.select_dtypes(include="number").corr()
    fig, ax = plt.subplots(figsize=(max(8, len(corr) * 0.6), max(6, len(corr) * 0.5)))
    sns.heatmap(corr, annot=len(corr) <= 15, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_categorical_bars(
    df: pd.DataFrame, top_n: int = 15, save_path: Path | None = None
) -> plt.Figure:
    """Bar plots for categorical columns."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No categorical columns", ha="center", va="center")
        return fig

    ncols = min(2, len(cat_cols))
    nrows = (len(cat_cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows))
    axes = [axes] if len(cat_cols) == 1 else axes.flat

    for i, col in enumerate(cat_cols):
        counts = df[col].value_counts().head(top_n)
        counts.plot.barh(ax=axes[i])
        axes[i].set_title(col)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Categorical Distributions", y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
