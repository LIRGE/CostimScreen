"""
Visualization functions for screening results.

This module provides plotting functions for visualizing differential motif
effects, including volcano plots for displaying effect sizes and significance.

Functions
---------
volcano_plot
    Create a volcano plot of log fold change vs significance.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def volcano_plot(
    df: pd.DataFrame,
    *,
    x_col: str = "logFC",
    y_col: str = "neglog10_q",
    label_col: str = "motif",
    q_col: str = "qvalue",
    q_thresh: float = 0.10,
    lfc_thresh: float = 1.0,
    title: Optional[str] = None,
    top_n_labels: int = 10,
    outpath: Optional[str | Path] = None,
    dpi: int = 200,
):
    """Create a volcano plot of log fold change vs significance.

    Generates a scatter plot with log fold change on the x-axis and
    -log10(q-value) on the y-axis. Optionally labels the most significant
    points and draws threshold lines.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing contrast results (from :func:`motif_contrast_table`).
    x_col : str, default "logFC"
        Column name for x-axis values (log fold change).
    y_col : str, default "neglog10_q"
        Column name for y-axis values (-log10 q-value).
    label_col : str, default "motif"
        Column name for point labels.
    q_col : str, default "qvalue"
        Column name for q-values (used to select top labels).
    q_thresh : float, default 0.10
        Q-value significance threshold. A horizontal line is drawn at
        -log10(q_thresh).
    lfc_thresh : float, default 1.0
        Log fold change threshold. Vertical lines are drawn at
        +/- lfc_thresh.
    title : str or None, default None
        Plot title.
    top_n_labels : int, default 10
        Number of most significant points to label.
    outpath : str, Path, or None, default None
        If provided, save the figure to this path.
    dpi : int, default 200
        Resolution for saved figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    ax : matplotlib.axes.Axes
        The matplotlib axes object.

    Raises
    ------
    ValueError
        If the input DataFrame is empty.

    Examples
    --------
    >>> result = motif_contrast_table(fit, motifs, p="EM_High", q="CM_High")
    >>> fig, ax = volcano_plot(
    ...     result,
    ...     q_thresh=0.10,
    ...     lfc_thresh=1.0,
    ...     title="EM vs CM",
    ...     outpath="volcano.png"
    ... )
    """
    if df.empty:
        raise ValueError("volcano_plot received an empty DataFrame.")

    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)

    fig, ax = plt.subplots()
    ax.scatter(x, y, s=12, alpha=0.7)

    # Threshold lines
    y_line = -np.log10(max(q_thresh, 1e-300))
    ax.axhline(y_line, linestyle="--", linewidth=1)
    ax.axvline(+lfc_thresh, linestyle="--", linewidth=1)
    ax.axvline(-lfc_thresh, linestyle="--", linewidth=1)

    ax.set_xlabel(f"{x_col} (typically log2 fold change)")
    ax.set_ylabel(f"{y_col} (typically -log10 q)")
    if title:
        ax.set_title(title)

    # Label top N by qvalue (and finite)
    if top_n_labels and q_col in df.columns:
        sub = (
            df[np.isfinite(df[q_col])]
            .sort_values(q_col, ascending=True)
            .head(int(top_n_labels))
        )
        for _, r in sub.iterrows():
            ax.text(float(r[x_col]), float(r[y_col]), str(r[label_col]), fontsize=8)

    ax.margins(0.05)
    fig.tight_layout()

    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight")

    return fig, ax