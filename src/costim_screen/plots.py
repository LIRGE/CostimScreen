"""
Visualization functions for screening results.

This module provides plotting functions for visualizing differential motif
effects, including volcano plots for displaying effect sizes and significance
and heatmaps for coefficient visualization across phenotypes.

Functions
---------
volcano_plot
    Create a volcano plot of log fold change vs significance.
coef_heatmap
    Create a heatmap of Z-scores (or coefficients) across phenotypes.
pooled_coef_heatmap
    Create a heatmap of Z-scores for pooled phenotypes (T-subsets and PD1).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

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

    Generates a scatter plot with log₂ fold change on the x-axis and
    -log₁₀(q-value) on the y-axis. Points are colored by direction:
    blue for positive LFC (enriched), red for negative LFC (depleted).
    Optionally labels the most significant points and draws threshold lines.

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

    # Color by direction: blue for positive LFC, red for negative
    colors = np.where(x >= 0, "#3182bd", "#e34a33")  # blue / red

    fig, ax = plt.subplots()
    ax.scatter(x, y, c=colors, s=12, alpha=0.7)

    # Threshold lines (light gray)
    y_line = -np.log10(max(q_thresh, 1e-300))
    ax.axhline(y_line, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(+lfc_thresh, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(-lfc_thresh, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xlabel(r"$\log_2$ fold change")
    ax.set_ylabel(r"$-\log_{10}$ q-value")
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


def coef_heatmap(
    fit,
    motifs: Iterable[str],
    *,
    phenotypes: Sequence[str] = ("Naïve_High", "Naïve_Low", "CM_High", "CM_Low", "EM_High", "EM_Low"),
    value: str = "z",
    cmap: str = "RdBu_r",
    center: float = 0.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
    figsize: Optional[tuple[float, float]] = None,
    outpath: Optional[str | Path] = None,
    dpi: int = 200,
):
    """Create a heatmap of Z-scores or coefficients across phenotypes.

    Extracts motif:phenotype interaction coefficients from the fitted model
    and displays them as a heatmap with motifs on rows and phenotypes on columns.

    Parameters
    ----------
    fit : FitResult
        Fitted model result from :func:`fit_nb_glm_iter_alpha`.
    motifs : iterable of str
        Names of the motif features to include.
    phenotypes : sequence of str, default all 6 phenotypes
        Phenotype levels to include as columns.
    value : str, default "z"
        What to display: "z" for Z-scores (coef/se), "coef" for raw
        coefficients, or "neglog10p" for -log10(p-value).
    cmap : str, default "RdBu_r"
        Matplotlib colormap name.
    center : float, default 0.0
        Value to center the colormap on.
    vmin : float or None, default None
        Minimum value for colormap. If None, determined from data.
    vmax : float or None, default None
        Maximum value for colormap. If None, determined from data.
    title : str or None, default None
        Plot title.
    figsize : tuple of float or None, default None
        Figure size (width, height). If None, auto-sized based on data.
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
    data : pd.DataFrame
        The heatmap data (motifs × phenotypes).

    Examples
    --------
    >>> fig, ax, data = coef_heatmap(
    ...     fit,
    ...     motifs=motif_cols,
    ...     phenotypes=["CM_High", "CM_Low", "EM_High", "EM_Low"],
    ...     value="z",
    ...     title="Motif Z-scores by Phenotype"
    ... )
    """
    from .contrasts import coef_name_for_motif_phenotype

    motifs = list(motifs)
    phenotypes = list(phenotypes)

    # Build data matrix
    rows = []
    for m in motifs:
        row = {"motif": m}
        for ph in phenotypes:
            coef_name = coef_name_for_motif_phenotype(m, ph)
            if coef_name in fit.data_cols:
                coef = fit.res.params[coef_name]
                se = fit.res.bse[coef_name]
                pval = fit.res.pvalues[coef_name]

                if value == "z":
                    row[ph] = coef / se if se > 0 else np.nan
                elif value == "coef":
                    row[ph] = coef
                elif value == "neglog10p":
                    row[ph] = -np.log10(max(pval, 1e-300))
                else:
                    raise ValueError(f"Unknown value='{value}'. Use 'z', 'coef', or 'neglog10p'.")
            else:
                row[ph] = np.nan
        rows.append(row)

    data = pd.DataFrame(rows).set_index("motif")

    # Drop rows that are all NaN
    data = data.dropna(how="all")

    if data.empty:
        raise ValueError("No valid motif-phenotype coefficients found.")

    # Auto-size figure
    if figsize is None:
        n_rows, n_cols = data.shape
        figsize = (max(4, n_cols * 0.8 + 2), max(4, n_rows * 0.3 + 1))

    # Symmetric vmin/vmax around center if not specified
    if vmin is None or vmax is None:
        abs_max = np.nanmax(np.abs(data.values - center))
        if vmin is None:
            vmin = center - abs_max
        if vmax is None:
            vmax = center + abs_max

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        data.values,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    # Axis labels
    ax.set_xticks(range(len(phenotypes)))
    ax.set_xticklabels(phenotypes, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.index, fontsize=8)

    ax.set_xlabel("Phenotype")
    ax.set_ylabel("Motif")

    if title:
        ax.set_title(title)

    # Colorbar
    value_labels = {"z": "Z-score", "coef": "Coefficient", "neglog10p": r"$-\log_{10}$ p-value"}
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(value_labels.get(value, value))

    fig.tight_layout()

    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight")

    return fig, ax, data


def pooled_coef_heatmap(
    fit,
    motifs: Iterable[str],
    *,
    tsubsets: Sequence[str] = ("Naïve", "CM", "EM"),
    pd1_levels: Sequence[str] = ("High", "Low"),
    include_pd1_low: bool = True,
    cmap: str = "RdBu_r",
    center: float = 0.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
    figsize: Optional[tuple[float, float]] = None,
    outpath: Optional[str | Path] = None,
    dpi: int = 200,
):
    """Create a heatmap of Z-scores for pooled phenotypes.

    Computes pooled motif effects by averaging coefficients across:
    - T-subsets (Naïve, CM, EM): averaged over PD1 levels
    - PD1_Low: averaged over T-subsets (optional)

    The Z-score for each pooled effect is computed as the pooled coefficient
    divided by its pooled standard error (using the delta method via the
    covariance matrix).

    Parameters
    ----------
    fit : FitResult
        Fitted model result from :func:`fit_nb_glm_iter_alpha`.
    motifs : iterable of str
        Names of the motif features to include.
    tsubsets : sequence of str, default ("Naïve", "CM", "EM")
        T-cell subset levels.
    pd1_levels : sequence of str, default ("High", "Low")
        PD1 status levels to pool over for T-subset columns.
    include_pd1_low : bool, default True
        Whether to include a PD1_Low column (pooled over T-subsets).
    cmap : str, default "RdBu_r"
        Matplotlib colormap name.
    center : float, default 0.0
        Value to center the colormap on.
    vmin : float or None, default None
        Minimum value for colormap. If None, determined from data.
    vmax : float or None, default None
        Maximum value for colormap. If None, determined from data.
    title : str or None, default None
        Plot title.
    figsize : tuple of float or None, default None
        Figure size (width, height). If None, auto-sized based on data.
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
    data : pd.DataFrame
        The heatmap data (motifs × pooled phenotypes).

    Examples
    --------
    >>> fig, ax, data = pooled_coef_heatmap(
    ...     fit,
    ...     motifs=motif_cols,
    ...     title="Motif Z-scores by Pooled Phenotype"
    ... )
    """
    from .contrasts import coef_name_for_motif_phenotype

    motifs = list(motifs)
    tsubsets = list(tsubsets)
    pd1_levels = list(pd1_levels)

    # Get covariance matrix for proper SE computation
    cov = fit.res.cov_params()
    params = fit.res.params
    cols = fit.data_cols

    def _pooled_z(coef_names: list[str], weights: list[float]) -> float:
        """Compute Z-score for a weighted sum of coefficients."""
        # Check all coefficients exist
        for name in coef_names:
            if name not in cols:
                return np.nan

        # Pooled coefficient: sum of weighted coefficients
        pooled_coef = sum(w * params[name] for w, name in zip(weights, coef_names))

        # Pooled variance via delta method: w' * Cov * w
        indices = [cols.index(name) for name in coef_names]
        w = np.array(weights)
        sub_cov = cov.iloc[indices, indices].values
        pooled_var = w @ sub_cov @ w

        if pooled_var <= 0:
            return np.nan

        return pooled_coef / np.sqrt(pooled_var)

    # Build column names
    column_names = list(tsubsets)
    if include_pd1_low:
        column_names.append("PD1_Low")

    # Build data matrix
    rows = []
    k_pd1 = len(pd1_levels)
    k_tsubset = len(tsubsets)

    for m in motifs:
        row = {"motif": m}

        # T-subset columns (pooled over PD1)
        for ts in tsubsets:
            coef_names = [
                coef_name_for_motif_phenotype(m, f"{ts}_{pd1}")
                for pd1 in pd1_levels
            ]
            weights = [1.0 / k_pd1] * k_pd1
            row[ts] = _pooled_z(coef_names, weights)

        # PD1_Low column (pooled over T-subsets)
        if include_pd1_low:
            coef_names = [
                coef_name_for_motif_phenotype(m, f"{ts}_Low")
                for ts in tsubsets
            ]
            weights = [1.0 / k_tsubset] * k_tsubset
            row["PD1_Low"] = _pooled_z(coef_names, weights)

        rows.append(row)

    data = pd.DataFrame(rows).set_index("motif")

    # Drop rows that are all NaN
    data = data.dropna(how="all")

    if data.empty:
        raise ValueError("No valid motif-phenotype coefficients found.")

    # Auto-size figure
    if figsize is None:
        n_rows, n_cols = data.shape
        figsize = (max(4, n_cols * 0.8 + 2), max(4, n_rows * 0.3 + 1))

    # Symmetric vmin/vmax around center if not specified
    if vmin is None or vmax is None:
        abs_max = np.nanmax(np.abs(data.values - center))
        if vmin is None:
            vmin = center - abs_max
        if vmax is None:
            vmax = center + abs_max

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        data.values,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    # Axis labels
    ax.set_xticks(range(len(column_names)))
    ax.set_xticklabels(column_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.index, fontsize=8)

    ax.set_xlabel("Pooled Phenotype")
    ax.set_ylabel("Motif")

    if title:
        ax.set_title(title)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Z-score")

    fig.tight_layout()

    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight")

    return fig, ax, data