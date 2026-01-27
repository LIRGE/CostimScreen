from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    pvalues : array-like
        Raw p-values.

    Returns
    -------
    np.ndarray
        FDR-adjusted p-values (q-values).
    """
    pvalues = np.asarray(pvalues)
    n = len(pvalues)

    # Handle NaN values
    valid_mask = ~np.isnan(pvalues)
    valid_pvals = pvalues[valid_mask]

    if len(valid_pvals) == 0:
        return np.full_like(pvalues, np.nan)

    # Sort and compute adjusted p-values
    sorted_idx = np.argsort(valid_pvals)
    sorted_pvals = valid_pvals[sorted_idx]

    # BH adjustment: p_adj[i] = min(p[i] * n / rank[i], 1)
    ranks = np.arange(1, len(sorted_pvals) + 1)
    adjusted = sorted_pvals * len(sorted_pvals) / ranks

    # Ensure monotonicity (cumulative minimum from right to left)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    # Unsort
    unsorted = np.zeros_like(sorted_pvals)
    unsorted[sorted_idx] = adjusted

    # Put back into full array
    result = np.full_like(pvalues, np.nan)
    result[valid_mask] = unsorted

    return result


def normalize_phenotype(s: str) -> str:
    """Normalize phenotype-ish strings to a canonical ASCII form.

    - Removes diacritics (Naïve -> Naive)
    - Strips whitespace
    - Normalizes separators to underscore
    - Canonicalizes common tokens:
        * naive/nai -> Naive
        * cm/centralmemory -> CM
        * em/effectormemory -> EM
        * high/hi/pd1high -> High
        * low/lo/pd1low -> Low

    This is intentionally conservative: if the string doesn't look like a
    (Tsubset, PD1) or token we recognize, it just returns the cleaned ASCII.
    """
    # ASCII fold (remove combining marks)
    normalized = unicodedata.normalize("NFKD", str(s))
    ascii_str = "".join(
        c for c in normalized if not unicodedata.category(c).startswith("M")
    ).strip()

    if ascii_str == "":
        return ascii_str

    # Normalize separators
    ascii_str = re.sub(r"[\s\-\/]+", "_", ascii_str)      # spaces/dashes/slashes -> _
    ascii_str = re.sub(r"_+", "_", ascii_str).strip("_")  # collapse __ -> _

    def _canon_token(tok: str) -> str:
        t = tok.strip()
        tl = t.lower()

        # T-subsets
        if tl in {"naive", "nai", "naïve"}:
            return "Naive"
        if tl in {"cm", "centralmemory", "central_memory", "central"}:
            return "CM"
        if tl in {"em", "effectormemory", "effector_memory", "effector"}:
            return "EM"

        # PD1 status (handle common encodings)
        if tl in {"high", "hi", "pd1high", "pd1_hi", "pd1hi"}:
            return "High"
        if tl in {"low", "lo", "pd1low", "pd1_lo", "pd1lo"}:
            return "Low"

        # Leave everything else alone (but keep original casing)
        return t

    toks = ascii_str.split("_")
    toks = [_canon_token(t) for t in toks if t != ""]

    return "_".join(toks)


# =============================================================================
# Dirichlet-Multinomial Visualization Functions
# =============================================================================


def dm_volcano_plot(
    dm_result,
    dm_data,
    phenotype_p: str,
    phenotype_q: str,
    q_thresh: float = 0.10,
    lfc_thresh: float = 0.1,
    title: Optional[str] = None,
    top_n_labels: int = 12,
    outpath: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (8, 6),
    ax: Optional[plt.Axes] = None,
) -> pd.DataFrame:
    """Create volcano plot comparing two phenotypes from DM results.

    The contrast is: phenotype_p - phenotype_q (log-odds scale).
    Uses proper covariance-aware standard errors via wald_contrast().

    Parameters
    ----------
    dm_result : DirichletMultinomialResult
        Fitted DM model result.
    dm_data : DirichletMultinomialData
        Data used for fitting.
    phenotype_p : str
        First phenotype (positive direction).
    phenotype_q : str
        Second phenotype (negative direction).
    q_thresh : float
        FDR threshold for significance coloring.
    lfc_thresh : float
        Log2 fold change threshold for significance.
    title : str, optional
        Plot title.
    top_n_labels : int
        Number of top hits to label.
    outpath : str or Path, optional
        Path to save figure.
    figsize : tuple
        Figure size.
    ax : matplotlib.Axes, optional
        Axes to plot on. If None, creates new figure.

    Returns
    -------
    pd.DataFrame
        Results table with motif, log2FC, pvalue, qvalue.
    """
    records = []

    # Build contrast weights: +1 for phenotype_p, -1 for phenotype_q
    weights = {phenotype_p: 1.0, phenotype_q: -1.0}

    for elm_name in dm_data.elm_names:
        # Use wald_contrast for proper covariance-aware SE
        effect, se, pval = dm_result.wald_contrast(elm_name, weights)

        records.append({
            "motif": elm_name,
            "log2FC": effect / np.log(2),
            "pvalue": pval,
            "z": effect / se if se > 0 else 0,
        })

    res = pd.DataFrame(records)
    res["qvalue"] = bh_fdr(res["pvalue"])
    res["-log10p"] = -np.log10(res["pvalue"].clip(lower=1e-300))

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Color by significance
    sig_mask = (res["qvalue"] < q_thresh) & (abs(res["log2FC"]) > lfc_thresh)

    # Non-significant points
    ax.scatter(
        res.loc[~sig_mask, "log2FC"],
        res.loc[~sig_mask, "-log10p"],
        c="gray", alpha=0.5, s=30, label="NS"
    )

    # Significant points
    colors = np.where(res.loc[sig_mask, "log2FC"] > 0, "firebrick", "steelblue")
    ax.scatter(
        res.loc[sig_mask, "log2FC"],
        res.loc[sig_mask, "-log10p"],
        c=colors, alpha=0.8, s=50, label=f"FDR < {q_thresh}"
    )

    # Threshold lines
    ax.axhline(-np.log10(0.05), color="gray", linestyle="--", alpha=0.5)
    ax.axvline(-lfc_thresh, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(lfc_thresh, color="gray", linestyle="--", alpha=0.5)

    # Labels for top hits
    top_hits = res[sig_mask].nlargest(top_n_labels, "-log10p")
    for _, row in top_hits.iterrows():
        ax.annotate(
            row["motif"].replace("ELM_", ""),
            (row["log2FC"], row["-log10p"]),
            fontsize=8,
            alpha=0.8,
            xytext=(5, 5),
            textcoords="offset points",
        )

    # Use mathtext for subscripts to avoid font warnings
    ax.set_xlabel(rf"$\log_2$ Fold Change ({phenotype_p} vs {phenotype_q})")
    ax.set_ylabel(r"-$\log_{10}$(p-value)")
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{phenotype_p} vs {phenotype_q}")

    plt.tight_layout()

    if outpath:
        fig.savefig(outpath, dpi=300, bbox_inches="tight")

    return res


def dm_pooled_tsubset_contrast(
    dm_result,
    dm_data,
    tsubset_p: str,
    tsubset_q: str,
    q_thresh: float = 0.10,
    lfc_thresh: float = 0.1,
    title: Optional[str] = None,
    top_n_labels: int = 12,
    outpath: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (8, 6),
    ax: Optional[plt.Axes] = None,
) -> pd.DataFrame:
    """Pooled T-subset contrast volcano plot.

    Computes: 0.5 * (tsubset_p_High - tsubset_q_High) + 0.5 * (tsubset_p_Low - tsubset_q_Low)
    Uses proper covariance-aware standard errors.

    Parameters
    ----------
    dm_result : DirichletMultinomialResult
        Fitted DM model result.
    dm_data : DirichletMultinomialData
        Data used for fitting.
    tsubset_p : str
        First T-subset (positive direction): "Naive", "CM", or "EM".
    tsubset_q : str
        Second T-subset (negative direction).
    q_thresh : float
        FDR threshold for significance coloring.
    lfc_thresh : float
        Log2 fold change threshold for significance.
    title : str, optional
        Plot title.
    top_n_labels : int
        Number of top hits to label.
    outpath : str or Path, optional
        Path to save figure.
    figsize : tuple
        Figure size.
    ax : matplotlib.Axes, optional
        Axes to plot on.

    Returns
    -------
    pd.DataFrame
        Results table.
    """
    records = []

    # Build contrast weights: 0.5 for each phenotype of tsubset_p, -0.5 for tsubset_q
    weights = {}
    for pd1 in ["High", "Low"]:
        weights[f"{tsubset_p}_{pd1}"] = 0.5
        weights[f"{tsubset_q}_{pd1}"] = -0.5

    for elm_name in dm_data.elm_names:
        # Use wald_contrast for proper covariance-aware SE
        effect, se, pval = dm_result.wald_contrast(elm_name, weights)

        records.append({
            "motif": elm_name,
            "log2FC": effect / np.log(2),
            "pvalue": pval,
            "z": effect / se if se > 0 else 0,
        })

    res = pd.DataFrame(records)
    res["qvalue"] = bh_fdr(res["pvalue"])
    res["-log10p"] = -np.log10(res["pvalue"].clip(lower=1e-300))

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sig_mask = (res["qvalue"] < q_thresh) & (abs(res["log2FC"]) > lfc_thresh)

    ax.scatter(
        res.loc[~sig_mask, "log2FC"],
        res.loc[~sig_mask, "-log10p"],
        c="gray", alpha=0.5, s=30, label="NS"
    )

    colors = np.where(res.loc[sig_mask, "log2FC"] > 0, "firebrick", "steelblue")
    ax.scatter(
        res.loc[sig_mask, "log2FC"],
        res.loc[sig_mask, "-log10p"],
        c=colors, alpha=0.8, s=50, label=f"FDR < {q_thresh}"
    )

    ax.axhline(-np.log10(0.05), color="gray", linestyle="--", alpha=0.5)
    ax.axvline(-lfc_thresh, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(lfc_thresh, color="gray", linestyle="--", alpha=0.5)

    top_hits = res[sig_mask].nlargest(top_n_labels, "-log10p")
    for _, row in top_hits.iterrows():
        ax.annotate(
            row["motif"].replace("ELM_", ""),
            (row["log2FC"], row["-log10p"]),
            fontsize=8,
            alpha=0.8,
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.set_xlabel(rf"$\log_2$ Fold Change ({tsubset_p} vs {tsubset_q}, pooled PD1)")
    ax.set_ylabel(r"-$\log_{10}$(p-value)")
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{tsubset_p} vs {tsubset_q} (pooled over PD1)")

    plt.tight_layout()

    if outpath:
        fig.savefig(outpath, dpi=300, bbox_inches="tight")

    return res


def dm_pooled_pd1_contrast(
    dm_result,
    dm_data,
    pd1_high: str = "Low",
    pd1_low: str = "High",
    tsubsets: Tuple[str, ...] = ("Naive", "CM", "EM"),
    q_thresh: float = 0.10,
    lfc_thresh: float = 0.1,
    title: Optional[str] = None,
    top_n_labels: int = 12,
    outpath: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (8, 6),
    ax: Optional[plt.Axes] = None,
) -> pd.DataFrame:
    """Pooled PD1 contrast volcano plot.

    Computes average effect across T-subsets.
    Uses proper covariance-aware standard errors.

    Parameters
    ----------
    dm_result : DirichletMultinomialResult
        Fitted DM model result.
    dm_data : DirichletMultinomialData
        Data used for fitting.
    pd1_high : str
        PD1 level for positive direction ("High" or "Low").
    pd1_low : str
        PD1 level for negative direction.
    tsubsets : tuple
        T-subsets to pool over.
    q_thresh : float
        FDR threshold for significance coloring.
    lfc_thresh : float
        Log2 fold change threshold for significance.
    title : str, optional
        Plot title.
    top_n_labels : int
        Number of top hits to label.
    outpath : str or Path, optional
        Path to save figure.
    figsize : tuple
        Figure size.
    ax : matplotlib.Axes, optional
        Axes to plot on.

    Returns
    -------
    pd.DataFrame
        Results table.
    """
    records = []
    n_tsubsets = len(tsubsets)

    # Build contrast weights
    weights = {}
    for ts in tsubsets:
        weights[f"{ts}_{pd1_high}"] = 1.0 / n_tsubsets
        weights[f"{ts}_{pd1_low}"] = -1.0 / n_tsubsets

    for elm_name in dm_data.elm_names:
        effect, se, pval = dm_result.wald_contrast(elm_name, weights)

        records.append({
            "motif": elm_name,
            "log2FC": effect / np.log(2),
            "pvalue": pval,
            "z": effect / se if se > 0 else 0,
        })

    res = pd.DataFrame(records)
    res["qvalue"] = bh_fdr(res["pvalue"])
    res["-log10p"] = -np.log10(res["pvalue"].clip(lower=1e-300))

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sig_mask = (res["qvalue"] < q_thresh) & (abs(res["log2FC"]) > lfc_thresh)

    ax.scatter(
        res.loc[~sig_mask, "log2FC"],
        res.loc[~sig_mask, "-log10p"],
        c="gray", alpha=0.5, s=30, label="NS"
    )

    colors = np.where(res.loc[sig_mask, "log2FC"] > 0, "firebrick", "steelblue")
    ax.scatter(
        res.loc[sig_mask, "log2FC"],
        res.loc[sig_mask, "-log10p"],
        c=colors, alpha=0.8, s=50, label=f"FDR < {q_thresh}"
    )

    ax.axhline(-np.log10(0.05), color="gray", linestyle="--", alpha=0.5)
    ax.axvline(-lfc_thresh, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(lfc_thresh, color="gray", linestyle="--", alpha=0.5)

    top_hits = res[sig_mask].nlargest(top_n_labels, "-log10p")
    for _, row in top_hits.iterrows():
        ax.annotate(
            row["motif"].replace("ELM_", ""),
            (row["log2FC"], row["-log10p"]),
            fontsize=8,
            alpha=0.8,
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.set_xlabel(rf"$\log_2$ Fold Change (PD1 {pd1_high} vs {pd1_low}, pooled T-subsets)")
    ax.set_ylabel(r"-$\log_{10}$(p-value)")
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"PD1 {pd1_high} vs {pd1_low} (pooled over T-subsets)")

    plt.tight_layout()

    if outpath:
        fig.savefig(outpath, dpi=300, bbox_inches="tight")

    return res


def dm_coef_heatmap(
    dm_result,
    dm_data,
    value: str = "z",
    title: Optional[str] = None,
    outpath: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (10, 12),
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Heatmap of ELM effects across phenotypes.

    Parameters
    ----------
    dm_result : DirichletMultinomialResult
        Fitted DM model result.
    dm_data : DirichletMultinomialData
        Data used for fitting.
    value : str
        Value to plot: "z" (z-scores), "effect" (log-odds), or "log2FC".
    title : str, optional
        Plot title.
    outpath : str or Path, optional
        Path to save figure.
    figsize : tuple
        Figure size.
    ax : matplotlib.Axes, optional
        Axes to plot on.

    Returns
    -------
    fig : matplotlib.Figure
    ax : matplotlib.Axes
    df : pd.DataFrame
        Heatmap data.
    """
    non_ref_phenotypes = dm_result.non_ref_phenotypes

    # Build matrix
    data = []
    for elm_name in dm_data.elm_names:
        row = {"ELM": elm_name}

        for phen in non_ref_phenotypes:
            eff, se = dm_result.get_elm_effect(elm_name, phen)
            z = eff / se if se > 0 else 0

            if value == "z":
                row[phen] = z
            elif value == "effect":
                row[phen] = eff
            elif value == "log2FC":
                row[phen] = eff / np.log(2)

        data.append(row)

    df = pd.DataFrame(data).set_index("ELM")

    # Sort by max absolute value
    df = df.loc[df.abs().max(axis=1).sort_values(ascending=False).index]

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    vmax = df.abs().values.max()
    vmin = -vmax

    sns.heatmap(
        df,
        cmap="RdBu_r",
        center=0,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        cbar_kws={"label": value},
    )

    ax.set_xlabel("Phenotype")
    ax.set_ylabel("ELM")

    # Clean up ELM names
    ax.set_yticklabels([t.get_text().replace("ELM_", "") for t in ax.get_yticklabels()], fontsize=8)

    if title:
        ax.set_title(title)

    plt.tight_layout()

    if outpath:
        fig.savefig(outpath, dpi=300, bbox_inches="tight")

    return fig, ax, df


def dm_pooled_heatmap(
    dm_result,
    dm_data,
    include_pd1_low: bool = True,
    title: Optional[str] = None,
    outpath: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (6, 12),
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Heatmap with pooled phenotypes: Naive, CM, EM (pooled over PD1) and optionally PD1_Low.

    Parameters
    ----------
    dm_result : DirichletMultinomialResult
        Fitted DM model result.
    dm_data : DirichletMultinomialData
        Data used for fitting.
    include_pd1_low : bool
        Whether to include pooled PD1_Low column.
    title : str, optional
        Plot title.
    outpath : str or Path, optional
        Path to save figure.
    figsize : tuple
        Figure size.
    ax : matplotlib.Axes, optional
        Axes to plot on.

    Returns
    -------
    fig : matplotlib.Figure
    ax : matplotlib.Axes
    df : pd.DataFrame
        Heatmap data.
    """
    ref_phenotype = dm_result.reference_phenotype
    tsubsets = ["Naive", "CM", "EM"]

    data = []
    for elm_name in dm_data.elm_names:
        row = {"ELM": elm_name}

        # Pooled T-subset effects
        for ts in tsubsets:
            eff_high, se_high = dm_result.get_elm_effect(elm_name, f"{ts}_High") if f"{ts}_High" != ref_phenotype else (0, 0)
            eff_low, se_low = dm_result.get_elm_effect(elm_name, f"{ts}_Low") if f"{ts}_Low" != ref_phenotype else (0, 0)

            pooled_eff = 0.5 * (eff_high + eff_low)
            pooled_se = 0.5 * np.sqrt(se_high**2 + se_low**2)
            pooled_z = pooled_eff / pooled_se if pooled_se > 0 else 0

            row[ts] = pooled_z

        # PD1_Low pooled over T-subsets
        if include_pd1_low:
            pd1_eff = 0
            pd1_var = 0
            for ts in tsubsets:
                eff_low, se_low = dm_result.get_elm_effect(elm_name, f"{ts}_Low") if f"{ts}_Low" != ref_phenotype else (0, 0)
                eff_high, se_high = dm_result.get_elm_effect(elm_name, f"{ts}_High") if f"{ts}_High" != ref_phenotype else (0, 0)
                pd1_eff += (1/3) * (eff_low - eff_high)
                pd1_var += (1/9) * (se_low**2 + se_high**2)

            pd1_se = np.sqrt(pd1_var)
            pd1_z = pd1_eff / pd1_se if pd1_se > 0 else 0
            row["PD1_Low"] = pd1_z

        data.append(row)

    df = pd.DataFrame(data).set_index("ELM")

    # Sort by max absolute value
    df = df.loc[df.abs().max(axis=1).sort_values(ascending=False).index]

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    vmax = df.abs().values.max()
    vmin = -vmax

    sns.heatmap(
        df,
        cmap="RdBu_r",
        center=0,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        cbar_kws={"label": "Z-score"},
    )

    ax.set_xlabel("Pooled Phenotype")
    ax.set_ylabel("ELM")
    ax.set_yticklabels([t.get_text().replace("ELM_", "") for t in ax.get_yticklabels()], fontsize=8)

    if title:
        ax.set_title(title)

    plt.tight_layout()

    if outpath:
        fig.savefig(outpath, dpi=300, bbox_inches="tight")

    return fig, ax, df


def assemble_dm_main_figure(
    results_path: Union[str, Path],
    *,
    volcano_em_vs_cm: str = "volcano_EM_vs_CM_pooledPD1.png",
    volcano_naive_vs_cm: str = "volcano_Naive_vs_CM_pooledPD1.png",
    volcano_naive_vs_em: str = "volcano_Naive_vs_EM_pooledPD1.png",
    volcano_pd1low_vs_high: str = "volcano_PD1Low_vs_PD1High_pooledTsubset.png",
    heatmap_pooled: str = "heatmap_pooled.png",
    out_png: str = "figure_dm_volcanos_plus_heatmap.png",
    out_pdf: Optional[str] = "figure_dm_volcanos_plus_heatmap.pdf",
    dpi: int = 300,
    width_ratios: Tuple[float, float] = (4.0, 1.2),
) -> Path:
    """Assemble main figure from individual panels.

    Parameters
    ----------
    results_path : str or Path
        Directory containing the individual panel images.
    volcano_em_vs_cm : str
        Filename of EM vs CM volcano plot.
    volcano_naive_vs_cm : str
        Filename of Naive vs CM volcano plot.
    volcano_naive_vs_em : str
        Filename of Naive vs EM volcano plot.
    volcano_pd1low_vs_high : str
        Filename of PD1 Low vs High volcano plot.
    heatmap_pooled : str
        Filename of pooled heatmap.
    out_png : str
        Output PNG filename.
    out_pdf : str, optional
        Output PDF filename.
    dpi : int
        Output DPI.
    width_ratios : tuple
        Width ratios for left (volcanos) and right (heatmap) columns.

    Returns
    -------
    Path
        Path to saved PNG figure.
    """
    results_path = Path(results_path)
    paths = {
        "A": results_path / volcano_em_vs_cm,
        "B": results_path / volcano_naive_vs_cm,
        "C": results_path / volcano_naive_vs_em,
        "D": results_path / volcano_pd1low_vs_high,
        "E": results_path / heatmap_pooled,
    }

    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing expected figure files:\n"
            + "\n".join([f"  {k}: {paths[k]}" for k in missing])
        )

    imgs = {k: mpimg.imread(str(p)) for k, p in paths.items()}

    fig = plt.figure(figsize=(14, 8))
    outer = fig.add_gridspec(
        nrows=2, ncols=2,
        width_ratios=list(width_ratios),
        height_ratios=[1, 1],
        wspace=0.02, hspace=0.05,
    )

    left = outer[:, 0].subgridspec(nrows=2, ncols=2, wspace=0.05, hspace=0.08)

    axA = fig.add_subplot(left[0, 0])
    axB = fig.add_subplot(left[0, 1])
    axC = fig.add_subplot(left[1, 0])
    axD = fig.add_subplot(left[1, 1])
    axE = fig.add_subplot(outer[:, 1])

    for ax, key in [(axA, "A"), (axB, "B"), (axC, "C"), (axD, "D"), (axE, "E")]:
        ax.imshow(imgs[key])
        ax.axis("off")
        ax.text(
            0.01, 0.99, key,
            transform=ax.transAxes,
            ha="left", va="top",
            fontweight="bold",
            fontsize=14,
        )

    out_png_path = results_path / out_png
    fig.savefig(out_png_path, dpi=dpi, bbox_inches="tight")
    if out_pdf is not None:
        fig.savefig(results_path / out_pdf, bbox_inches="tight")
    plt.close(fig)

    return out_png_path


# =============================================================================
# Mann-Whitney Analysis Functions
# =============================================================================


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
    """Compute Cliff's delta effect size for two samples.

    Parameters
    ----------
    x, y : array-like
        Two samples to compare.

    Returns
    -------
    delta : float
        Cliff's delta (-1 to 1).
    n_pairs : int
        Number of pairwise comparisons.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Drop NaNs
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    n_x = len(x)
    n_y = len(y)
    if n_x == 0 or n_y == 0:
        return np.nan, 0

    # Vectorized computation
    n_gt = np.sum(x[:, None] > y[None, :])
    n_lt = np.sum(x[:, None] < y[None, :])

    delta = (n_gt - n_lt) / (n_x * n_y)
    return delta, n_x * n_y


def mann_whitney_elm_test(
    residuals: pd.DataFrame,
    elm_assignments: pd.DataFrame,
    elm_name: str,
    elm_col: str = "ELMs (collapsed)",
) -> Dict:
    """Run Mann-Whitney U test comparing CCDs with vs without an ELM.

    Parameters
    ----------
    residuals : pd.DataFrame
        Pearson residuals with CCDs as index and samples as columns.
    elm_assignments : pd.DataFrame
        DataFrame with CCD IDs as index and a column containing lists of ELMs.
    elm_name : str
        Name of the ELM to test.
    elm_col : str
        Column name containing ELM lists.

    Returns
    -------
    dict
        Dictionary with test results.
    """
    # Find CCDs with this ELM
    has_elm = elm_assignments[elm_col].apply(lambda lst: elm_name in lst)
    ccds_with_elm = elm_assignments[has_elm].index
    ccds_without_elm = elm_assignments[~has_elm].index

    # Filter to common CCDs
    ccds_with_elm = ccds_with_elm.intersection(residuals.index)
    ccds_without_elm = ccds_without_elm.intersection(residuals.index)

    if len(ccds_with_elm) == 0 or len(ccds_without_elm) == 0:
        return {
            "ELM": elm_name,
            "n_with": len(ccds_with_elm),
            "n_without": len(ccds_without_elm),
            "u_stat": np.nan,
            "pvalue": np.nan,
            "cliff_delta": np.nan,
            "n_pairs": 0,
            "mean_with": np.nan,
            "mean_without": np.nan,
            "median_with": np.nan,
            "median_without": np.nan,
        }

    # Get residuals as flat arrays
    res_with = residuals.loc[ccds_with_elm].values.flatten()
    res_without = residuals.loc[ccds_without_elm].values.flatten()

    # Remove NaN values
    res_with = res_with[~np.isnan(res_with)]
    res_without = res_without[~np.isnan(res_without)]

    # Mann-Whitney U test
    u_stat, pval = stats.mannwhitneyu(res_with, res_without, alternative="two-sided")

    # Cliff's delta
    delta, n_pairs = cliffs_delta(res_with, res_without)

    return {
        "ELM": elm_name,
        "n_with": len(ccds_with_elm),
        "n_without": len(ccds_without_elm),
        "u_stat": u_stat,
        "pvalue": pval,
        "cliff_delta": delta,
        "n_pairs": n_pairs,
        "mean_with": np.mean(res_with),
        "mean_without": np.mean(res_without),
        "median_with": np.median(res_with),
        "median_without": np.median(res_without),
    }


def mw_volcano_plot(
    results_df: pd.DataFrame,
    q_thresh: float = 0.10,
    delta_thresh: float = 0.1,
    title: Optional[str] = None,
    top_n_labels: int = 12,
    outpath: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (8, 6),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Create volcano plot for Mann-Whitney results.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from Mann-Whitney tests with columns: ELM, pvalue, cliff_delta.
    q_thresh : float
        FDR threshold for significance.
    delta_thresh : float
        Cliff's delta threshold for significance.
    title : str, optional
        Plot title.
    top_n_labels : int
        Number of top hits to label.
    outpath : str or Path, optional
        Path to save figure.
    figsize : tuple
        Figure size.
    ax : matplotlib.Axes, optional
        Axes to plot on.

    Returns
    -------
    matplotlib.Figure
    """
    df = results_df.copy()

    # Compute FDR
    df["qvalue"] = bh_fdr(df["pvalue"])
    df["-log10p"] = -np.log10(df["pvalue"].clip(lower=1e-300))

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Significance mask
    sig_mask = (df["qvalue"] < q_thresh) & (abs(df["cliff_delta"]) > delta_thresh)

    # Non-significant
    ax.scatter(
        df.loc[~sig_mask, "cliff_delta"],
        df.loc[~sig_mask, "-log10p"],
        c="gray", alpha=0.5, s=30, label="NS"
    )

    # Significant
    colors = np.where(df.loc[sig_mask, "cliff_delta"] > 0, "firebrick", "steelblue")
    ax.scatter(
        df.loc[sig_mask, "cliff_delta"],
        df.loc[sig_mask, "-log10p"],
        c=colors, alpha=0.8, s=50, label=f"FDR < {q_thresh}"
    )

    # Threshold lines
    ax.axhline(-np.log10(0.05), color="gray", linestyle="--", alpha=0.5)
    ax.axvline(-delta_thresh, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(delta_thresh, color="gray", linestyle="--", alpha=0.5)

    # Labels
    top_hits = df[sig_mask].nlargest(top_n_labels, "-log10p")
    for _, row in top_hits.iterrows():
        ax.annotate(
            row["ELM"].replace("ELM_", "") if "ELM" in row else str(row.name),
            (row["cliff_delta"], row["-log10p"]),
            fontsize=8,
            alpha=0.8,
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.set_xlabel("Cliff's Delta")
    ax.set_ylabel(r"-$\log_{10}$(p-value)")
    if title:
        ax.set_title(title)

    plt.tight_layout()

    if outpath:
        fig.savefig(outpath, dpi=300, bbox_inches="tight")

    return fig


def compute_pearson_residuals(counts_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson residuals for count data.

    Residual = (observed - expected) / sqrt(expected)
    where expected = row_sum * col_sum / total

    Parameters
    ----------
    counts_df : pd.DataFrame
        Count matrix with samples as columns, observations as rows.

    Returns
    -------
    pd.DataFrame
        Pearson residuals with same shape as input.
    """
    counts = counts_df.values.astype(float)

    # Total, row sums, column sums
    total = counts.sum()
    row_sums = counts.sum(axis=1, keepdims=True)
    col_sums = counts.sum(axis=0, keepdims=True)

    # Expected under independence
    expected = (row_sums * col_sums) / total

    # Pearson residuals
    residuals = (counts - expected) / np.sqrt(expected + 1e-10)

    return pd.DataFrame(
        residuals,
        index=counts_df.index,
        columns=counts_df.columns
    )


def run_mw_analysis_for_phenotype(
    residuals_df: pd.DataFrame,
    elm_df: pd.DataFrame,
    phenotype: str,
    column_pattern: str,
    elm_col: str = "ELMs (collapsed)",
) -> pd.DataFrame:
    """Run Mann-Whitney analysis for all ELMs on a single phenotype.

    Parameters
    ----------
    residuals_df : pd.DataFrame
        Pearson residuals with CCDs as index and samples as columns.
    elm_df : pd.DataFrame
        DataFrame with CCD IDs as index and column containing ELM lists.
    phenotype : str
        Name of the phenotype being analyzed (for labeling).
    column_pattern : str
        Pattern to match column names for this phenotype.
    elm_col : str
        Column name in elm_df containing ELM lists.

    Returns
    -------
    pd.DataFrame
        Results for all ELMs with columns: ELM, phenotype, pvalue, cliff_delta, etc.
    """
    # Filter columns to this phenotype
    pheno_cols = [c for c in residuals_df.columns if column_pattern in c]
    if len(pheno_cols) == 0:
        return pd.DataFrame()

    pheno_residuals = residuals_df[pheno_cols]

    # Get unique ELMs
    all_elms = sorted(set(elm for elms in elm_df[elm_col] for elm in elms))

    results = []
    for elm_name in all_elms:
        result = mann_whitney_elm_test(
            pheno_residuals, elm_df, elm_name, elm_col=elm_col
        )
        result["phenotype"] = phenotype
        results.append(result)

    return pd.DataFrame(results)


def compute_pooled_tsubset_effect(
    mw_results: pd.DataFrame,
    elm_name: str,
    tsubset: str,
) -> Dict:
    """Compute pooled effect for a T-subset (average across PD1 levels).

    Parameters
    ----------
    mw_results : pd.DataFrame
        Full Mann-Whitney results with columns: ELM, phenotype, cliff_delta, pvalue.
    elm_name : str
        Name of the ELM.
    tsubset : str
        T-subset name ("Naive", "CM", or "EM").

    Returns
    -------
    dict
        Pooled effect with keys: ELM, tsubset, cliff_delta, pvalue.
    """
    high_df = mw_results[(mw_results["ELM"] == elm_name) & (mw_results["phenotype"] == f"{tsubset}_High")]
    low_df = mw_results[(mw_results["ELM"] == elm_name) & (mw_results["phenotype"] == f"{tsubset}_Low")]

    if len(high_df) == 0 or len(low_df) == 0:
        return {"ELM": elm_name, "tsubset": tsubset, "cliff_delta": np.nan, "pvalue": np.nan}

    # Average effect
    delta = 0.5 * (high_df["cliff_delta"].values[0] + low_df["cliff_delta"].values[0])

    # Fisher's method for combining p-values
    p1 = high_df["pvalue"].values[0]
    p2 = low_df["pvalue"].values[0]
    if pd.isna(p1) or pd.isna(p2):
        p_combined = np.nan
    else:
        # Fisher's method: -2 * sum(log(p)) ~ chi2(2k)
        chi2_stat = -2 * (np.log(p1 + 1e-300) + np.log(p2 + 1e-300))
        p_combined = 1 - stats.chi2.cdf(chi2_stat, df=4)

    return {
        "ELM": elm_name,
        "tsubset": tsubset,
        "cliff_delta": delta,
        "pvalue": p_combined,
    }


def get_sig_stars(p: float) -> str:
    """Convert p-value to significance stars.

    Parameters
    ----------
    p : float
        P-value (or q-value).

    Returns
    -------
    str
        Significance indicator: "***" (p<0.001), "**" (p<0.01), "*" (p<0.05), or "".
    """
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def create_elm_name_mapping(
    raw_elms: List[str],
    prefix: str = "ELM_",
) -> Dict[str, str]:
    """Create mapping between raw ELM names and patsy-safe names.

    Parameters
    ----------
    raw_elms : list of str
        Raw ELM names (e.g., "14-3-3", "WD40").
    prefix : str
        Prefix for names starting with digits.

    Returns
    -------
    dict
        Mapping from raw names to safe names.
    """
    from .features import make_patsy_safe_columns
    _, mapping = make_patsy_safe_columns(raw_elms, prefix=prefix)
    return mapping
