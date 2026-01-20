"""
Pooled contrast analyses across phenotype groups.

This module provides functions for computing contrasts that pool across
multiple phenotype levels. For example, comparing T-cell subsets while
averaging over PD1 status, or comparing PD1 levels while averaging over
T-cell subsets.

Functions
---------
motif_diff_between_tsubsets_pooled_pd1
    Contrast between T-subsets, pooled over PD1 levels.
motif_diff_between_pd1_pooled_tsubset
    Contrast between PD1 levels, pooled over T-subsets.
motif_contrast_table_tsubset_pooled_pd1
    Per-motif contrast table for T-subset comparison.
motif_contrast_table_pd1_pooled_tsubset
    Per-motif contrast table for PD1 comparison.
volcano_tsubset_pooled_pd1
    Volcano plot for T-subset comparison.
volcano_pd1_pooled_tsubset
    Volcano plot for PD1 comparison.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .contrasts import coef_name_for_motif_phenotype, wald_contrast
from .plots import volcano_plot
from .stats import bh_fdr


def _phenotype_label(tsubset: str, pd1: str) -> str:
    """Construct phenotype label from T-subset and PD1 status."""
    return f"{tsubset}_{pd1}"


def _build_L_from_weights(fit, weights: dict[str, float]) -> np.ndarray:
    """Build contrast vector from coefficient weights.

    Parameters
    ----------
    fit : FitResult
        Fitted model result.
    weights : dict
        Mapping from coefficient name to weight.

    Returns
    -------
    np.ndarray
        Contrast vector of shape (1, n_coefficients).

    Raises
    ------
    KeyError
        If any coefficient names are not found in the model.
    """
    cols = fit.data_cols
    L = np.zeros((1, len(cols)), dtype=float)
    missing = [k for k in weights.keys() if k not in cols]
    if missing:
        raise KeyError(
            f"Missing coefficients in model: {missing[:10]}"
            + (" ..." if len(missing) > 10 else "")
        )

    for name, w in weights.items():
        L[0, cols.index(name)] = float(w)
    return L


def motif_diff_between_tsubsets_pooled_pd1(
    fit,
    motif: str,
    tsubset_p: str,
    tsubset_q: str,
    *,
    pd1_levels: Sequence[str] = ("High", "Low"),
) -> Tuple[np.ndarray, str]:
    """Build contrast for T-subset comparison pooled over PD1 levels.

    Constructs a contrast comparing the average motif effect in tsubset_p
    to the average effect in tsubset_q, where the average is taken over
    all PD1 levels.

    The contrast is::

        (1/k) * sum_i(beta_{motif:tsubset_p_pd1_i}) - (1/k) * sum_i(beta_{motif:tsubset_q_pd1_i})

    where k is the number of PD1 levels.

    Parameters
    ----------
    fit : FitResult
        Fitted model result.
    motif : str
        Name of the motif feature.
    tsubset_p : str
        First T-subset (e.g., "EM").
    tsubset_q : str
        Second T-subset (e.g., "CM").
    pd1_levels : sequence of str, default ("High", "Low")
        PD1 status levels to pool over.

    Returns
    -------
    L : np.ndarray
        Contrast vector.
    name : str
        Descriptive name for the contrast.

    Examples
    --------
    >>> L, name = motif_diff_between_tsubsets_pooled_pd1(
    ...     fit, "ELM_SH3", "EM", "CM"
    ... )
    >>> print(name)
    ELM_SH3: EM - CM (pooled PD1)
    """
    w = {}
    k = len(pd1_levels)
    for pd1 in pd1_levels:
        ph_p = _phenotype_label(tsubset_p, pd1)
        ph_q = _phenotype_label(tsubset_q, pd1)
        w[coef_name_for_motif_phenotype(motif, ph_p)] = +1.0 / k
        w[coef_name_for_motif_phenotype(motif, ph_q)] = -1.0 / k

    L = _build_L_from_weights(fit, w)
    name = f"{motif}: {tsubset_p} - {tsubset_q} (pooled PD1)"
    return L, name


def motif_diff_between_pd1_pooled_tsubset(
    fit,
    motif: str,
    *,
    tsubsets: Sequence[str] = ("Naïve", "CM", "EM"),
    pd1_high: str = "High",
    pd1_low: str = "Low",
) -> Tuple[np.ndarray, str]:
    """Build contrast for PD1 comparison pooled over T-subsets.

    Constructs a contrast comparing the average motif effect in PD1 high
    to the average effect in PD1 low, where the average is taken over
    all T-subsets.

    The contrast is::

        (1/k) * sum_i(beta_{motif:tsubset_i_high}) - (1/k) * sum_i(beta_{motif:tsubset_i_low})

    where k is the number of T-subsets.

    Parameters
    ----------
    fit : FitResult
        Fitted model result.
    motif : str
        Name of the motif feature.
    tsubsets : sequence of str, default ("Naïve", "CM", "EM")
        T-subset levels to pool over.
    pd1_high : str, default "High"
        Label for PD1 high status.
    pd1_low : str, default "Low"
        Label for PD1 low status.

    Returns
    -------
    L : np.ndarray
        Contrast vector.
    name : str
        Descriptive name for the contrast.

    Examples
    --------
    >>> L, name = motif_diff_between_pd1_pooled_tsubset(
    ...     fit, "ELM_SH3", tsubsets=("CM", "EM")
    ... )
    >>> print(name)
    ELM_SH3: PD1 High - Low (pooled Tsubset)
    """
    w = {}
    k = len(tsubsets)
    for ts in tsubsets:
        ph_h = _phenotype_label(ts, pd1_high)
        ph_l = _phenotype_label(ts, pd1_low)
        w[coef_name_for_motif_phenotype(motif, ph_h)] = +1.0 / k
        w[coef_name_for_motif_phenotype(motif, ph_l)] = -1.0 / k

    L = _build_L_from_weights(fit, w)
    name = f"{motif}: PD1 {pd1_high} - {pd1_low} (pooled Tsubset)"
    return L, name


def motif_contrast_table_tsubset_pooled_pd1(
    fit,
    motifs: Iterable[str],
    *,
    tsubset_p: str,
    tsubset_q: str,
    pd1_levels: Sequence[str] = ("High", "Low"),
    adjust: str = "BH",
    log_base: float = 2.0,
) -> pd.DataFrame:
    """Compute per-motif T-subset contrasts pooled over PD1.

    For each motif, computes the pooled contrast between T-subsets,
    converts to log fold change, and applies FDR correction.

    Parameters
    ----------
    fit : FitResult
        Fitted model result.
    motifs : iterable of str
        Names of motif features to test.
    tsubset_p : str
        First T-subset (the "numerator").
    tsubset_q : str
        Second T-subset (the "denominator").
    pd1_levels : sequence of str, default ("High", "Low")
        PD1 levels to pool over.
    adjust : str, default "BH"
        Multiple testing adjustment method.
    log_base : float, default 2.0
        Base for log fold change.

    Returns
    -------
    pd.DataFrame
        Contrast results with columns: motif, contrast, log_effect,
        logFC, pvalue, qvalue, neglog10_q.
    """
    rows = []
    ln_base = math.log(float(log_base))

    for m in motifs:
        L, name = motif_diff_between_tsubsets_pooled_pd1(
            fit, m, tsubset_p, tsubset_q, pd1_levels=pd1_levels
        )
        est_log, pval = wald_contrast(fit, L, name)
        rows.append(
            {
                "motif": m,
                "contrast": f"{tsubset_p}-{tsubset_q} pooled_PD1",
                "log_effect": float(est_log),  # natural log
                "logFC": float(est_log) / ln_base,
                "pvalue": float(pval),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if adjust.upper() in {"BH", "FDR", "BENJAMINI-HOCHBERG"}:
        df["qvalue"] = bh_fdr(df["pvalue"].values)
    else:
        raise ValueError(f"Unknown adjust='{adjust}'. Use 'BH'.")

    df["neglog10_q"] = -np.log10(df["qvalue"].clip(lower=1e-300))
    df = df.sort_values(["qvalue", "pvalue"], ascending=True).reset_index(drop=True)
    return df


def motif_contrast_table_pd1_pooled_tsubset(
    fit,
    motifs: Iterable[str],
    *,
    tsubsets: Sequence[str] = ("Naïve", "CM", "EM"),
    pd1_high: str = "High",
    pd1_low: str = "Low",
    adjust: str = "BH",
    log_base: float = 2.0,
) -> pd.DataFrame:
    """Compute per-motif PD1 contrasts pooled over T-subsets.

    For each motif, computes the pooled contrast between PD1 levels,
    converts to log fold change, and applies FDR correction.

    Parameters
    ----------
    fit : FitResult
        Fitted model result.
    motifs : iterable of str
        Names of motif features to test.
    tsubsets : sequence of str, default ("Naïve", "CM", "EM")
        T-subsets to pool over.
    pd1_high : str, default "High"
        Label for PD1 high.
    pd1_low : str, default "Low"
        Label for PD1 low.
    adjust : str, default "BH"
        Multiple testing adjustment method.
    log_base : float, default 2.0
        Base for log fold change.

    Returns
    -------
    pd.DataFrame
        Contrast results with columns: motif, contrast, log_effect,
        logFC, pvalue, qvalue, neglog10_q.
    """
    rows = []
    ln_base = math.log(float(log_base))

    for m in motifs:
        L, name = motif_diff_between_pd1_pooled_tsubset(
            fit, m, tsubsets=tsubsets, pd1_high=pd1_high, pd1_low=pd1_low
        )
        est_log, pval = wald_contrast(fit, L, name)
        rows.append(
            {
                "motif": m,
                "contrast": f"PD1{pd1_high}-PD1{pd1_low} pooled_Tsubset",
                "log_effect": float(est_log),
                "logFC": float(est_log) / ln_base,
                "pvalue": float(pval),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if adjust.upper() in {"BH", "FDR", "BENJAMINI-HOCHBERG"}:
        df["qvalue"] = bh_fdr(df["pvalue"].values)
    else:
        raise ValueError(f"Unknown adjust='{adjust}'. Use 'BH'.")

    df["neglog10_q"] = -np.log10(df["qvalue"].clip(lower=1e-300))
    df = df.sort_values(["qvalue", "pvalue"], ascending=True).reset_index(drop=True)
    return df


def volcano_tsubset_pooled_pd1(
    fit,
    motifs: Iterable[str],
    *,
    tsubset_p: str,
    tsubset_q: str,
    pd1_levels: Sequence[str] = ("High", "Low"),
    q_thresh: float = 0.10,
    lfc_thresh: float = 1.0,
    title: Optional[str] = None,
    top_n_labels: int = 12,
    outpath: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Generate volcano plot for T-subset comparison pooled over PD1.

    Convenience function that computes the contrast table and generates
    a volcano plot in one call.

    Parameters
    ----------
    fit : FitResult
        Fitted model result.
    motifs : iterable of str
        Names of motif features to test.
    tsubset_p : str
        First T-subset.
    tsubset_q : str
        Second T-subset.
    pd1_levels : sequence of str, default ("High", "Low")
        PD1 levels to pool over.
    q_thresh : float, default 0.10
        Q-value threshold for significance line.
    lfc_thresh : float, default 1.0
        Log fold change threshold for effect size lines.
    title : str or None, default None
        Plot title.
    top_n_labels : int, default 12
        Number of top hits to label.
    outpath : str, Path, or None, default None
        Path to save the figure.

    Returns
    -------
    pd.DataFrame
        The contrast results table.
    """
    tab = motif_contrast_table_tsubset_pooled_pd1(
        fit, motifs, tsubset_p=tsubset_p, tsubset_q=tsubset_q, pd1_levels=pd1_levels
    )
    volcano_plot(
        tab,
        q_thresh=q_thresh,
        lfc_thresh=lfc_thresh,
        title=title or f"{tsubset_p} vs {tsubset_q} (pooled PD1)",
        top_n_labels=top_n_labels,
        outpath=outpath,
    )
    return tab


def volcano_pd1_pooled_tsubset(
    fit,
    motifs: Iterable[str],
    *,
    tsubsets: Sequence[str] = ("Naïve", "CM", "EM"),
    pd1_high: str = "High",
    pd1_low: str = "Low",
    q_thresh: float = 0.10,
    lfc_thresh: float = 1.0,
    title: Optional[str] = None,
    top_n_labels: int = 12,
    outpath: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Generate volcano plot for PD1 comparison pooled over T-subsets.

    Convenience function that computes the contrast table and generates
    a volcano plot in one call.

    Parameters
    ----------
    fit : FitResult
        Fitted model result.
    motifs : iterable of str
        Names of motif features to test.
    tsubsets : sequence of str, default ("Naïve", "CM", "EM")
        T-subsets to pool over.
    pd1_high : str, default "High"
        Label for PD1 high.
    pd1_low : str, default "Low"
        Label for PD1 low.
    q_thresh : float, default 0.10
        Q-value threshold for significance line.
    lfc_thresh : float, default 1.0
        Log fold change threshold for effect size lines.
    title : str or None, default None
        Plot title.
    top_n_labels : int, default 12
        Number of top hits to label.
    outpath : str, Path, or None, default None
        Path to save the figure.

    Returns
    -------
    pd.DataFrame
        The contrast results table.
    """
    tab = motif_contrast_table_pd1_pooled_tsubset(
        fit, motifs, tsubsets=tsubsets, pd1_high=pd1_high, pd1_low=pd1_low
    )
    volcano_plot(
        tab,
        q_thresh=q_thresh,
        lfc_thresh=lfc_thresh,
        title=title or f"PD1 {pd1_high} vs {pd1_low} (pooled Tsubset)",
        top_n_labels=top_n_labels,
        outpath=outpath,
    )
    return tab