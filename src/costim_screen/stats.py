"""
Statistical utilities for hypothesis testing and multiple comparison correction.

This module provides functions for FDR correction and computing per-motif
contrast tables with adjusted p-values.

Functions
---------
bh_fdr
    Benjamini-Hochberg FDR adjustment.
motif_contrast_table
    Compute per-motif contrasts between phenotypes with FDR correction.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from .contrasts import motif_diff_between_phenotypes, wald_contrast


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment.

    Adjusts p-values to control the false discovery rate using the
    Benjamini-Hochberg procedure.

    Parameters
    ----------
    pvals : array-like
        Raw p-values.

    Returns
    -------
    qvals : np.ndarray
        BH-adjusted q-values, same shape as pvals. Q-values represent
        the minimum FDR at which the corresponding hypothesis would
        be rejected.

    Notes
    -----
    The procedure ranks p-values and computes q_i = p_i * n / rank_i,
    then enforces monotonicity (q_i >= q_{i-1} for sorted p-values).

    Examples
    --------
    >>> pvals = np.array([0.001, 0.01, 0.05, 0.1])
    >>> qvals = bh_fdr(pvals)
    >>> qvals
    array([0.004, 0.02 , 0.067, 0.1  ])
    """
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    qvals = np.full(n, np.nan, dtype=float)

    ok = np.isfinite(pvals)
    if ok.sum() == 0:
        return qvals

    p = pvals[ok]
    order = np.argsort(p)
    ranks = np.arange(1, p.size + 1)

    q = p[order] * p.size / ranks
    # enforce monotonicity
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)

    out = np.full_like(pvals, np.nan, dtype=float)
    out_idx = np.where(ok)[0][order]
    out[out_idx] = q
    qvals = out
    return qvals


def motif_contrast_table(
    fit,
    motifs: Iterable[str],
    p: str,
    q: str,
    *,
    adjust: str = "BH",
    log_base: float = 2.0,
    keep_missing: bool = False,
) -> pd.DataFrame:
    """Compute per-motif contrasts between phenotypes with FDR correction.

    For each motif, computes the Wald contrast between phenotypes p and q,
    converts to log fold change, and applies multiple testing correction.

    Parameters
    ----------
    fit : FitResult
        Fitted model result from :func:`fit_nb_glm_iter_alpha`.
    motifs : iterable of str
        Names of the motif features to test.
    p : str
        First phenotype (the "numerator" in the comparison).
    q : str
        Second phenotype (the "denominator" in the comparison).
    adjust : str, default "BH"
        Multiple testing adjustment method. Currently only "BH"
        (Benjamini-Hochberg) is supported.
    log_base : float, default 2.0
        Base for log fold change calculation (2.0 gives log2FC).
    keep_missing : bool, default False
        If True, include rows for motifs with missing coefficients
        (with NaN values). If False, skip them silently.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - motif: motif name
        - phenotype_p: first phenotype
        - phenotype_q: second phenotype
        - log_effect: natural log effect size (beta_p - beta_q)
        - logFC: log fold change in specified base
        - pvalue: raw Wald p-value
        - qvalue: FDR-adjusted q-value
        - neglog10_q: -log10(qvalue) for volcano plots

        Sorted by qvalue (most significant first).

    Notes
    -----
    The log_effect represents the difference in log-scale coefficients,
    which corresponds to the log of the fold change in expected counts
    due to the motif.

    Examples
    --------
    >>> result = motif_contrast_table(
    ...     fit, motifs=motif_cols, p="EM_High", q="CM_High"
    ... )
    >>> result[result["qvalue"] < 0.1]  # significant motifs
    """
    rows = []
    motifs = list(motifs)

    for m in motifs:
        try:
            L, name = motif_diff_between_phenotypes(fit, m, p, q)
            est_log, pval = wald_contrast(fit, L, name)
            rows.append(
                {
                    "motif": m,
                    "phenotype_p": p,
                    "phenotype_q": q,
                    "log_effect": float(est_log),  # natural log
                    "pvalue": float(pval),
                }
            )
        except KeyError:
            if keep_missing:
                rows.append(
                    {
                        "motif": m,
                        "phenotype_p": p,
                        "phenotype_q": q,
                        "log_effect": np.nan,
                        "pvalue": np.nan,
                    }
                )
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Convert ln effect to log-base fold change
    ln_base = math.log(float(log_base))
    df["logFC"] = df["log_effect"] / ln_base

    # Adjust p-values
    if adjust.upper() in {"BH", "FDR", "BENJAMINI-HOCHBERG"}:
        df["qvalue"] = bh_fdr(df["pvalue"].values)
    else:
        raise ValueError(f"Unknown adjust='{adjust}'. Use 'BH'.")

    # Volcano y-axis
    df["neglog10_q"] = -np.log10(df["qvalue"].clip(lower=1e-300))

    # Sort: most significant first
    df = df.sort_values(["qvalue", "pvalue"], ascending=True).reset_index(drop=True)
    return df