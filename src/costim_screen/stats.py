# src/costim_screen/stats.py
from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .contrasts import motif_diff_between_phenotypes, wald_contrast


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR adjustment.

    Parameters
    ----------
    pvals : array-like
        Raw p-values.

    Returns
    -------
    qvals : np.ndarray
        BH-adjusted q-values, same shape as pvals.
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
    """
    Compute per-motif contrasts between phenotypes p and q:
      - estimate on log scale (beta_p - beta_q)
      - log2FC (or log_base)
      - Wald p-value
      - BH-adjusted q-value

    Notes
    -----
    With your model:
      count ~ 0 + C(phenotype) + C(block) + motif:C(phenotype)
    the contrast (beta_motif:p - beta_motif:q) is the difference in motif-associated
    multiplicative effects between phenotypes on the log scale.

    Returns
    -------
    DataFrame with columns:
      motif, phenotype_p, phenotype_q, log_effect, logFC, pvalue, qvalue, neglog10_q
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
