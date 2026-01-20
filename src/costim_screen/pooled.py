# src/costim_screen/pooled.py
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence, Tuple, Optional

import numpy as np
import pandas as pd

from .contrasts import coef_name_for_motif_phenotype, wald_contrast
from .stats import bh_fdr
from .plots import volcano_plot


def _phenotype_label(tsubset: str, pd1: str) -> str:
    return f"{tsubset}_{pd1}"


def _build_L_from_weights(fit, weights: dict[str, float]) -> np.ndarray:
    """
    weights: mapping from coefficient name -> weight
    Returns 1xP contrast vector aligned to fit.data_cols.
    """
    cols = fit.data_cols
    L = np.zeros((1, len(cols)), dtype=float)
    missing = [k for k in weights.keys() if k not in cols]
    if missing:
        raise KeyError(f"Missing coefficients in model: {missing[:10]}" + (" ..." if len(missing) > 10 else ""))

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
    """
    Contrast: (avg over PD1 of motif effect in tsubset_p) - (avg over PD1 in tsubset_q)

    Example: EM vs CM pooled over PD1
      0.5*(beta_EM_H + beta_EM_L) - 0.5*(beta_CM_H + beta_CM_L)
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
    tsubsets: Sequence[str] = ("Naive", "CM", "EM"),
    pd1_high: str = "High",
    pd1_low: str = "Low",
) -> Tuple[np.ndarray, str]:
    """
    Contrast: (avg over Tsubset of motif effect in PD1_high) - (avg over Tsubset in PD1_low)

    Example:
      (1/3)*(beta_Naive_H + beta_CM_H + beta_EM_H) - (1/3)*(beta_Naive_L + beta_CM_L + beta_EM_L)
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
    """
    Per-motif: pooled-PD1 Tsubset contrast -> logFC + p + q
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
                "log_effect": float(est_log),   # natural log
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
    tsubsets: Sequence[str] = ("Naive", "CM", "EM"),
    pd1_high: str = "High",
    pd1_low: str = "Low",
    adjust: str = "BH",
    log_base: float = 2.0,
) -> pd.DataFrame:
    """
    Per-motif: pooled-Tsubset PD1 contrast -> logFC + p + q
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
    """
    Convenience: compute pooled-PD1 Tsubset contrast table and plot volcano.
    Returns the table.
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
    tsubsets: Sequence[str] = ("Naive", "CM", "EM"),
    pd1_high: str = "High",
    pd1_low: str = "Low",
    q_thresh: float = 0.10,
    lfc_thresh: float = 1.0,
    title: Optional[str] = None,
    top_n_labels: int = 12,
    outpath: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Convenience: compute pooled-Tsubset PD1 contrast table and plot volcano.
    Returns the table.
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
