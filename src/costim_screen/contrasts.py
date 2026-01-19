from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple


def coef_name_for_motif_phenotype(motif: str, phenotype: str) -> str:
    # patsy names interaction as "motif:C(phenotype)[<phenotype>]" when no T-coding intercept is used
    return f"{motif}:C(phenotype)[{phenotype}]"


def motif_diff_between_phenotypes(fit, motif: str, p: str, q: str) -> Tuple[np.ndarray, str]:
    """
    Build contrast vector for (beta_motif:p - beta_motif:q)
    """
    cols = fit.data_cols
    L = np.zeros((1, len(cols)))

    cn_p = coef_name_for_motif_phenotype(motif, p)
    cn_q = coef_name_for_motif_phenotype(motif, q)

    if cn_p not in cols or cn_q not in cols:
        missing = [c for c in (cn_p, cn_q) if c not in cols]
        raise KeyError(f"Missing coefficients: {missing}")

    L[0, cols.index(cn_p)] = 1.0
    L[0, cols.index(cn_q)] = -1.0
    name = f"{motif}: {p} - {q}"
    return L, name


def wald_contrast(
    fit,
    L: np.ndarray,
    name: str,
) -> Tuple[float, float]:
    """
    Returns (estimate, pvalue) for linear contrast L' beta
    """
    test = fit.res.t_test(L)
    # Handle both scalar and array results
    effect = test.effect
    pvalue = test.pvalue
    est = float(effect.item()) if hasattr(effect, 'item') else float(effect)
    p = float(pvalue.item()) if hasattr(pvalue, 'item') else float(pvalue)
    return est, p
