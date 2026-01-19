from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

from .diagnostics import estimate_alpha_nb2_moments


@dataclass
class FitResult:
    res: sm.GLM
    alpha: float
    formula: str
    data_cols: list[str]


def build_joint_formula(motif_cols: list[str]) -> str:
    """
    Joint model, no intercept:
      - phenotype-specific intercepts: 0 + C(phenotype)
      - block fixed effects: C(block)
      - motif effects varying by phenotype: motif:C(phenotype)
    """
    # interactions for each motif
    inter = " + ".join([f"{m}:C(phenotype)" for m in motif_cols])
    formula = f"count ~ 0 + C(phenotype) + C(block)"
    if inter:
        formula += " + " + inter
    return formula


def fit_nb_glm_iter_alpha(
    df: pd.DataFrame,
    formula: str,
    offset_col: str = "offset",
    max_iter: int = 8,
    alpha_init: float = 0.1,
    cluster_col: Optional[str] = "block",
    regularization: float = 0.0,
) -> FitResult:
    """
    Iteratively estimate NB2 alpha by moments:
      1) fit GLM NB with current alpha
      2) update alpha from y, mu
      3) repeat

    Parameters
    ----------
    df : pd.DataFrame
        Data frame with count data and covariates.
    formula : str
        Patsy formula string.
    offset_col : str
        Column name for the offset (log library size).
    max_iter : int
        Maximum iterations for alpha estimation.
    alpha_init : float
        Initial dispersion parameter.
    cluster_col : str | None
        Column for cluster-robust standard errors.
    regularization : float
        L2 regularization strength (0 = no regularization).

    Returns robustcov results if cluster_col is provided.
    """
    y, X = patsy.dmatrices(formula, data=df, return_type="dataframe")
    y = np.asarray(y).ravel()
    offset = np.asarray(df[offset_col], dtype=float)

    # Drop columns with zero variance (can cause numerical issues)
    col_std = X.std()
    zero_var_cols = col_std[col_std == 0].index.tolist()
    if zero_var_cols:
        import warnings
        warnings.warn(f"Dropping {len(zero_var_cols)} zero-variance columns from design matrix")
        X = X.drop(columns=zero_var_cols)

    alpha = float(alpha_init)
    glm_res = None
    used_regularization = regularization > 0

    for iteration in range(max_iter):
        fam = sm.families.NegativeBinomial(alpha=alpha)
        model = sm.GLM(y, X, family=fam, offset=offset)
        try:
            if used_regularization:
                glm_res = model.fit_regularized(alpha=regularization, L1_wt=0)  # L2 only
            else:
                glm_res = model.fit(maxiter=100)
        except (ValueError, np.linalg.LinAlgError) as e:
            # Try with regularization if standard fit fails
            if not used_regularization:
                import warnings
                warnings.warn(f"Standard fit failed on iteration {iteration}, trying with L2 regularization")
                used_regularization = True
                glm_res = model.fit_regularized(alpha=0.01, L1_wt=0)
            else:
                raise e

        mu = glm_res.fittedvalues
        alpha_new = estimate_alpha_nb2_moments(y, np.asarray(mu))
        # stabilize updates
        alpha_new = 0.5 * alpha + 0.5 * alpha_new
        # Bound alpha to reasonable range
        alpha_new = np.clip(alpha_new, 1e-4, 100.0)
        if abs(alpha_new - alpha) / (alpha + 1e-9) < 0.05:
            alpha = alpha_new
            break
        alpha = alpha_new

    # cluster-robust SE (only works with non-regularized fit)
    if cluster_col is not None and cluster_col in df.columns and not used_regularization:
        groups = df[cluster_col].astype(str).values
        # Re-fit with robust covariance
        try:
            glm_res = model.fit(maxiter=100, cov_type="cluster", cov_kwds={"groups": groups})
        except Exception:
            import warnings
            warnings.warn("Cluster-robust SEs failed; using standard SEs.")
    elif used_regularization:
        import warnings
        warnings.warn("Regularization was used; cluster-robust SEs not available.")

    return FitResult(res=glm_res, alpha=alpha, formula=formula, data_cols=list(X.columns))
