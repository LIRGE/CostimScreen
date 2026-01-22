"""
Negative binomial GLM fitting with iterative dispersion estimation.

This module provides the core statistical modeling functionality, including
building the joint GLM formula and fitting negative binomial models with
iteratively estimated dispersion parameters.

Functions
---------
build_joint_formula
    Construct a patsy formula for the joint motif-phenotype model.
fit_nb_glm_iter_alpha
    Fit a negative binomial GLM with iterative alpha estimation.

Classes
-------
FitResult
    Container for fitted model results.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

from .diagnostics import estimate_alpha_nb2_moments
from .utils import normalize_phenotype


@dataclass
class FitResult:
    """Container for fitted negative binomial GLM results."""

    #: The fitted GLM results object from statsmodels.
    res: sm.GLM
    #: The estimated dispersion parameter (NB2 parameterization).
    alpha: float
    #: The patsy formula string used for fitting.
    formula: str
    #: Column names from the design matrix, used for contrast construction.
    data_cols: list[str]


def build_joint_formula(motif_cols: list[str]) -> str:
    """Construct a patsy formula for the joint motif-phenotype model.

    Creates a formula string for a negative binomial GLM with:
    - Phenotype-specific intercepts (no global intercept)
    - CCR fixed effects for batch correction
    - Motif effects that vary by phenotype (interactions)

    The model structure is::

        count ~ 0 + C(phenotype) + C(CCR) + motif1:C(phenotype) + motif2:C(phenotype) + ...

    Parameters
    ----------
    motif_cols : list of str
        Names of the motif/ELM feature columns to include as predictors.

    Returns
    -------
    str
        Patsy formula string.

    Examples
    --------
    >>> formula = build_joint_formula(["ELM_SH3", "ELM_SUMO"])
    >>> print(formula)
    count ~ 0 + C(phenotype) + C(CCR) + ELM_SH3:C(phenotype) + ELM_SUMO:C(phenotype)
    """
    # interactions for each motif
    inter = " + ".join([f"{m}:C(phenotype)" for m in motif_cols])
    formula = "count ~ 0 + C(phenotype) + C(CCR)"
    if inter:
        formula += " + " + inter
    return formula


def fit_nb_glm_iter_alpha(
        df: pd.DataFrame,
        formula: str,
        offset_col: str = "offset",
        max_iter: int = 8,
        alpha_init: float = 0.1,
        cluster_col: Optional[str] = "CCR",
        regularization: float = 0.0,
) -> FitResult:
    """Fit a negative binomial GLM with iterative dispersion estimation.

    Fits a negative binomial GLM (NB2 parameterization) using an iterative
    procedure to estimate the dispersion parameter alpha:

    1. Fit GLM with current alpha
    2. Update alpha using method-of-moments from residuals
    3. Repeat until convergence

    The NB2 variance function is: Var(Y) = mu + alpha * mu^2

    Parameters
    ----------
    df : pd.DataFrame
        Long-format data with count, phenotype, CCR, offset, and motif columns.
    formula : str
        Patsy formula string (see :func:`build_joint_formula`).
    offset_col : str, default "offset"
        Name of the column containing log library sizes.
    max_iter : int, default 8
        Maximum iterations for alpha estimation.
    alpha_init : float, default 0.1
        Initial dispersion parameter value.
    cluster_col : str or None, default "CCR"
        Column name for cluster-robust standard errors. If None, uses
        standard (non-robust) SEs.
    regularization : float, default 0.0
        L2 regularization strength. If > 0, uses regularized fitting
        (cluster-robust SEs not available with regularization).

    Returns
    -------
    FitResult
        Fitted model results including the GLM object, estimated alpha,
        formula, and design matrix column names.

    Notes
    -----
    If the standard fit fails due to numerical issues, the function
    automatically falls back to L2 regularization with alpha=0.01.

    Examples
    --------
    >>> formula = build_joint_formula(motif_cols)
    >>> fit = fit_nb_glm_iter_alpha(df, formula, offset_col="offset")
    >>> print(f"Dispersion: {fit.alpha:.3f}")
    Dispersion: 1.152
    """
    # Normalize phenotype column to ASCII before fitting
    df = df.copy()
    if "phenotype" in df.columns:
        df["phenotype"] = df["phenotype"].apply(normalize_phenotype)

    y, X = patsy.dmatrices(formula, data=df, return_type="dataframe")
    y = np.asarray(y).ravel()
    offset = np.asarray(df[offset_col], dtype=float)

    # Drop columns with zero variance (can cause numerical issues)
    col_std = X.std()
    zero_var_cols = col_std[col_std == 0].index.tolist()
    if zero_var_cols:
        import warnings

        warnings.warn(
            f"Dropping {len(zero_var_cols)} zero-variance columns from design matrix"
        )
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
            # If standard fit fails, use L2 regularization only to get start_params,
            # then refit unregularized so we can compute cluster-robust SEs.
            import warnings
            warnings.warn(
                f"Standard fit failed on iteration {iteration}; "
                "using L2 regularization to initialize, then refitting unregularized."
            )
            reg_res = model.fit_regularized(alpha=max(regularization, 0.01), L1_wt=0)

            try:
                glm_res = model.fit(start_params=np.asarray(reg_res.params), maxiter=200)
                used_regularization = False
            except Exception as e2:
                warnings.warn(
                    "Refit without regularization failed; keeping regularized fit "
                    "(cluster-robust SEs will not be available)."
                )
                glm_res = reg_res
                used_regularization = True

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

    # Final refit with the final alpha so that fit.res matches fit.alpha
    fam_final = sm.families.NegativeBinomial(alpha=float(alpha))
    model_final = sm.GLM(y, X, family=fam_final, offset=offset)

    if cluster_col is not None and cluster_col in df.columns and not used_regularization:
        groups = df[cluster_col].astype(str).values
        try:
            glm_res = model_final.fit(
                maxiter=200,
                cov_type="cluster",
                cov_kwds={"groups": groups},
            )
        except Exception:
            import warnings
            warnings.warn("Cluster-robust SEs failed; using standard SEs.")
            glm_res = model_final.fit(maxiter=200)
    else:
        glm_res = model_final.fit(maxiter=200)
        if used_regularization:
            import warnings
            warnings.warn(
                "Regularization was used during fitting; cluster-robust SEs are not available "
                "for regularized fits."
            )

    return FitResult(res=glm_res, alpha=float(alpha), formula=formula, data_cols=list(X.columns))
