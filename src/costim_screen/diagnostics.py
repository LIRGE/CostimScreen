"""
Model diagnostics and dispersion estimation.

This module provides functions for assessing model fit, estimating dispersion
parameters, and detecting overdispersion in count data.

Functions
---------
estimate_alpha_nb2_moments
    Estimate NB2 dispersion parameter using method of moments.
per_sample_dispersion
    Compute dispersion statistics for each sample.
poisson_overdispersion_test
    Test for overdispersion relative to Poisson model.
zero_fraction
    Compute the fraction of zero counts per sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_alpha_nb2_moments(y: np.ndarray, mu: np.ndarray) -> float:
    """Estimate NB2 dispersion parameter using method of moments.

    Estimates the dispersion parameter alpha for the NB2 (quadratic)
    parameterization where Var(Y) = mu + alpha * mu^2.

    The estimator solves: sum((y - mu)^2 - mu) = alpha * sum(mu^2)

    Parameters
    ----------
    y : np.ndarray
        Observed counts.
    mu : np.ndarray
        Fitted mean values from the model.

    Returns
    -------
    float
        Estimated alpha, clipped to be non-negative.

    Notes
    -----
    The NB2 parameterization is commonly used in ecology and genomics.
    Alpha = 0 corresponds to Poisson (no overdispersion).
    Larger alpha indicates more overdispersion.

    Examples
    --------
    >>> y = np.array([10, 20, 5, 15])
    >>> mu = np.array([12, 18, 7, 14])
    >>> alpha = estimate_alpha_nb2_moments(y, mu)
    """
    mu = np.clip(mu, 1e-9, None)
    num = np.sum((y - mu) ** 2 - mu)
    den = np.sum(mu**2)
    alpha = num / max(den, 1e-12)
    return float(max(alpha, 0.0))


def per_sample_dispersion(counts_wide: pd.DataFrame) -> pd.DataFrame:
    """Compute dispersion statistics for each sample.

    Calculates mean, variance, and variance-to-mean ratio for each
    sample across all candidates. The variance-to-mean ratio (VMR)
    indicates the degree of overdispersion: VMR = 1 for Poisson,
    VMR > 1 for overdispersed data.

    Parameters
    ----------
    counts_wide : pd.DataFrame
        Wide-format count matrix with candidates as rows and samples
        as columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - mean: mean count across candidates
        - var: variance of counts across candidates
        - var_over_mean: variance-to-mean ratio (dispersion index)

    Examples
    --------
    >>> counts = pd.DataFrame({"S1": [10, 20, 30], "S2": [5, 15, 25]})
    >>> disp = per_sample_dispersion(counts)
    >>> disp["var_over_mean"]
    """
    means = counts_wide.mean(axis=0)
    vars_ = counts_wide.var(axis=0, ddof=1)
    out = pd.DataFrame(
        {"mean": means, "var": vars_, "var_over_mean": vars_ / means.replace(0, np.nan)}
    )
    return out


def poisson_overdispersion_test(y: np.ndarray, mu: np.ndarray, df_resid: int) -> float:
    """Test for overdispersion relative to Poisson model.

    Computes the Pearson chi-square statistic divided by residual degrees
    of freedom. Under Poisson assumptions, this ratio should be approximately 1.
    Values significantly greater than 1 indicate overdispersion.

    Parameters
    ----------
    y : np.ndarray
        Observed counts.
    mu : np.ndarray
        Fitted mean values.
    df_resid : int
        Residual degrees of freedom (n_obs - n_params).

    Returns
    -------
    float
        Dispersion ratio (Pearson chi-square / df_resid).
        Values > 1 suggest overdispersion.

    Examples
    --------
    >>> y = np.array([10, 20, 5, 15, 8])
    >>> mu = np.array([12, 18, 7, 14, 9])
    >>> ratio = poisson_overdispersion_test(y, mu, df_resid=3)
    >>> if ratio > 1.5:
    ...     print("Significant overdispersion detected")
    """
    pearson = np.sum((y - mu) ** 2 / np.clip(mu, 1e-9, None))
    return pearson / max(df_resid, 1)


def zero_fraction(counts_wide: pd.DataFrame) -> pd.Series:
    """Compute the fraction of zero counts per sample.

    Calculates the proportion of candidates with zero counts in each
    sample. High zero fractions may indicate zero-inflation or
    technical dropouts.

    Parameters
    ----------
    counts_wide : pd.DataFrame
        Wide-format count matrix with candidates as rows and samples
        as columns.

    Returns
    -------
    pd.Series
        Fraction of zeros for each sample (values between 0 and 1).

    Examples
    --------
    >>> counts = pd.DataFrame({"S1": [0, 10, 0, 5], "S2": [1, 0, 3, 0]})
    >>> zero_fraction(counts)
    S1    0.5
    S2    0.5
    dtype: float64
    """
    return (counts_wide == 0).mean(axis=0)