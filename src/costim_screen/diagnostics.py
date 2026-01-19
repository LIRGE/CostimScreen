from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_alpha_nb2_moments(y: np.ndarray, mu: np.ndarray) -> float:
    """
    Method-of-moments alpha for NB2: Var = mu + alpha*mu^2.
    """
    mu = np.clip(mu, 1e-9, None)
    num = np.sum((y - mu) ** 2 - mu)
    den = np.sum(mu ** 2)
    alpha = num / max(den, 1e-12)
    return float(max(alpha, 0.0))


def per_sample_dispersion(counts_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Computes mean, variance, var/mean for each sample across domains.
    """
    means = counts_wide.mean(axis=0)
    vars_ = counts_wide.var(axis=0, ddof=1)
    out = pd.DataFrame({"mean": means, "var": vars_, "var_over_mean": vars_ / means.replace(0, np.nan)})
    return out


def poisson_overdispersion_test(y: np.ndarray, mu: np.ndarray, df_resid: int) -> float:
    """
    Pearson chi-square / df_resid. ~1 indicates Poisson-like.
    """
    pearson = np.sum((y - mu) ** 2 / np.clip(mu, 1e-9, None))
    return pearson / max(df_resid, 1)


def zero_fraction(counts_wide: pd.DataFrame) -> pd.Series:
    return (counts_wide == 0).mean(axis=0)
