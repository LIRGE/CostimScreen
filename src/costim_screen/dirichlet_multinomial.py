"""
Dirichlet-Multinomial model for costimulatory screen analysis.

This module implements a principled model for the FACS partitioning structure:
- Cells from each CCD in each CCR are partitioned into 6 phenotypes
- The counts across phenotypes sum to the total for that CCD-CCR combination
- This compositional structure is naturally modeled by the Dirichlet-Multinomial

Model specification:
    (n_ic,1, ..., n_ic,6) | π_ic ~ Multinomial(N_ic, π_ic)
    π_ic ~ Dirichlet(α · μ_ic)

    log(μ_ic,p / μ_ic,ref) = β_0,p + β_CCR[c],p + Σ_m β_m,p · ELM_im

where:
    - i indexes CCDs, c indexes CCRs, p indexes phenotypes
    - N_ic = Σ_p n_ic,p is the total count for CCD i in CCR c
    - α is the concentration parameter (controls overdispersion)
    - β_m,p is the effect of ELM m on phenotype p (relative to reference)

Classes
-------
DirichletMultinomialModel
    Main model class for fitting and inference.

DirichletMultinomialResult
    Container for fitted model results.

Functions
---------
fit_dm_model
    Fit the Dirichlet-Multinomial model to data.
simulate_dm_data
    Generate synthetic data from the model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, minimize_scalar
from scipy.special import softmax, digamma, gammaln
from scipy.stats import norm
import warnings

from .features import split_elm_list
from .utils import normalize_phenotype


# Phenotype definitions (support both Naïve and Naive spellings)
TSUBSETS = ["Naive", "CM", "EM"]
PD1_LEVELS = ["High", "Low"]
PHENOTYPES = [f"{ts}_{pd1}" for ts in TSUBSETS for pd1 in PD1_LEVELS]
N_PHENOTYPES = len(PHENOTYPES)
REFERENCE_PHENOTYPE = "Naive_Low"  # Reference category for identifiability

# Also define alternate spellings for compatibility
TSUBSETS_ALT = ["Naive", "CM", "EM"]
PHENOTYPES_ALT = [f"{ts}_{pd1}" for ts in TSUBSETS_ALT for pd1 in PD1_LEVELS]
REFERENCE_PHENOTYPE_ALT = "Naive_Low"


@dataclass
class DirichletMultinomialData:
    """Container for data in the format needed for DM model.

    Attributes
    ----------
    counts : np.ndarray
        Count matrix of shape (n_observations, 6) where each row is a CCD-CCR
        combination and columns are phenotype counts.
    elm_matrix : np.ndarray
        Binary ELM indicator matrix of shape (n_observations, n_elms).
    ccr_ids : np.ndarray
        CCR identifier for each observation (for random effects).
    ccd_ids : np.ndarray
        CCD identifier for each observation.
    totals : np.ndarray
        Row sums of counts (total per CCD-CCR).
    elm_names : List[str]
        Names of ELMs.
    phenotype_names : List[str]
        Names of phenotypes (column order of counts).
    """
    counts: np.ndarray
    elm_matrix: np.ndarray
    ccr_ids: np.ndarray
    ccd_ids: np.ndarray
    totals: np.ndarray
    elm_names: List[str]
    phenotype_names: List[str] = field(default_factory=lambda: PHENOTYPES.copy())

    @property
    def n_ccrs(self) -> int:
        return len(np.unique(self.ccr_ids))

    @property
    def n_elms(self) -> int:
        return self.elm_matrix.shape[1]

    @property
    def n_obs(self) -> int:
        return self.counts.shape[0]

    @property
    def n_phenotypes(self) -> int:
        return self.counts.shape[1]


@dataclass
class DirichletMultinomialResult:
    """Container for fitted DM model results.

    Attributes
    ----------
    beta_0 : np.ndarray
        Baseline phenotype effects, shape (n_phenotypes-1,).
    beta_elm : np.ndarray
        ELM effects, shape (n_elms, n_phenotypes-1).
    beta_ccr : np.ndarray
        CCR random effects, shape (n_ccrs, n_phenotypes-1).
    alpha : float
        Concentration parameter.
    se_beta_elm : np.ndarray
        Standard errors for ELM effects.
    cov_beta_elm_full : np.ndarray, optional
        Full covariance matrix for ELM effects, shape (n_elms*(n_phenotypes-1), n_elms*(n_phenotypes-1)).
        Used for computing proper standard errors for contrasts.
    log_likelihood : float
        Log-likelihood at MLE.
    converged : bool
        Whether optimization converged.
    data : DirichletMultinomialData
        Reference to the data used for fitting.
    """
    beta_0: np.ndarray
    beta_elm: np.ndarray
    beta_ccr: np.ndarray
    alpha: float
    se_beta_elm: Optional[np.ndarray]
    cov_beta_elm_full: Optional[np.ndarray]
    log_likelihood: float
    converged: bool
    data: DirichletMultinomialData

    def contrast_pd1(self, elm_name: str) -> Tuple[float, float, float]:
        """Compute contrast between PD1 High vs Low, pooled over T-subsets.

        Returns (effect, se, pvalue).
        """
        # Build contrast weights
        weights = {}
        for pheno in self.data.phenotype_names:
            ts, pd1 = pheno.rsplit("_", 1)
            if pd1 == "High":
                weights[pheno] = 1/3
            elif pd1 == "Low":
                weights[pheno] = -1/3

        return self.wald_contrast(elm_name, weights)

    def contrast_tsubset(self, elm_name: str, tsubset1: str, tsubset2: str) -> Tuple[float, float, float]:
        """Compute contrast between T-subsets, pooled over PD1.

        Returns (effect, se, pvalue) for the contrast:
        (1/2)[β_{tsubset1_High} + β_{tsubset1_Low}] - (1/2)[β_{tsubset2_High} + β_{tsubset2_Low}]
        """
        # Build contrast weights
        weights = {}
        for pheno in self.data.phenotype_names:
            ts, pd1 = pheno.rsplit("_", 1)
            if ts == tsubset1:
                weights[pheno] = 0.5
            elif ts == tsubset2:
                weights[pheno] = -0.5

        return self.wald_contrast(elm_name, weights)

    def get_elm_effect(self, elm_name: str, phenotype: str) -> Tuple[float, float]:
        """Get effect estimate and SE for an ELM-phenotype combination.

        Parameters
        ----------
        elm_name : str
            Name of the ELM.
        phenotype : str
            Name of the phenotype.

        Returns
        -------
        effect : float
            Log-odds effect (relative to reference phenotype).
        se : float
            Standard error of the effect.
        """
        elm_idx = self.data.elm_names.index(elm_name)

        if phenotype == self.reference_phenotype:
            return 0.0, 0.0

        pheno_idx = self.non_ref_phenotypes.index(phenotype)

        effect = self.beta_elm[elm_idx, pheno_idx]
        se = self.se_beta_elm[elm_idx, pheno_idx] if self.se_beta_elm is not None else np.nan

        return effect, se

    def get_elm_pvalues(self, elm_name: str) -> pd.Series:
        """Get p-values for all phenotypes for an ELM.

        Tests H0: β_m,p = 0 for each phenotype p.
        """
        pvals = {}
        for pheno in self.data.phenotype_names:
            if pheno == self.reference_phenotype:
                pvals[pheno] = np.nan
            else:
                effect, se = self.get_elm_effect(elm_name, pheno)
                if se > 0 and np.isfinite(se):
                    z = effect / se
                    pvals[pheno] = 2 * stats.norm.sf(np.abs(z))
                else:
                    pvals[pheno] = np.nan
        return pd.Series(pvals)

    def get_probability_shift(self, elm_name: str) -> pd.Series:
        """Get the shift in phenotype probabilities due to an ELM.

        Computes P(phenotype | ELM=1) - P(phenotype | ELM=0) at baseline.
        """
        elm_idx = self.data.elm_names.index(elm_name)

        # Baseline log-odds (without ELM)
        log_odds_base = np.zeros(self.data.n_phenotypes)
        ref_idx = self.data.phenotype_names.index(self.reference_phenotype)
        non_ref = [i for i in range(self.data.n_phenotypes) if i != ref_idx]
        log_odds_base[non_ref] = self.beta_0

        # Log-odds with ELM
        log_odds_elm = log_odds_base.copy()
        log_odds_elm[non_ref] += self.beta_elm[elm_idx, :]

        # Convert to probabilities
        prob_base = softmax(log_odds_base)
        prob_elm = softmax(log_odds_elm)

        shift = prob_elm - prob_base
        return pd.Series(shift, index=self.data.phenotype_names)

    @property
    def non_ref_phenotypes(self) -> List[str]:
        """Get the non-reference phenotypes."""
        return [p for p in self.data.phenotype_names if p != self.reference_phenotype]

    def omnibus_elm_test(self, elm_name: str) -> Tuple[float, int, float]:
        """Omnibus Wald test for whether an ELM has any effect across phenotypes.

        Tests H0: beta_elm[elm, :] == 0 (vector of length K-1).

        Returns
        -------
        chi2 : float
        df : int
        pvalue : float
        """
        elm_idx = self.data.elm_names.index(elm_name)
        b = np.asarray(self.beta_elm[elm_idx, :], dtype=float)

        cov = self._cov_block_for_elm(elm_idx)
        if cov is None:
            return np.nan, b.size, np.nan

        # Use a stable solve; fall back to pseudo-inverse if needed
        try:
            chi2 = float(b.T @ np.linalg.solve(cov, b))
        except np.linalg.LinAlgError:
            pinv = np.linalg.pinv(cov)
            chi2 = float(b.T @ pinv @ b)

        df = int(b.size)
        p = float(stats.chi2.sf(chi2, df=df))
        return chi2, df, p

    @property
    def reference_phenotype(self) -> str:
        """Get the reference phenotype for this result."""
        return _get_reference_phenotype(self.data.phenotype_names)

    def _cov_block_for_elm(self, elm_idx: int) -> Optional[np.ndarray]:
        """Extract covariance matrix for a single ELM's coefficients.

        Parameters
        ----------
        elm_idx : int
            Index of the ELM.

        Returns
        -------
        np.ndarray or None
            Covariance matrix of shape (K-1, K-1) where K is the number of phenotypes.
            Returns None if full covariance matrix is not available.
        """
        if self.cov_beta_elm_full is None:
            return None

        k = len(self.non_ref_phenotypes)  # = K-1
        s = elm_idx * k
        return self.cov_beta_elm_full[s:s+k, s:s+k]

    def wald_contrast(self, elm_name: str, weights: dict) -> Tuple[float, float, float]:
        """Compute a Wald-type contrast with proper covariance accounting.

        This is the correct way to compute contrasts - it uses the full covariance
        matrix to account for correlations between phenotype coefficients.

        Parameters
        ----------
        elm_name : str
            Name of the ELM.
        weights : dict
            Dictionary mapping phenotype names to contrast weights.
            Example: {"EM_High": 1.0, "CM_High": -1.0} for EM_High vs CM_High.
            The reference phenotype can be included (weight is effectively 0).

        Returns
        -------
        effect : float
            Contrast effect (log-odds scale).
        se : float
            Standard error of the contrast.
        pvalue : float
            Two-sided p-value from Wald test.

        Examples
        --------
        >>> # EM_High vs CM_High
        >>> eff, se, p = result.wald_contrast("LIG_SH3", {"EM_High": 1, "CM_High": -1})
        >>>
        >>> # Pooled PD1 effect: (1/3) * sum over T-subsets of (Low - High)
        >>> eff, se, p = result.wald_contrast("LIG_SH3", {
        ...     "Naive_Low": 1/3, "Naive_High": -1/3,
        ...     "CM_Low": 1/3, "CM_High": -1/3,
        ...     "EM_Low": 1/3, "EM_High": -1/3,
        ... })
        """
        elm_idx = self.data.elm_names.index(elm_name)
        non_ref = self.non_ref_phenotypes

        # Build contrast vector over non-reference phenotypes
        w = np.zeros(len(non_ref))
        for phenotype, coef in weights.items():
            if phenotype == self.reference_phenotype:
                continue  # Reference is always 0
            w[non_ref.index(phenotype)] += coef

        # Compute effect
        effect = float(w @ self.beta_elm[elm_idx, :])

        # Compute SE using covariance matrix
        cov = self._cov_block_for_elm(elm_idx)
        if cov is None:
            return effect, np.nan, np.nan

        var = float(w @ cov @ w)
        se = np.sqrt(max(var, 0.0))

        # Wald test
        z = effect / se if se > 0 else 0.0
        pval = 2 * stats.norm.sf(abs(z))

        return effect, se, pval


def _compute_alpha_mu(
    beta_0: np.ndarray,
    beta_elm: np.ndarray,
    beta_ccr: np.ndarray,
    alpha: float,
    elm_matrix: np.ndarray,
    ccr_ids: np.ndarray,
    ref_idx: int,
) -> np.ndarray:
    """Compute Dirichlet parameters α·μ for all observations.

    Parameters
    ----------
    beta_0 : np.ndarray
        Baseline effects, shape (n_phenotypes-1,).
    beta_elm : np.ndarray
        ELM effects, shape (n_elms, n_phenotypes-1).
    beta_ccr : np.ndarray
        CCR effects, shape (n_ccrs, n_phenotypes-1).
    alpha : float
        Concentration parameter.
    elm_matrix : np.ndarray
        ELM indicators, shape (n_obs, n_elms).
    ccr_ids : np.ndarray
        CCR indices, shape (n_obs,).
    ref_idx : int
        Index of reference phenotype.

    Returns
    -------
    np.ndarray
        Dirichlet parameters, shape (n_obs, n_phenotypes).
    """
    n_obs = elm_matrix.shape[0]
    n_phenotypes = len(beta_0) + 1

    # Compute log-odds for non-reference phenotypes
    # η = β_0 + X @ β_elm + β_ccr[ccr]
    log_odds = (beta_0[np.newaxis, :] +
                elm_matrix @ beta_elm +
                beta_ccr[ccr_ids, :])  # (n_obs, n_phenotypes-1)

    # Insert 0 for reference phenotype
    full_log_odds = np.zeros((n_obs, n_phenotypes))
    non_ref = [i for i in range(n_phenotypes) if i != ref_idx]
    full_log_odds[:, non_ref] = log_odds

    # Convert to probabilities via softmax
    mu = softmax(full_log_odds, axis=1)

    # Smooth mu to avoid gammaln(0) without creating discontinuities
    # This preserves differentiability unlike clipping
    eps = 1e-12
    mu = (mu + eps) / (1.0 + eps * n_phenotypes)

    # Dirichlet parameters
    alpha_mu = alpha * mu

    return alpha_mu


def compute_observed_fisher_information(
    result: DirichletMultinomialResult,
    obs_weights: Optional[np.ndarray] = None,
    include_ccr_effects: bool = True,
    eps: float = 1e-5,
    verbose: bool = False,
    fd_scheme: str = "central",   # "central" (more accurate) or "forward" (faster)
    jitter: float = 1e-10,
) -> np.ndarray:
    """
    Compute a covariance matrix via finite differences of the analytic gradient.

    Returns
    -------
    cov : np.ndarray
        Approximate covariance matrix for the fitted parameters:
        cov ≈ [∇² nll(θ̂)]^{-1}
        where nll is the *weighted* negative log-likelihood used in fitting.
    """
    data = result.data
    n_obs = data.n_obs
    n_elms = data.n_elms
    n_phenotypes = data.n_phenotypes
    n_ccrs = data.n_ccrs

    ref_idx = data.phenotype_names.index(result.reference_phenotype)
    non_ref_idx = [k for k in range(n_phenotypes) if k != ref_idx]
    n_non_ref = n_phenotypes - 1

    # --- weights (must match fitting objective scaling) ---
    if obs_weights is None:
        obs_weights = np.ones(n_obs, dtype=float)
    else:
        obs_weights = np.asarray(obs_weights, dtype=float)
        if obs_weights.shape != (n_obs,):
            raise ValueError(f"obs_weights must have shape {(n_obs,)}, got {obs_weights.shape}.")
        if np.any(obs_weights < 0) or not np.isfinite(obs_weights).all():
            raise ValueError("obs_weights must be finite and non-negative.")

    # --- parameter sizes (must match fit_dm_model parameterization) ---
    n_beta_0 = n_non_ref
    n_beta_elm = n_elms * n_non_ref
    n_beta_ccr = ((n_ccrs - 1) * n_non_ref) if (include_ccr_effects and n_ccrs > 1) else 0
    n_params = 1 + n_beta_0 + n_beta_elm + n_beta_ccr  # +1 for log(alpha)

    # --- pack params from fitted result (in the same identifiable CCR parameterization) ---
    params = np.zeros(n_params, dtype=float)
    params[0] = np.log(result.alpha)
    params[1:1 + n_beta_0] = np.asarray(result.beta_0, dtype=float)
    params[1 + n_beta_0:1 + n_beta_0 + n_beta_elm] = np.asarray(result.beta_elm, dtype=float).ravel()

    if n_beta_ccr > 0:
        # store only first (n_ccrs-1) rows; last row is implied by sum-to-zero
        params[1 + n_beta_0 + n_beta_elm:] = np.asarray(result.beta_ccr[:-1, :], dtype=float).ravel()

    def _unpack(p: np.ndarray):
        """Unpack parameter vector into alpha, beta_0, beta_elm, beta_ccr(full)."""
        log_alpha = p[0]
        alpha = float(np.exp(log_alpha))

        beta_0 = p[1:1 + n_beta_0]
        beta_elm = p[1 + n_beta_0:1 + n_beta_0 + n_beta_elm].reshape(n_elms, n_non_ref)

        if n_beta_ccr > 0:
            raw_ccr = p[1 + n_beta_0 + n_beta_elm:].reshape(n_ccrs - 1, n_non_ref)
            last_row = -raw_ccr.sum(axis=0, keepdims=True)
            beta_ccr = np.vstack([raw_ccr, last_row])
        else:
            beta_ccr = np.zeros((n_ccrs, n_non_ref), dtype=float)

        return alpha, beta_0, beta_elm, beta_ccr

    def grad_nll(p: np.ndarray) -> np.ndarray:
        """
        Analytic gradient of the *weighted* negative log-likelihood w.r.t. packed params.
        Mirrors fit_dm_model's neg_ll_and_grad (with l2_penalty assumed 0 here).
        """
        alpha, beta_0, beta_elm, beta_ccr = _unpack(p)

        # logits for non-ref phenotypes
        eta = (
            beta_0[np.newaxis, :]
            + data.elm_matrix @ beta_elm
            + beta_ccr[data.ccr_ids, :]
        )  # (n_obs, K-1)

        full_z = np.zeros((n_obs, n_phenotypes), dtype=float)
        full_z[:, non_ref_idx] = eta

        mu0 = softmax(full_z, axis=1)  # (n_obs, K)

        eps_smooth = 1e-12
        denom = 1.0 + eps_smooth * n_phenotypes
        mu = (mu0 + eps_smooth) / denom  # smoothed probs

        a = alpha * mu
        N = data.counts.sum(axis=1)
        alpha_sum = a.sum(axis=1)

        # g = d ll / d a
        base = (digamma(alpha_sum) - digamma(alpha_sum + N))[:, None]
        g = base + digamma(data.counts + a) - digamma(a)  # (n_obs, K)

        # d ll / d z (softmax backprop), then restrict to eta
        g_dot_mu0 = np.sum(g * mu0, axis=1, keepdims=True)
        dLL_dz = (alpha / denom) * mu0 * (g - g_dot_mu0)  # (n_obs, K)
        dLL_deta = dLL_dz[:, non_ref_idx]  # (n_obs, K-1)

        # apply weights
        dLL_deta_w = obs_weights[:, None] * dLL_deta

        grad_ll_beta_0 = dLL_deta_w.sum(axis=0)  # (K-1,)
        grad_ll_beta_elm = data.elm_matrix.T @ dLL_deta_w  # (n_elms, K-1)

        # CCR raw gradients if included
        if n_beta_ccr > 0:
            grad_ll_beta_ccr_full = np.zeros((n_ccrs, n_non_ref), dtype=float)
            np.add.at(grad_ll_beta_ccr_full, data.ccr_ids, dLL_deta_w)
            grad_ll_beta_ccr_raw = grad_ll_beta_ccr_full[:-1, :] - grad_ll_beta_ccr_full[-1:, :]
        else:
            grad_ll_beta_ccr_raw = np.zeros((0, n_non_ref), dtype=float)

        # log(alpha) gradient
        g_dot_mu_alpha = np.sum(g * mu, axis=1)  # (n_obs,)
        grad_ll_log_alpha = alpha * float(np.sum(obs_weights * g_dot_mu_alpha))

        # Assemble gradient of nll = -ll
        grad = np.zeros(n_params, dtype=float)
        grad[0] = -grad_ll_log_alpha
        grad[1:1 + n_beta_0] = -grad_ll_beta_0
        grad[1 + n_beta_0:1 + n_beta_0 + n_beta_elm] = (-grad_ll_beta_elm).ravel()
        if n_beta_ccr > 0:
            grad[1 + n_beta_0 + n_beta_elm:] = (-grad_ll_beta_ccr_raw).ravel()

        return grad

    if verbose:
        print(f"Computing Hessian via finite differences ({n_params} parameters, scheme={fd_scheme})...")

    H = np.zeros((n_params, n_params), dtype=float)

    if fd_scheme.lower() == "forward":
        g0 = grad_nll(params)
        for i in range(n_params):
            step = eps * (1.0 + abs(params[i]))
            p_plus = params.copy()
            p_plus[i] += step
            g_plus = grad_nll(p_plus)
            H[:, i] = (g_plus - g0) / step
            if verbose and (i + 1) % 50 == 0:
                print(f"  {i+1}/{n_params} columns complete...")
    else:
        # central difference (more accurate)
        for i in range(n_params):
            step = eps * (1.0 + abs(params[i]))
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus[i] += step
            p_minus[i] -= step
            g_plus = grad_nll(p_plus)
            g_minus = grad_nll(p_minus)
            H[:, i] = (g_plus - g_minus) / (2.0 * step)
            if verbose and (i + 1) % 50 == 0:
                print(f"  {i+1}/{n_params} columns complete...")

    # Symmetrize for numerical stability
    H = 0.5 * (H + H.T)

    # Invert Hessian to get covariance
    I = np.eye(n_params, dtype=float)
    try:
        cov = np.linalg.inv(H)
        return cov
    except np.linalg.LinAlgError:
        # Retry with a tiny diagonal jitter before falling back to pinv
        try:
            cov = np.linalg.inv(H + float(jitter) * I)
            return cov
        except np.linalg.LinAlgError:
            if verbose:
                print("Warning: Hessian singular/ill-conditioned; using pseudo-inverse.")
            return np.linalg.pinv(H)


def _dm_log_likelihood(
    counts: np.ndarray,
    alpha_mu: np.ndarray,
) -> float:
    """Compute Dirichlet-Multinomial log-likelihood.

    Parameters
    ----------
    counts : np.ndarray
        Count matrix, shape (n_obs, n_phenotypes).
    alpha_mu : np.ndarray
        Dirichlet parameters α·μ, shape (n_obs, n_phenotypes).

    Returns
    -------
    float
        Log-likelihood.
    """
    # DirMult(n | α) = Γ(Σα) / Γ(N + Σα) × Π_p Γ(n_p + α_p) / Γ(α_p)
    # where N = Σ n_p

    N = counts.sum(axis=1)  # (n_obs,)
    alpha_sum = alpha_mu.sum(axis=1)  # (n_obs,)

    ll = (gammaln(alpha_sum) - gammaln(N + alpha_sum) +
          np.sum(gammaln(counts + alpha_mu) - gammaln(alpha_mu), axis=1))

    return np.sum(ll)


def _dm_log_likelihood_per_obs(
    counts: np.ndarray,
    alpha_mu: np.ndarray,
) -> np.ndarray:
    """Compute per-observation Dirichlet-Multinomial log-likelihood.

    Parameters
    ----------
    counts : np.ndarray
        Count matrix, shape (n_obs, n_phenotypes).
    alpha_mu : np.ndarray
        Dirichlet parameters α·μ, shape (n_obs, n_phenotypes).

    Returns
    -------
    np.ndarray
        Per-observation log-likelihoods, shape (n_obs,).
    """
    N = counts.sum(axis=1)  # (n_obs,)
    alpha_sum = alpha_mu.sum(axis=1)  # (n_obs,)

    ll = (gammaln(alpha_sum) - gammaln(N + alpha_sum) +
          np.sum(gammaln(counts + alpha_mu) - gammaln(alpha_mu), axis=1))

    return ll


def eb_spike_slab_pip(
        beta: np.ndarray,
        se: np.ndarray,
        max_iter: int = 200,
        tol: float = 1e-6
) -> Tuple[np.ndarray, float, float]:
    """
    Empirical-Bayes spike-and-slab on normal means with known SEs.

    Model (per coefficient j):
        beta_j | slab ~ N(0, se_j^2 + tau2)
        beta_j | spike ~ N(0, se_j^2)
        P(slab) = 1 - pi0

    Returns
    -------
    pip : array, P(effect != 0) per coefficient
    pi0 : estimated spike weight
    tau2 : estimated slab variance
    """
    beta = np.asarray(beta, float)
    se = np.asarray(se, float)

    mask = np.isfinite(beta) & np.isfinite(se) & (se > 0)
    pip = np.full(beta.shape, np.nan, dtype=float)

    if mask.sum() == 0:
        return pip, np.nan, np.nan

    b = beta[mask]
    v = se[mask] ** 2

    # --- init ---
    # Rough init: assume many are null
    pi0 = 0.9
    tau2 = max(1e-8, float(np.var(b) - np.mean(v)))

    for _ in range(max_iter):
        s0 = np.sqrt(v)
        s1 = np.sqrt(v + tau2)

        f0 = norm.pdf(b, loc=0.0, scale=s0)
        f1 = norm.pdf(b, loc=0.0, scale=s1)

        denom = pi0 * f0 + (1.0 - pi0) * f1 + 1e-300
        w1 = (1.0 - pi0) * f1 / denom  # posterior prob slab = PIP

        pi0_new = float(np.clip(1.0 - np.mean(w1), 1e-3, 1.0 - 1e-3))

        # Update tau2 by 1D optimization (v differs per coefficient)
        def neg_Q(tau2_candidate: float) -> float:
            tau2_candidate = max(float(tau2_candidate), 1e-12)
            s2 = v + tau2_candidate
            # weighted expected log-lik under slab
            return -float(np.sum(w1 * (-0.5*np.log(2*np.pi*s2) - 0.5*(b*b)/s2)))

        upper = max(1e-6, float(np.quantile(b*b, 0.95)))
        res = minimize_scalar(neg_Q, bounds=(1e-12, upper), method="bounded")
        tau2_new = float(max(res.x, 1e-12))

        if abs(pi0_new - pi0) < tol and abs(tau2_new - tau2) < tol:
            pi0, tau2 = pi0_new, tau2_new
            break

        pi0, tau2 = pi0_new, tau2_new

    # final PIP
    s0 = np.sqrt(v)
    s1 = np.sqrt(v + tau2)
    f0 = norm.pdf(b, 0.0, s0)
    f1 = norm.pdf(b, 0.0, s1)
    denom = pi0 * f0 + (1.0 - pi0) * f1 + 1e-300
    w1 = (1.0 - pi0) * f1 / denom

    pip[mask] = w1
    return pip, pi0, tau2


def estimate_alpha_per_ccr(
    data: DirichletMultinomialData,
    mu: np.ndarray,  # (n_obs, K) fitted mean probs per observation
    ccr_ids: np.ndarray,
    alpha_bounds=(1e-3, 1e5),
):
    """
    Profile-likelihood estimate of Dirichlet-Multinomial concentration alpha per CCR,
    holding mu fixed.
    """
    counts = np.asarray(data.counts, int)
    mu = np.asarray(mu, float)
    K = counts.shape[1]

    # small smoothing to avoid mu=0
    eps = 1e-12
    mu = (mu + eps) / (1.0 + eps * K)

    n_ccrs = int(np.max(ccr_ids)) + 1
    alpha_hat = np.full(n_ccrs, np.nan, float)

    def dm_ll(alpha, counts_sub, mu_sub):
        a = alpha * mu_sub
        N = counts_sub.sum(axis=1)
        alpha_sum = a.sum(axis=1)
        ll = (
            gammaln(alpha_sum) - gammaln(alpha_sum + N)
            + np.sum(gammaln(counts_sub + a) - gammaln(a), axis=1)
        )
        return float(np.sum(ll))

    for c in range(n_ccrs):
        idx = np.where(ccr_ids == c)[0]
        if idx.size == 0:
            continue

        counts_sub = counts[idx]
        mu_sub = mu[idx]

        def neg_ll(log_alpha):
            alpha = np.exp(log_alpha)
            return -dm_ll(alpha, counts_sub, mu_sub)

        lo, hi = np.log(alpha_bounds[0]), np.log(alpha_bounds[1])
        res = minimize_scalar(neg_ll, bounds=(lo, hi), method="bounded")
        alpha_hat[c] = float(np.exp(res.x))

    return alpha_hat


def fit_dm_model(
    data: DirichletMultinomialData,
    alpha_init: float = 10.0,
    include_ccr_effects: bool = True,
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: bool = False,
    weights: Optional[np.ndarray] = None,
    starting_freqs: Optional[np.ndarray] = None,
    l2_penalty: float = 0.0,
    use_finite_diff_hessian: bool = False,
) -> DirichletMultinomialResult:
    """Fit Dirichlet-Multinomial model via maximum likelihood.

    Parameters
    ----------
    data : DirichletMultinomialData
        Prepared data.
    alpha_init : float
        Initial value for concentration parameter.
    include_ccr_effects : bool
        Whether to include CCR random effects.
    max_iter : int
        Maximum iterations for optimization.
    tol : float
        Convergence tolerance.
    verbose : bool
        Print progress.
    weights : np.ndarray, optional
        Observation weights of shape (n_obs, n_phenotypes). Used to adjust
        for differences between sequencing depth and actual cell counts
        (e.g., from FACS data). Weights are applied to the log-likelihood.
    starting_freqs : np.ndarray, optional
        Starting frequencies for each observation, shape (n_obs,). Used as
        a prior to account for baseline CCD frequencies in the unsorted pool.
        Currently used as observation weights in the likelihood.
    l2_penalty : float
        L2 regularization strength for beta_elm and beta_ccr parameters.
        Default is 0.0 (no regularization) for unbiased inference.
        Can be increased for numerical stability if needed.
    use_finite_diff_hessian : bool
        If True, compute covariance matrix using finite differences of the
        analytic gradient (slower but more accurate). If False (default),
        use the L-BFGS-B inverse Hessian approximation (faster but less accurate).

    Returns
    -------
    DirichletMultinomialResult
        Fitted model results.
    """
    n_obs = data.n_obs
    n_phenotypes = data.n_phenotypes
    n_elms = data.n_elms
    n_ccrs = data.n_ccrs

    ref_phenotype = _get_reference_phenotype(data.phenotype_names)
    ref_idx = data.phenotype_names.index(ref_phenotype)
    n_non_ref = n_phenotypes - 1

    # Parameter dimensions
    # Use (n_ccrs - 1) free parameters for CCR effects to ensure identifiability
    # The last CCR row is derived from sum-to-zero constraint
    n_beta_0 = n_non_ref
    n_beta_elm = n_elms * n_non_ref
    n_beta_ccr = (n_ccrs - 1) * n_non_ref if include_ccr_effects else 0
    n_params = 1 + n_beta_0 + n_beta_elm + n_beta_ccr  # +1 for log(alpha)

    def unpack_params(params):
        idx = 0
        log_alpha = params[idx]; idx += 1
        beta_0 = params[idx:idx+n_beta_0]; idx += n_beta_0
        beta_elm = params[idx:idx+n_beta_elm].reshape(n_elms, n_non_ref); idx += n_beta_elm

        if include_ccr_effects:
            # Identifiable parameterization: use (n_ccrs - 1) free rows
            # Derive last row from sum-to-zero constraint
            raw_ccr = params[idx:idx+n_beta_ccr].reshape(n_ccrs - 1, n_non_ref)
            last_row = -raw_ccr.sum(axis=0, keepdims=True)
            beta_ccr = np.vstack([raw_ccr, last_row])  # shape (n_ccrs, n_non_ref)
        else:
            beta_ccr = np.zeros((n_ccrs, n_non_ref))

        return np.exp(log_alpha), beta_0, beta_elm, beta_ccr

    # Compute combined observation weights if provided
    obs_weights = np.ones(n_obs)
    if starting_freqs is not None:
        obs_weights = obs_weights * starting_freqs
    if weights is not None:
        # For phenotype-specific weights, we use the mean across phenotypes
        # as an observation-level weight
        obs_weights = obs_weights * np.mean(weights, axis=1)

    # Validation
    obs_weights = np.asarray(obs_weights, dtype=float)

    if np.any(obs_weights < 0):
        raise ValueError("obs_weights must be non-negative (check starting_freqs/weights).")

    s = float(obs_weights.sum())
    if not np.isfinite(s) or s <= 0:
        raise ValueError("obs_weights sum must be positive and finite.")

    # Normalize observation weights
    obs_weights = obs_weights / obs_weights.sum() * n_obs

    def neg_log_likelihood(params):
        """Negative log-likelihood used by the optimizer (includes optional L2 penalty)."""
        alpha, beta_0, beta_elm, beta_ccr = unpack_params(params)

        alpha_mu = _compute_alpha_mu(
            beta_0, beta_elm, beta_ccr, alpha,
            data.elm_matrix, data.ccr_ids, ref_idx
        )

        # Weighted log-likelihood
        ll_per_obs = _dm_log_likelihood_per_obs(data.counts, alpha_mu)
        ll = float(np.sum(obs_weights * ll_per_obs))

        # L2 penalty (ONLY beta_elm and beta_ccr; do NOT penalize beta_0 by default)
        penalty = 0.0
        if l2_penalty > 0:
            penalty += float(l2_penalty) * float(np.sum(beta_elm ** 2))

            if include_ccr_effects:
                # Penalize the *full* CCR matrix, even though we store only raw rows.
                # raw_ccr: (n_ccrs-1, n_non_ref); last_row = -sum(raw_ccr)
                raw_ccr = params[1 + n_beta_0 + n_beta_elm:].reshape(n_ccrs - 1, n_non_ref)
                last_row = -raw_ccr.sum(axis=0, keepdims=True)
                penalty += float(l2_penalty) * float(np.sum(raw_ccr ** 2) + np.sum(last_row ** 2))

        return -ll + penalty

    def neg_ll_and_grad(params):
        """Compute negative log-likelihood and analytic gradient (for that same objective).

        Returns
        -------
        nll : float
            Negative log-likelihood (+ optional L2 penalty).
        grad : np.ndarray
            Gradient of nll w.r.t. params (same shape as params).
        """
        alpha, beta_0, beta_elm, beta_ccr = unpack_params(params)

        # Indices of non-reference phenotypes in the K-vector
        non_ref_idx = [k for k in range(n_phenotypes) if k != ref_idx]

        # ---- Forward pass: logits -> probabilities ----
        eta = (beta_0[np.newaxis, :] +
               data.elm_matrix @ beta_elm +
               beta_ccr[data.ccr_ids, :])  # (n_obs, K-1)

        full_z = np.zeros((n_obs, n_phenotypes))
        full_z[:, non_ref_idx] = eta

        # Use mu0 for softmax Jacobian; then apply your smooth positivity transform.
        mu0 = softmax(full_z, axis=1)  # (n_obs, K)

        eps = 1e-12
        denom = 1.0 + eps * n_phenotypes
        mu = (mu0 + eps) / denom  # smoothed probabilities (still sum to 1)

        a = alpha * mu  # Dirichlet parameters per obs (n_obs, K)
        N = data.counts.sum(axis=1)  # (n_obs,)
        alpha_sum = a.sum(axis=1)  # (n_obs,) ~ alpha

        # ---- Log-likelihood (weighted) ----
        ll_per_obs = (
                gammaln(alpha_sum) - gammaln(N + alpha_sum)
                + np.sum(gammaln(data.counts + a) - gammaln(a), axis=1)
        )
        ll = float(np.sum(obs_weights * ll_per_obs))

        # ---- Penalty (match neg_log_likelihood exactly) ----
        penalty = 0.0
        if l2_penalty > 0:
            penalty += float(l2_penalty) * float(np.sum(beta_elm ** 2))

            if include_ccr_effects:
                raw_ccr = params[1 + n_beta_0 + n_beta_elm:].reshape(n_ccrs - 1, n_non_ref)
                last_row = -raw_ccr.sum(axis=0, keepdims=True)
                penalty += float(l2_penalty) * float(np.sum(raw_ccr ** 2) + np.sum(last_row ** 2))

        nll = -ll + penalty

        # ---- Gradients of the (weighted) LOG-LIKELIHOOD ----
        # g_ip = d ll_i / d a_ip
        base = (digamma(alpha_sum) - digamma(alpha_sum + N))[:, None]  # (n_obs, 1)
        g = base + digamma(data.counts + a) - digamma(a)  # (n_obs, K)

        # For d/dz, we must backprop through:
        # mu0 = softmax(z); mu = (mu0 + eps)/denom
        # d mu / d mu0 = 1/denom
        #
        # d ll / d z_k = (alpha/denom) * mu0_k * (g_k - sum_p g_p * mu0_p)
        g_dot_mu0 = np.sum(g * mu0, axis=1, keepdims=True)  # (n_obs, 1)
        dLL_dz = (alpha / denom) * mu0 * (g - g_dot_mu0)  # (n_obs, K)
        dLL_deta = dLL_dz[:, non_ref_idx]  # (n_obs, K-1)

        # Apply observation weights
        dLL_deta_w = obs_weights[:, None] * dLL_deta

        # Grad ll w.r.t beta_0 and beta_elm
        grad_ll_beta_0 = dLL_deta_w.sum(axis=0)  # (K-1,)
        grad_ll_beta_elm = data.elm_matrix.T @ dLL_deta_w  # (n_elms, K-1)

        # Grad ll w.r.t beta_ccr (full), then map to raw parameterization
        if include_ccr_effects:
            grad_ll_beta_ccr_full = np.zeros((n_ccrs, n_non_ref))
            np.add.at(grad_ll_beta_ccr_full, data.ccr_ids, dLL_deta_w)

            # beta_ccr[-1] = -sum(raw_ccr), so
            # d ll / d raw_ccr[j] = d ll / d beta_ccr[j] - d ll / d beta_ccr[-1]
            grad_ll_beta_ccr_raw = grad_ll_beta_ccr_full[:-1, :] - grad_ll_beta_ccr_full[-1:, :]
        else:
            grad_ll_beta_ccr_raw = np.zeros((0, n_non_ref))

        # Grad ll w.r.t log(alpha):
        # d ll / d alpha = sum_p g_ip * mu_ip  (uses smoothed mu)
        g_dot_mu_alpha = np.sum(g * mu, axis=1)  # (n_obs,)
        grad_ll_log_alpha = alpha * float(np.sum(obs_weights * g_dot_mu_alpha))

        # ---- Convert to gradients of the OBJECTIVE: nll = -ll + penalty ----
        grad = np.zeros(n_params, dtype=float)

        # log(alpha): no penalty term
        grad[0] = -grad_ll_log_alpha

        # beta_0: no penalty term (unless you explicitly want it)
        grad[1:1 + n_beta_0] = -grad_ll_beta_0

        # beta_elm: add ridge gradient
        grad_beta_elm_nll = -grad_ll_beta_elm
        if l2_penalty > 0:
            grad_beta_elm_nll = grad_beta_elm_nll + 2.0 * float(l2_penalty) * beta_elm
        grad[1 + n_beta_0:1 + n_beta_0 + n_beta_elm] = grad_beta_elm_nll.ravel()

        # beta_ccr raw: add correct ridge gradient under the "full beta_ccr" penalty
        if include_ccr_effects:
            grad_beta_ccr_raw_nll = -grad_ll_beta_ccr_raw

            if l2_penalty > 0:
                raw_ccr = params[1 + n_beta_0 + n_beta_elm:].reshape(n_ccrs - 1, n_non_ref)
                last_row = -raw_ccr.sum(axis=0, keepdims=True)

                # If penalty is λ(||raw||^2 + ||last||^2), then:
                # d penalty / d raw_j = 2λ (raw_j - last)
                grad_pen_raw = 2.0 * float(l2_penalty) * (raw_ccr - last_row)
                grad_beta_ccr_raw_nll = grad_beta_ccr_raw_nll + grad_pen_raw

            grad[1 + n_beta_0 + n_beta_elm:] = grad_beta_ccr_raw_nll.ravel()

        return nll, grad

    def _debug_grad_check(p0, eps=1e-6, n_check=25, seed=0, tol_rel=1e-4):
        """Finite-difference check on a random subset of parameters."""
        rng = np.random.default_rng(seed)
        idxs = rng.choice(p0.size, size=min(int(n_check), p0.size), replace=False)

        f0, g0 = neg_ll_and_grad(p0)
        max_rel = 0.0

        for j in idxs:
            p_hi = p0.copy();
            p_hi[j] += eps
            p_lo = p0.copy();
            p_lo[j] -= eps
            f_hi = neg_ll_and_grad(p_hi)[0]
            f_lo = neg_ll_and_grad(p_lo)[0]
            g_num = (f_hi - f_lo) / (2.0 * eps)

            rel = abs(g0[j] - g_num) / (1.0 + abs(g_num))
            max_rel = max(max_rel, rel)

            if verbose:
                print(f"gradcheck idx={j:5d}  ana={g0[j]: .3e}  num={g_num: .3e}  rel={rel: .3e}")

        if verbose:
            print(f"Gradient check: max_rel={max_rel:.3e} (tol={tol_rel:.3e})")

        if max_rel > tol_rel:
            raise RuntimeError(f"Gradient check failed: max_rel={max_rel:.3e} > tol={tol_rel:.3e}")

    # Initialize parameters from empirical composition
    params_init = np.zeros(n_params)
    params_init[0] = np.log(alpha_init)  # log(alpha)

    # Smart initialization for beta_0 from marginal composition
    counts_sum = data.counts.sum(axis=0) + 0.5  # add smoothing
    p_hat = counts_sum / counts_sum.sum()
    non_ref_idx = [k for k in range(n_phenotypes) if k != ref_idx]
    beta_0_init = np.log(p_hat[non_ref_idx]) - np.log(p_hat[ref_idx])
    params_init[1:1+n_beta_0] = beta_0_init

    # Optional: initialize CCR effects from per-CCR deviations
    if include_ccr_effects and n_ccrs > 1:
        # Compute per-CCR composition
        ccr_comps = []
        for c in range(n_ccrs):
            mask = (data.ccr_ids == c)
            if mask.sum() > 0:
                ccr_sum = data.counts[mask, :].sum(axis=0) + 0.5
                ccr_p = ccr_sum / ccr_sum.sum()
                ccr_log_odds = np.log(ccr_p[non_ref_idx]) - np.log(ccr_p[ref_idx])
                ccr_comps.append(ccr_log_odds - beta_0_init)  # deviation from baseline
            else:
                ccr_comps.append(np.zeros(n_non_ref))
        ccr_comps = np.array(ccr_comps)  # (n_ccrs, n_non_ref)

        # Enforce sum-to-zero and store first (n_ccrs-1) rows
        ccr_comps = ccr_comps - ccr_comps.mean(axis=0, keepdims=True)
        params_init[1+n_beta_0+n_beta_elm:] = ccr_comps[:-1, :].flatten()

    # Optimize with analytic gradient
    if verbose:
        print(f"Fitting DM model with {n_params} parameters...")
        print(f"  {n_obs} observations, {n_elms} ELMs, {n_ccrs} CCRs")
        print(f"  Using analytic gradient for faster convergence")

    # Keep alpha in a reasonable numeric range (prevents overflow/underflow pathologies)
    bounds = [(np.log(1e-3), np.log(1e5))] + [(None, None)] * (n_params - 1)

    # Optional: sanity-check analytic gradient before optimizing
    # (best to run on small synthetic problems or small real subsets)
    # _debug_grad_check(params_init)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(
            fun=lambda p: neg_ll_and_grad(p)[0],
            jac=lambda p: neg_ll_and_grad(p)[1],
            x0=params_init,
            method='L-BFGS-B',
            bounds=bounds,  # <--- ADD THIS
            options={'maxiter': max_iter, 'ftol': tol, 'disp': verbose}
        )

    # Extract fitted parameters
    alpha, beta_0, beta_elm, beta_ccr = unpack_params(result.x)

    # Extract inverse Hessian from L-BFGS-B optimizer
    # ============================================================================
    # WARNING: L-BFGS-B hess_inv is a limited-memory quasi-Newton approximation
    # ============================================================================
    # This is NOT the observed information matrix. It can be "good enough" for
    # rough standard errors but has limitations:
    #   - It's especially fragile with near-singular directions
    #   - It's sensitive to parameter scaling
    #   - It can appear reasonable while still being wrong
    #
    # For publication-quality inference, consider:
    #   1. Parametric bootstrap (most robust, slowest)
    #   2. Sandwich/cluster-robust covariance (accounts for CCD clustering)
    #   3. Observed Hessian from finite differences of analytic gradient
    #
    # The identifiable CCR parameterization and analytic gradient help make
    # this approximation more reliable, but validation is still recommended.
    # ============================================================================
    Hinv = None
    cov_beta_elm_full = None
    se_beta_elm = None

    if l2_penalty > 0:
        warnings.warn(
            "l2_penalty>0: penalized optimization. Hessian-based SEs/p-values are not "
            "valid for inference; returning se_beta_elm=None and cov_beta_elm_full=None.",
            RuntimeWarning,
        )
    else:
        try:
            if use_finite_diff_hessian:
                if verbose:
                    print("Computing covariance matrix via finite differences...")

                temp_result = DirichletMultinomialResult(
                    beta_0=beta_0,
                    beta_elm=beta_elm,
                    beta_ccr=beta_ccr,
                    alpha=alpha,
                    se_beta_elm=None,
                    cov_beta_elm_full=None,
                    log_likelihood=0.0,
                    converged=result.success,
                    data=data,
                )

                Hinv = compute_observed_fisher_information(
                    temp_result,
                    obs_weights=obs_weights,
                    include_ccr_effects=include_ccr_effects,
                    verbose=verbose,
                    fd_scheme="central",  # "forward" for speed
                )
            else:
                Hinv = np.asarray(result.hess_inv.todense(), dtype=float)

            n_elm_params = n_elms * n_non_ref
            elm_start = 1 + n_beta_0
            elm_end = elm_start + n_elm_params

            cov_beta_elm_full = Hinv[elm_start:elm_end, elm_start:elm_end]

            se_elm_flat = np.sqrt(np.clip(np.diag(cov_beta_elm_full), 0.0, None))
            se_beta_elm = se_elm_flat.reshape(n_elms, n_non_ref)

        except Exception:
            cov_beta_elm_full = None
            se_beta_elm = None

    # Weighted log-likelihood at MLE (NO L2 penalty)
    alpha_mu = _compute_alpha_mu(
        beta_0, beta_elm, beta_ccr, alpha,
        data.elm_matrix, data.ccr_ids, ref_idx
    )
    final_ll = float(np.sum(obs_weights * _dm_log_likelihood_per_obs(data.counts, alpha_mu)))

    return DirichletMultinomialResult(
        beta_0=beta_0,
        beta_elm=beta_elm,
        beta_ccr=beta_ccr,
        alpha=alpha,
        se_beta_elm=se_beta_elm,
        cov_beta_elm_full=cov_beta_elm_full,
        log_likelihood=final_ll,
        converged=result.success,
        data=data,
    )


def _get_reference_phenotype(phenotype_names):
    """Get the reference phenotype from a list of phenotype names."""
    if REFERENCE_PHENOTYPE in phenotype_names:
        return REFERENCE_PHENOTYPE
    elif REFERENCE_PHENOTYPE_ALT in phenotype_names:
        return REFERENCE_PHENOTYPE_ALT
    else:
        # Fall back to first phenotype with "Low" in name, or first phenotype
        for p in phenotype_names:
            if "Low" in p and ("Naive" in p or "Naïve" in p):
                return p
        return phenotype_names[0]


def prepare_dm_data(
    counts_df: pd.DataFrame,
    sample_meta: pd.DataFrame,
    candidate_meta: pd.DataFrame,
    exp_cond: str = "CAR:Raji",
    min_total: int = 10,
    min_elm_freq: float = 0.05,
) -> DirichletMultinomialData:
    """Prepare data for Dirichlet-Multinomial model fitting.

    Parameters
    ----------
    counts_df : pd.DataFrame
        Count matrix with CCDs as rows and samples as columns.
    sample_meta : pd.DataFrame
        Sample metadata with columns: sample_id, Donor, ExpCond, Tsubset, PD1Status, Replicate.
    candidate_meta : pd.DataFrame
        Candidate metadata with ELMCategory column.
    exp_cond : str
        Experimental condition to filter to.
    min_total : int
        Minimum total count per CCD-CCR to include.
    min_elm_freq : float
        Minimum ELM frequency to include.

    Returns
    -------
    DirichletMultinomialData
        Prepared data for model fitting.
    """
    # Filter to experimental condition
    cond_meta = sample_meta[sample_meta['ExpCond'].str.contains(exp_cond, na=False, regex=False)].copy()
    cond_samples = cond_meta['sample_id'].tolist()
    cond_counts = counts_df[cond_samples].copy()

    # Create CCR identifier
    cond_meta['CCR'] = cond_meta['Donor'].astype(str) + '_' + cond_meta['Replicate'].astype(str)

    # Get unique CCRs
    ccrs = sorted(cond_meta['CCR'].unique())
    ccr_to_idx = {ccr: i for i, ccr in enumerate(ccrs)}

    # Normalize phenotype strings
    cond_meta["PD1Status"] = cond_meta["PD1Status"].apply(normalize_phenotype)
    cond_meta["Tsubset"] = cond_meta["Tsubset"].apply(normalize_phenotype)

    # Create phenotype identifier
    cond_meta["phenotype"] = (cond_meta["Tsubset"] + "_" + cond_meta["PD1Status"]).apply(normalize_phenotype)

    # Build mapping from sample to (CCR, phenotype)
    sample_to_ccr = dict(zip(cond_meta['sample_id'], cond_meta['CCR']))
    sample_to_pheno = dict(zip(cond_meta['sample_id'], cond_meta['phenotype']))

    # Fast lookup for phenotype index (use normalized keys)
    pheno_to_idx = {normalize_phenotype(p): i for i, p in enumerate(PHENOTYPES)}

    # Only consider CCDs that will actually be eligible for modeling
    eligible_ccds = [ccd for ccd in cond_counts.index if ccd in candidate_meta.index]
    if len(eligible_ccds) == 0:
        raise ValueError("No CCDs overlap between counts_df and candidate_meta after filtering.")

    # Count ELM frequencies on eligible CCDs
    elm_counts = {}
    for ccd in eligible_ccds:
        for elm in split_elm_list(candidate_meta.loc[ccd, "ELMCategory"]):
            elm_counts[elm] = elm_counts.get(elm, 0) + 1

    # Filter ELMs by frequency (using eligible CCD denominator)
    n_ccds = len(eligible_ccds)
    active_elms = sorted(
        [elm for elm, count in elm_counts.items() if (count / n_ccds) >= min_elm_freq]
    )

    # Build data arrays
    count_rows = []
    elm_rows = []
    ccr_ids = []
    ccd_ids = []

    for ccd in cond_counts.index:
        if ccd not in candidate_meta.index:
            continue

        # Parse ELMs for this CCD
        ccd_elms = set(split_elm_list(candidate_meta.loc[ccd, 'ELMCategory']))
        elm_vec = np.array([1 if elm in ccd_elms else 0 for elm in active_elms])

        for ccr in ccrs:
            # Get samples for this CCR
            ccr_samples = [s for s in cond_samples if sample_to_ccr[s] == ccr]

            # Build count vector in phenotype order
            count_vec = np.zeros(N_PHENOTYPES)
            for s in ccr_samples:
                pheno = normalize_phenotype(sample_to_pheno[s])
                if pheno not in pheno_to_idx:
                    # Unknown phenotype label; skip rather than crashing
                    continue
                pheno_idx = pheno_to_idx[pheno]
                count_vec[pheno_idx] += float(cond_counts.loc[ccd, s])

            count_vec = np.asarray(np.round(count_vec), dtype=int)

            total = count_vec.sum()

            # Skip if below minimum
            if total < min_total:
                continue

            count_rows.append(count_vec)
            elm_rows.append(elm_vec)
            ccr_ids.append(ccr_to_idx[ccr])
            ccd_ids.append(ccd)

    counts = np.array(count_rows)
    elm_matrix = np.array(elm_rows)
    ccr_ids = np.array(ccr_ids)
    ccd_ids = np.array(ccd_ids)
    totals = counts.sum(axis=1)

    return DirichletMultinomialData(
        counts=counts,
        elm_matrix=elm_matrix,
        ccr_ids=ccr_ids,
        ccd_ids=ccd_ids,
        totals=totals,
        elm_names=active_elms,
        phenotype_names=PHENOTYPES.copy(),
    )


def simulate_dm_data(
    n_ccds: int = 500,
    n_ccrs: int = 6,
    n_elms: int = 30,
    elm_prevalence: float = 0.15,
    alpha: float = 20.0,
    baseline_probs: Optional[np.ndarray] = None,
    elm_effects: Optional[Dict[str, Dict[str, float]]] = None,
    total_count_mean: float = 5000.0,
    total_count_cv: float = 0.8,
    ccr_effect_sd: float = 0.2,
    seed: int = 42,
) -> Tuple[DirichletMultinomialData, Dict]:
    """Generate synthetic data from the Dirichlet-Multinomial model.

    Parameters
    ----------
    n_ccds : int
        Number of CCDs.
    n_ccrs : int
        Number of CCRs.
    n_elms : int
        Number of ELMs.
    elm_prevalence : float
        Probability of a CCD having each ELM.
    alpha : float
        True concentration parameter.
    baseline_probs : np.ndarray, optional
        Baseline phenotype probabilities. If None, uses uniform.
    elm_effects : dict, optional
        True ELM effects: {elm_name: {phenotype: effect}}.
    total_count_mean : float
        Mean total count per CCD-CCR.
    total_count_cv : float
        CV of total counts.
    ccr_effect_sd : float
        SD of CCR fixed effects.
    seed : int
        Random seed.

    Returns
    -------
    data : DirichletMultinomialData
        Simulated data.
    truth : dict
        Ground truth parameters.
    """
    rng = np.random.default_rng(seed)

    elm_names = [f"ELM_{i}" for i in range(n_elms)]

    if baseline_probs is None:
        baseline_probs = np.ones(N_PHENOTYPES) / N_PHENOTYPES

    if elm_effects is None:
        elm_effects = {}

    # Generate ELM matrix
    elm_matrix_full = rng.binomial(1, elm_prevalence, size=(n_ccds, n_elms))

    # Convert elm_effects to matrix form
    ref_idx = PHENOTYPES.index(REFERENCE_PHENOTYPE)
    non_ref_phenos = [p for p in PHENOTYPES if p != REFERENCE_PHENOTYPE]

    beta_elm_true = np.zeros((n_elms, N_PHENOTYPES - 1))
    for elm_name, pheno_effects in elm_effects.items():
        if elm_name in elm_names:
            elm_idx = elm_names.index(elm_name)
            for pheno, effect in pheno_effects.items():
                norm_pheno = normalize_phenotype(pheno)
                if norm_pheno in non_ref_phenos:
                    pheno_idx = non_ref_phenos.index(norm_pheno)
                    beta_elm_true[elm_idx, pheno_idx] = effect

    # Generate CCR effects
    beta_ccr_true = rng.normal(0, ccr_effect_sd, size=(n_ccrs, N_PHENOTYPES - 1))

    # Baseline log-odds
    beta_0_true = np.log(baseline_probs[:-1] / baseline_probs[-1])
    # Adjust for reference phenotype
    log_odds_base = np.log(baseline_probs + 1e-10)
    log_odds_base -= log_odds_base[ref_idx]
    beta_0_true = np.delete(log_odds_base, ref_idx)

    # Generate data
    count_rows = []
    elm_rows = []
    ccr_ids = []
    ccd_ids = []

    for i in range(n_ccds):
        for c in range(n_ccrs):
            # Total count for this CCD-CCR
            log_total = np.log(total_count_mean) + rng.normal(0, total_count_cv)
            total = max(10, int(np.exp(log_total)))

            # Compute phenotype probabilities
            log_odds = beta_0_true.copy()
            log_odds += elm_matrix_full[i, :] @ beta_elm_true
            log_odds += beta_ccr_true[c, :]

            # Insert 0 for reference
            full_log_odds = np.zeros(N_PHENOTYPES)
            non_ref_idx = [j for j in range(N_PHENOTYPES) if j != ref_idx]
            full_log_odds[non_ref_idx] = log_odds

            mu = softmax(full_log_odds)

            # Sample from Dirichlet-Multinomial
            alpha_mu = alpha * mu
            pi = rng.dirichlet(alpha_mu)
            counts = rng.multinomial(total, pi)

            count_rows.append(counts)
            elm_rows.append(elm_matrix_full[i, :])
            ccr_ids.append(c)
            ccd_ids.append(f"CCD_{i}")

    counts = np.array(count_rows)
    elm_matrix = np.array(elm_rows)
    ccr_ids = np.array(ccr_ids)
    ccd_ids = np.array(ccd_ids)
    totals = counts.sum(axis=1)

    data = DirichletMultinomialData(
        counts=counts,
        elm_matrix=elm_matrix,
        ccr_ids=ccr_ids,
        ccd_ids=ccd_ids,
        totals=totals,
        elm_names=elm_names,
    )

    truth = {
        'alpha': alpha,
        'beta_0': beta_0_true,
        'beta_elm': beta_elm_true,
        'beta_ccr': beta_ccr_true,
        'elm_effects': elm_effects,
        'baseline_probs': baseline_probs,
    }

    return data, truth
