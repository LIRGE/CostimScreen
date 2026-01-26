"""
Model comparison framework for costimulatory screen analysis.

This module provides functions for comparing different modeling approaches
on synthetic data generated from the Dirichlet-Multinomial ground truth.

The comparison evaluates:
1. Type I error (FPR under null)
2. Power (TPR with true effects)
3. Effect size estimation accuracy
4. Calibration of p-values

Models compared:
- Joint NB-GLM (src/costim_screen)
- Mann-Whitney on residuals (Pooled-CAR-T-Analysis approach)
- Per-phenotype NB-GLM (Costim-NBGLM-Pipeline approach)
- Dirichlet-Multinomial (new principled model)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time
import warnings

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import mannwhitneyu, norm
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from .dirichlet_multinomial import (
    DirichletMultinomialData,
    DirichletMultinomialResult,
    eb_spike_slab_pip,
    estimate_alpha_per_ccr,
    simulate_dm_data,
    fit_dm_model,
    PHENOTYPES,
)


@dataclass
class MethodResult:
    """Results from one analysis method."""
    method_name: str
    pvalues: pd.DataFrame  # (ELM × phenotype)
    effects: pd.DataFrame  # Effect estimates
    significant: pd.DataFrame  # Boolean (after FDR)
    runtime_seconds: float = 0.0


@dataclass
class ComparisonMetrics:
    """Metrics for comparing methods."""
    method: str
    n_true_positives: int
    n_false_positives: int
    n_true_negatives: int
    n_false_negatives: int
    tpr: float  # True positive rate (power)
    fpr: float  # False positive rate
    precision: float
    effect_bias: float  # Mean(estimated - true) for true effects
    effect_rmse: float  # RMSE of effect estimates
    runtime: float


def compute_metrics(
    method_result: MethodResult,
    true_effects: Dict[str, Dict[str, float]],
    elm_names: List[str],
) -> ComparisonMetrics:
    """Compute comparison metrics for a method's results.

    Parameters
    ----------
    method_result : MethodResult
        Results from one method.
    true_effects : dict
        Ground truth effects: {elm_name: {phenotype: effect}}.
    elm_names : list
        List of all ELM names.

    Returns
    -------
    ComparisonMetrics
        Computed metrics.
    """
    tp, fp, tn, fn = 0, 0, 0, 0
    effect_errors = []

    for elm in elm_names:
        if elm not in method_result.significant.index:
            continue

        for pheno in PHENOTYPES:
            if pheno not in method_result.significant.columns:
                continue

            # Get significance call
            sig = method_result.significant.loc[elm, pheno]
            if pd.isna(sig):
                continue

            # Get true effect
            true_effect = 0.0
            if elm in true_effects and pheno in true_effects[elm]:
                true_effect = true_effects[elm][pheno]

            has_true_effect = abs(true_effect) > 0.01

            if has_true_effect:
                if sig:
                    tp += 1
                else:
                    fn += 1
            else:
                if sig:
                    fp += 1
                else:
                    tn += 1

            # Effect estimation error (only for true effects)
            if has_true_effect and elm in method_result.effects.index:
                est_effect = method_result.effects.loc[elm, pheno]
                if pd.notna(est_effect):
                    effect_errors.append(est_effect - true_effect)

    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    precision = tp / max(tp + fp, 1)

    effect_bias = np.mean(effect_errors) if effect_errors else np.nan
    effect_rmse = np.sqrt(np.mean(np.array(effect_errors)**2)) if effect_errors else np.nan

    return ComparisonMetrics(
        method=method_result.method_name,
        n_true_positives=tp,
        n_false_positives=fp,
        n_true_negatives=tn,
        n_false_negatives=fn,
        tpr=tpr,
        fpr=fpr,
        precision=precision,
        effect_bias=effect_bias,
        effect_rmse=effect_rmse,
        runtime=method_result.runtime_seconds,
    )


def _dm_data_to_long_format(data: DirichletMultinomialData) -> pd.DataFrame:
    """Convert DM data to long format for NB-GLM fitting."""
    records = []
    for i in range(data.n_obs):
        ccd_id = data.ccd_ids[i]
        ccr_id = data.ccr_ids[i]

        for p, pheno in enumerate(data.phenotype_names):
            record = {
                'CandidateID': ccd_id,
                'CCR': f'CCR_{ccr_id}',
                'phenotype': pheno,
                'count': data.counts[i, p],
            }
            # Add ELM indicators
            for e, elm_name in enumerate(data.elm_names):
                record[elm_name] = data.elm_matrix[i, e]
            records.append(record)

    df = pd.DataFrame(records)

    # Add library size (total per sample = sum over CCDs for that CCR-phenotype)
    lib_sizes = df.groupby(['CCR', 'phenotype'])['count'].transform('sum')
    df['lib_size'] = lib_sizes
    df['offset'] = np.log(df['lib_size'].clip(lower=1))

    return df


def run_comparison(
    data: DirichletMultinomialData,
    true_effects: Dict[str, Dict[str, float]],
    methods: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run all methods and compute comparison metrics.

    Parameters
    ----------
    data : DirichletMultinomialData
        Simulated data.
    true_effects : dict
        Ground truth effects.
    methods : list, optional
        Methods to run. If None, runs all.

    Returns
    -------
    pd.DataFrame
        Comparison metrics for all methods.
    """
    if methods is None:
        methods = ['Joint NB-GLM', 'Mann-Whitney', 'Per-Phenotype NB-GLM', 'Dirichlet-Multinomial']

    results = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if 'Joint NB-GLM' in methods:
            try:
                result = run_joint_nb_glm(data)
                metrics = compute_metrics(result, true_effects, data.elm_names)
                results.append(metrics)
            except Exception as e:
                print(f"Joint NB-GLM failed: {e}")

        if 'Mann-Whitney' in methods:
            try:
                result = run_mann_whitney(data)
                metrics = compute_metrics(result, true_effects, data.elm_names)
                results.append(metrics)
            except Exception as e:
                print(f"Mann-Whitney failed: {e}")

        if 'Per-Phenotype NB-GLM' in methods:
            try:
                result = run_per_phenotype_nb_glm(data)
                metrics = compute_metrics(result, true_effects, data.elm_names)
                results.append(metrics)
            except Exception as e:
                print(f"Per-Phenotype NB-GLM failed: {e}")

        if 'Dirichlet-Multinomial' in methods:
            try:
                result = run_dirichlet_multinomial(data)
                metrics = compute_metrics(result, true_effects, data.elm_names)
                results.append(metrics)
            except Exception as e:
                print(f"Dirichlet-Multinomial failed: {e}")

    return pd.DataFrame([vars(m) for m in results])


def run_dirichlet_multinomial(
    data: DirichletMultinomialData,
    alpha: float = 0.05,
    call_mode: str = "pip",
    pip_threshold: float = 0.90,
    ci_level: float = 0.90,
    diagnostics: bool = False,
) -> MethodResult:
    """Run Dirichlet-Multinomial model.

    call_mode:
        - "pip": Waterbear-ish calling using EB PIP + CI exclusion
        - "bh" : classical BH FDR on Wald p-values
    """
    start_time = time.time()

    # Fit DM model with accurate covariance estimation
    result = fit_dm_model(data, verbose=False, use_finite_diff_hessian=True)

    def fitted_mu_per_obs(dm_result: DirichletMultinomialResult) -> np.ndarray:
        d = dm_result.data
        ref = dm_result.reference_phenotype
        ref_idx = d.phenotype_names.index(ref)
        non_ref_idx = [i for i in range(d.n_phenotypes) if i != ref_idx]

        eta = (
            dm_result.beta_0[np.newaxis, :]
            + d.elm_matrix @ dm_result.beta_elm
            + dm_result.beta_ccr[d.ccr_ids, :]
        )
        full = np.zeros((d.n_obs, d.n_phenotypes))
        full[:, non_ref_idx] = eta
        return softmax(full, axis=1)

    # ---- (Optional) diagnostic: alpha per CCR (dispersion heterogeneity)
    mu_hat = fitted_mu_per_obs(result)
    alpha_ccr_hat = estimate_alpha_per_ccr(data, mu_hat, data.ccr_ids)
    if diagnostics and np.isfinite(alpha_ccr_hat).sum() > 1:
        fr = float(np.nanmax(alpha_ccr_hat) / np.nanmin(alpha_ccr_hat))
        print("alpha per CCR:", alpha_ccr_hat)
        print("alpha fold-range:", fr)

    # ---- Extract p-values and effects (ELM × phenotype)
    pval_records = []
    effect_records = []
    se_records = []

    for elm in data.elm_names:
        pvals = result.get_elm_pvalues(elm)
        for pheno in data.phenotype_names:
            effect, se = result.get_elm_effect(elm, pheno)
            pval = pvals.get(pheno, np.nan)

            pval_records.append({"ELM": elm, "phenotype": pheno, "pvalue": pval})
            effect_records.append({"ELM": elm, "phenotype": pheno, "effect": effect})
            se_records.append({"ELM": elm, "phenotype": pheno, "se": se})

    pval_df = (
        pd.DataFrame(pval_records)
        .pivot(index="ELM", columns="phenotype", values="pvalue")
        .reindex(index=data.elm_names, columns=data.phenotype_names)
    )
    effect_df = (
        pd.DataFrame(effect_records)
        .pivot(index="ELM", columns="phenotype", values="effect")
        .reindex(index=data.elm_names, columns=data.phenotype_names)
    )
    se_df = (
        pd.DataFrame(se_records)
        .pivot(index="ELM", columns="phenotype", values="se")
        .reindex(index=data.elm_names, columns=data.phenotype_names)
    )

    # ---- PIP-like calling (EB spike-slab on all coefficients)
    beta_flat = effect_df.values.flatten()
    se_flat = se_df.values.flatten()
    pip_flat, pi0_hat, tau2_hat = eb_spike_slab_pip(beta_flat, se_flat)

    pip_df = pd.DataFrame(
        pip_flat.reshape(effect_df.shape),
        index=effect_df.index,
        columns=effect_df.columns,
    )

    # CI exclusion rule
    z = float(norm.ppf(0.5 + ci_level / 2.0))  # e.g. ci_level=0.90 -> 0.95 quantile -> 1.645
    ci_lo = effect_df - z * se_df
    ci_hi = effect_df + z * se_df
    ci_excludes_zero = (ci_lo > 0) | (ci_hi < 0)

    # ---- Choose calling mode
    if call_mode.lower() == "pip":
        sig_df = (pip_df > pip_threshold) & ci_excludes_zero

        # Do not score the reference phenotype (DM defines it as 0 by construction)
        ref = result.reference_phenotype
        if ref in sig_df.columns:
            sig_df.loc[:, ref] = np.nan

    elif call_mode.lower() == "bh":
        all_pvals = pval_df.values.flatten()
        valid = np.isfinite(all_pvals)
        sig = np.zeros(len(all_pvals), dtype=bool)
        if valid.sum() > 0:
            _, qvals, _, _ = multipletests(all_pvals[valid], method="fdr_bh")
            sig[valid] = qvals < alpha

        sig_df = pd.DataFrame(
            sig.reshape(pval_df.shape),
            index=pval_df.index,
            columns=pval_df.columns,
        )

        # Do not score the reference phenotype (DM defines it as 0 by construction)
        ref = result.reference_phenotype
        if ref in sig_df.columns:
            sig_df.loc[:, ref] = np.nan
    else:
        raise ValueError(f"Unknown call_mode={call_mode!r}. Use 'pip' or 'bh'.")

    runtime = time.time() - start_time

    return MethodResult(
        method_name=f"Dirichlet-Multinomial[{call_mode}]",
        pvalues=pval_df,
        effects=effect_df,
        significant=sig_df,
        runtime_seconds=runtime,
    )


def run_joint_nb_glm(
    data: DirichletMultinomialData,
    alpha: float = 0.05,
) -> MethodResult:
    """Run joint NB-GLM approach (src/costim_screen style).

    Model: count ~ 0 + C(phenotype) + C(CCR) + ELM1:C(phenotype) + ...
    with log(library_size) offset.
    """
    start_time = time.time()

    df = _dm_data_to_long_format(data)

    # Optional (but helpful): force stable factor levels
    df["phenotype"] = pd.Categorical(df["phenotype"], categories=list(data.phenotype_names))
    df["CCR"] = pd.Categorical(df["CCR"])  # ok to leave inferred levels

    elm_terms = " + ".join([f"{elm}:C(phenotype)" for elm in data.elm_names])
    formula = f"count ~ 0 + C(phenotype) + C(CCR, Sum) + {elm_terms}"

    import patsy
    y, X = patsy.dmatrices(formula, df, return_type="dataframe")
    y = y.values.ravel()

    alpha_nb = 1.0
    result = None

    for _ in range(5):
        model = sm.GLM(
            y, X,
            family=sm.families.NegativeBinomial(alpha=alpha_nb),
            offset=df["offset"].values,
        )
        result = model.fit(maxiter=100, method="IRLS")

        mu = result.fittedvalues
        resid = y - mu
        alpha_new = np.sum(resid**2 - mu) / np.sum(mu**2)
        alpha_new = float(np.clip(alpha_new, 0.01, 10.0))
        alpha_nb = 0.5 * alpha_nb + 0.5 * alpha_new

    if result is None:
        raise RuntimeError("Joint NB-GLM failed to fit even once.")

    # ---- Extract ELM effects (EXACT coefficient name match; avoids ELM_1 vs ELM_10 bugs)
    pval_records = []
    effect_records = []

    coef_index = set(result.params.index)

    # Safer to iterate the phenotype names actually used in the data
    for elm in data.elm_names:
        for pheno in data.phenotype_names:
            coef_name = f"{elm}:C(phenotype)[{pheno}]"

            if coef_name in coef_index:
                effect = float(result.params[coef_name])
                pval = float(result.pvalues[coef_name])
            else:
                effect = np.nan
                pval = np.nan

            pval_records.append({"ELM": elm, "phenotype": pheno, "pvalue": pval})
            effect_records.append({"ELM": elm, "phenotype": pheno, "effect": effect})

    pval_df = (
        pd.DataFrame(pval_records)
        .pivot(index="ELM", columns="phenotype", values="pvalue")
        .reindex(index=data.elm_names, columns=data.phenotype_names)
    )
    effect_df = (
        pd.DataFrame(effect_records)
        .pivot(index="ELM", columns="phenotype", values="effect")
        .reindex(index=data.elm_names, columns=data.phenotype_names)
    )

    # FDR correction across all (ELM × phenotype) tests
    all_pvals = pval_df.values.flatten()
    valid = np.isfinite(all_pvals)
    sig = np.zeros(len(all_pvals), dtype=bool)
    if valid.sum() > 0:
        _, qvals, _, _ = multipletests(all_pvals[valid], method="fdr_bh")
        sig[valid] = qvals < alpha

    sig_df = pd.DataFrame(
        sig.reshape(pval_df.shape),
        index=pval_df.index,
        columns=pval_df.columns,
    )

    runtime = time.time() - start_time

    return MethodResult(
        method_name="Joint NB-GLM",
        pvalues=pval_df,
        effects=effect_df,
        significant=sig_df,
        runtime_seconds=runtime,
    )


def run_mann_whitney(
    data: DirichletMultinomialData,
    alpha: float = 0.05,
) -> MethodResult:
    """Run Mann-Whitney approach (Pooled-CAR-T-Analysis style).

    Computes Pearson residuals, then Mann-Whitney U comparing
    CCDs with vs without each ELM, separately per phenotype.
    """
    start_time = time.time()

    # Compute Pearson residuals
    # Simple approach: residual = (count - expected) / sqrt(expected)
    # where expected = (row_sum * col_sum) / total

    counts = data.counts  # (n_obs, n_phenotypes)

    # Compute expected under independence
    row_sums = counts.sum(axis=1, keepdims=True)
    col_sums = counts.sum(axis=0, keepdims=True)
    total = counts.sum()
    expected = (row_sums * col_sums) / total
    expected = np.clip(expected, 1e-6, None)

    # Pearson residuals
    residuals = (counts - expected) / np.sqrt(expected)

    pval_records = []
    effect_records = []

    for e, elm in enumerate(data.elm_names):
        has_elm = data.elm_matrix[:, e] == 1

        for p, pheno in enumerate(PHENOTYPES):
            resid_with = residuals[has_elm, p]
            resid_without = residuals[~has_elm, p]

            if len(resid_with) < 3 or len(resid_without) < 3:
                pval_records.append({'ELM': elm, 'phenotype': pheno, 'pvalue': np.nan})
                effect_records.append({'ELM': elm, 'phenotype': pheno, 'effect': np.nan})
                continue

            # Mann-Whitney U test
            try:
                _, pval = mannwhitneyu(resid_with, resid_without, alternative='two-sided')
            except Exception:
                pval = np.nan

            # Cliff's delta as effect size
            n_gt = np.sum(resid_with[:, None] > resid_without[None, :])
            n_lt = np.sum(resid_with[:, None] < resid_without[None, :])
            n_pairs = len(resid_with) * len(resid_without)
            delta = (n_gt - n_lt) / n_pairs if n_pairs > 0 else 0

            pval_records.append({'ELM': elm, 'phenotype': pheno, 'pvalue': pval})
            effect_records.append({'ELM': elm, 'phenotype': pheno, 'effect': delta})

    pval_df = pd.DataFrame(pval_records).pivot(index='ELM', columns='phenotype', values='pvalue')
    effect_df = pd.DataFrame(effect_records).pivot(index='ELM', columns='phenotype', values='effect')

    # FDR correction per phenotype
    sig_df = pval_df.copy()
    for pheno in PHENOTYPES:
        pvals = pval_df[pheno].values
        valid = np.isfinite(pvals)
        sig = np.zeros(len(pvals), dtype=bool)
        if valid.sum() > 0:
            _, qvals, _, _ = multipletests(pvals[valid], method='fdr_bh')
            sig[valid] = qvals < alpha
        sig_df[pheno] = sig

    runtime = time.time() - start_time

    return MethodResult(
        method_name="Mann-Whitney",
        pvalues=pval_df,
        effects=effect_df,
        significant=sig_df,
        runtime_seconds=runtime,
    )


def run_null_simulation(
    n_replicates: int = 50,
    n_ccds: int = 300,
    n_ccrs: int = 6,
    n_elms: int = 20,
    base_seed: int = 42,
) -> pd.DataFrame:
    """Run simulations under null to check FPR calibration.

    Parameters
    ----------
    n_replicates : int
        Number of simulation replicates.
    n_ccds : int
        Number of CCDs per simulation.
    n_ccrs : int
        Number of CCRs.
    n_elms : int
        Number of ELMs.
    base_seed : int
        Base random seed.

    Returns
    -------
    pd.DataFrame
        FPR results for each method and replicate.
    """
    results = []

    for rep in range(n_replicates):
        # Generate null data (no true effects)
        data, truth = simulate_dm_data(
            n_ccds=n_ccds,
            n_ccrs=n_ccrs,
            n_elms=n_elms,
            elm_effects={},  # No true effects
            seed=base_seed + rep,
        )

        # Run comparison
        metrics_df = run_comparison(data, {}, methods=None)
        metrics_df['replicate'] = rep
        results.append(metrics_df)

    return pd.concat(results, ignore_index=True)


def run_per_phenotype_nb_glm(
    data: DirichletMultinomialData,
    alpha: float = 0.05,
) -> MethodResult:
    """Run per-phenotype NB-GLM approach (Costim-NBGLM-Pipeline style).

    Fits separate NB-GLM for each phenotype:
    count ~ 1 + ELM1 + ELM2 + ...

    Note: This approach does NOT use an offset, matching the original implementation.
    """
    start_time = time.time()

    pval_records = []
    effect_records = []

    # Build design matrix (same for all phenotypes)
    X = np.column_stack([np.ones(data.n_obs), data.elm_matrix])

    for p, pheno in enumerate(PHENOTYPES):
        y = data.counts[:, p]

        # Skip if all zeros
        if y.sum() == 0:
            for elm in data.elm_names:
                pval_records.append({'ELM': elm, 'phenotype': pheno, 'pvalue': np.nan})
                effect_records.append({'ELM': elm, 'phenotype': pheno, 'effect': np.nan})
            continue

        try:
            # Fit NB-GLM without offset (matching original implementation)
            model = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=1.15))
            result = model.fit(maxiter=100, method='IRLS')

            # Extract ELM coefficients (skip intercept at index 0)
            for e, elm in enumerate(data.elm_names):
                coef_idx = e + 1  # +1 for intercept
                effect = result.params[coef_idx]
                pval = result.pvalues[coef_idx]

                pval_records.append({'ELM': elm, 'phenotype': pheno, 'pvalue': pval})
                effect_records.append({'ELM': elm, 'phenotype': pheno, 'effect': effect})

        except Exception:
            for elm in data.elm_names:
                pval_records.append({'ELM': elm, 'phenotype': pheno, 'pvalue': np.nan})
                effect_records.append({'ELM': elm, 'phenotype': pheno, 'effect': np.nan})

    pval_df = pd.DataFrame(pval_records).pivot(index='ELM', columns='phenotype', values='pvalue')
    effect_df = pd.DataFrame(effect_records).pivot(index='ELM', columns='phenotype', values='effect')

    # FDR correction across all tests
    all_pvals = pval_df.values.flatten()
    valid = np.isfinite(all_pvals)
    sig = np.zeros(len(all_pvals), dtype=bool)
    if valid.sum() > 0:
        _, qvals, _, _ = multipletests(all_pvals[valid], method='fdr_bh')
        sig[valid] = qvals < alpha

    sig_df = pd.DataFrame(
        sig.reshape(pval_df.shape),
        index=pval_df.index,
        columns=pval_df.columns,
    )

    runtime = time.time() - start_time

    return MethodResult(
        method_name="Per-Phenotype NB-GLM",
        pvalues=pval_df,
        effects=effect_df,
        significant=sig_df,
        runtime_seconds=runtime,
    )


def run_power_simulation(
    effect_sizes: List[float],
    n_replicates: int = 30,
    n_ccds: int = 300,
    n_ccrs: int = 6,
    n_elms: int = 20,
    n_active_elms: int = 3,
    base_seed: int = 42,
) -> pd.DataFrame:
    """Run simulations to evaluate power across effect sizes.

    Parameters
    ----------
    effect_sizes : list
        Effect sizes to test.
    n_replicates : int
        Number of replicates per effect size.
    n_ccds : int
        Number of CCDs.
    n_ccrs : int
        Number of CCRs.
    n_elms : int
        Number of ELMs.
    n_active_elms : int
        Number of ELMs with true effects.
    base_seed : int
        Base random seed.

    Returns
    -------
    pd.DataFrame
        Power results.
    """
    results = []

    for effect in effect_sizes:
        for rep in range(n_replicates):
            # Generate effects for first n_active_elms
            elm_effects = {}
            for i in range(n_active_elms):
                elm_name = f'ELM_{i}'
                # Effect on CM phenotypes (shift toward CM)
                elm_effects[elm_name] = {
                    'CM_High': effect,
                    'CM_Low': effect * 0.8,
                }

            data, truth = simulate_dm_data(
                n_ccds=n_ccds,
                n_ccrs=n_ccrs,
                n_elms=n_elms,
                elm_effects=elm_effects,
                seed=base_seed + rep + int(effect * 1000),
            )

            metrics_df = run_comparison(data, elm_effects, methods=None)
            metrics_df['effect_size'] = effect
            metrics_df['replicate'] = rep
            results.append(metrics_df)

    return pd.concat(results, ignore_index=True)
