"""Compare L-BFGS-B vs finite-diff Hessian for standard errors."""
import numpy as np
from costim_screen.dirichlet_multinomial import simulate_dm_data, fit_dm_model

elm_effects = {
    "ELM_0": {"CM_High": 0.6, "CM_Low": 0.4}
}

print("=" * 70)
print("Comparing Hessian Computation Methods")
print("=" * 70)

# Use moderate sample size
data, truth = simulate_dm_data(
    n_ccds=200,
    n_ccrs=6,
    n_elms=25,
    elm_prevalence=0.15,
    alpha=25.0,
    elm_effects=elm_effects,
    total_count_mean=2000.0,
    total_count_cv=0.6,
    seed=1,
)

print(f"\nData: {data.n_obs} obs, {data.n_elms} ELMs, {data.n_phenotypes} phenotypes\n")

# Method 1: L-BFGS-B approximation (default)
print("Method 1: L-BFGS-B Inverse Hessian (fast)")
print("-" * 70)
result_lbfgs = fit_dm_model(data, verbose=False, use_finite_diff_hessian=False)
print(f"Converged: {result_lbfgs.converged}, Alpha: {result_lbfgs.alpha:.2f}")

pvals_lbfgs = result_lbfgs.get_elm_pvalues("ELM_0")
print(f"\n{'Phenotype':<15} {'Effect':>10} {'SE':>10} {'P-value':>12}")
print("-" * 50)
for pheno in data.phenotype_names:
    effect, se = result_lbfgs.get_elm_effect("ELM_0", pheno)
    pval = pvals_lbfgs.get(pheno, np.nan)
    marker = " ← TRUE" if pheno in ["CM_High", "CM_Low"] else ""
    print(f"{pheno:<15} {effect:>10.4f} {se:>10.4f} {pval:>12.2e}{marker}")

# Method 2: Finite differences (more accurate)
print("\n" + "=" * 70)
print("Method 2: Finite Difference Hessian (accurate)")
print("-" * 70)
result_findiff = fit_dm_model(data, verbose=True, use_finite_diff_hessian=True)
print(f"\nConverged: {result_findiff.converged}, Alpha: {result_findiff.alpha:.2f}")

pvals_findiff = result_findiff.get_elm_pvalues("ELM_0")
print(f"\n{'Phenotype':<15} {'Effect':>10} {'SE':>10} {'P-value':>12}")
print("-" * 50)
for pheno in data.phenotype_names:
    effect, se = result_findiff.get_elm_effect("ELM_0", pheno)
    pval = pvals_findiff.get(pheno, np.nan)
    marker = " ← TRUE" if pheno in ["CM_High", "CM_Low"] else ""
    print(f"{pheno:<15} {effect:>10.4f} {se:>10.4f} {pval:>12.2e}{marker}")

# Compare SEs
print("\n" + "=" * 70)
print("SE Comparison (Method 2 should be smaller = more power)")
print("=" * 70)
print(f"{'Phenotype':<15} {'L-BFGS SE':>12} {'FinDiff SE':>12} {'Ratio':>10}")
print("-" * 52)
for pheno in data.phenotype_names:
    if pheno == result_lbfgs.reference_phenotype:
        continue
    _, se_lbfgs = result_lbfgs.get_elm_effect("ELM_0", pheno)
    _, se_findiff = result_findiff.get_elm_effect("ELM_0", pheno)
    ratio = se_lbfgs / se_findiff if se_findiff > 0 else np.nan
    marker = " ← TRUE" if pheno in ["CM_High", "CM_Low"] else ""
    print(f"{pheno:<15} {se_lbfgs:>12.4f} {se_findiff:>12.4f} {ratio:>10.2f}{marker}")

print("\n" + "=" * 70)