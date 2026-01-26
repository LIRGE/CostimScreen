"""Test DM power with larger sample size."""
import numpy as np
from costim_screen.dirichlet_multinomial import simulate_dm_data, fit_dm_model
from statsmodels.stats.multitest import multipletests

# Same effect, but 5x larger sample
elm_effects = {
    "ELM_0": {"CM_High": 0.6, "CM_Low": 0.4}
}

print("=" * 70)
print("Testing DM with larger sample (n=1000 CCDs)")
print("=" * 70)

data, truth = simulate_dm_data(
    n_ccds=1000,  # 5x larger
    n_ccrs=6,
    n_elms=25,
    elm_prevalence=0.15,
    alpha=25.0,
    elm_effects=elm_effects,
    total_count_mean=2000.0,
    total_count_cv=0.6,
    seed=1,
)

print(f"Data: {data.n_obs} obs")

result = fit_dm_model(data, verbose=True)
print(f"\nConverged: {result.converged}, Alpha: {result.alpha:.2f}")

# Check ELM_0
pvals = result.get_elm_pvalues("ELM_0")
print(f"\n{'Phenotype':<15} {'Effect':>10} {'SE':>10} {'P-value':>12}")
print("-" * 50)
for pheno in data.phenotype_names:
    effect, se = result.get_elm_effect("ELM_0", pheno)
    pval = pvals.get(pheno, np.nan)
    marker = " ← TRUE" if pheno in ["CM_High", "CM_Low"] else ""
    print(f"{pheno:<15} {effect:>10.4f} {se:>10.4f} {pval:>12.2e}{marker}")

# FDR correction
all_pvals = []
all_labels = []
for elm in data.elm_names:
    pvals_elm = result.get_elm_pvalues(elm)
    for pheno in data.phenotype_names:
        if pheno == result.reference_phenotype:
            continue
        pval = pvals_elm.get(pheno, np.nan)
        if not np.isnan(pval):
            all_pvals.append(pval)
            all_labels.append((elm, pheno))

_, qvals, _, _ = multipletests(all_pvals, method='fdr_bh')

n_sig = sum(q < 0.05 for q in qvals)
elm0_detected = [(elm, pheno, q) for (elm, pheno), q in zip(all_labels, qvals)
                  if elm == "ELM_0" and q < 0.05]

print(f"\n{'='*70}")
print(f"Total significant: {n_sig} at FDR < 0.05")
if elm0_detected:
    print(f"✓ Detected ELM_0 on: {[pheno for _, pheno, _ in elm0_detected]}")
else:
    print(f"✗ Did not detect ELM_0")
print("=" * 70)