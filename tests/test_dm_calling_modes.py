"""Test both calling modes for DM model."""
import numpy as np
import pandas as pd

from costim_screen.dirichlet_multinomial import simulate_dm_data, fit_dm_model

print("=" * 70)
print("Testing DM Calling Modes")
print("=" * 70)

# Simple synthetic effect: ELM_0 pushes cells toward CM_High / CM_Low
elm_effects = {
    "ELM_0": {"CM_High": 0.6, "CM_Low": 0.4}
}

print("\nSimulating data...")
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

print(f"Data: {data.n_obs} obs, {data.n_elms} ELMs, {data.n_phenotypes} phenotypes")

# Fit DM model
print("\nFitting DM model...")
result = fit_dm_model(data, verbose=True)

print(f"\nModel converged: {result.converged}")
print(f"Alpha: {result.alpha:.2f}")

# Check ELM_0 effects and p-values
print("\n" + "=" * 70)
print("ELM_0 Effects and P-values (True effect on CM_High=0.6, CM_Low=0.4)")
print("=" * 70)

pvals = result.get_elm_pvalues("ELM_0")

print(f"{'Phenotype':<15} {'Effect':>10} {'SE':>10} {'P-value':>12}")
print("-" * 50)
for pheno in data.phenotype_names:
    effect, se = result.get_elm_effect("ELM_0", pheno)
    pval = pvals.get(pheno, np.nan)
    marker = ""
    if pheno in ["CM_High", "CM_Low"]:
        marker = " ← TRUE EFFECT"
    print(f"{pheno:<15} {effect:>10.4f} {se:>10.4f} {pval:>12.2e}{marker}")

# Apply BH FDR correction
print("\n" + "=" * 70)
print("BH FDR Correction (alpha=0.05)")
print("=" * 70)

all_pvals = []
all_labels = []
for elm in data.elm_names:
    pvals_elm = result.get_elm_pvalues(elm)
    for pheno in data.phenotype_names:
        pval = pvals_elm.get(pheno, np.nan)
        if pd.notna(pval):
            all_pvals.append(pval)
            all_labels.append((elm, pheno))

from statsmodels.stats.multitest import multipletests

_, qvals, _, _ = multipletests(all_pvals, method='fdr_bh')

# Show significant results
sig_results = []
for (elm, pheno), pval, qval in zip(all_labels, all_pvals, qvals):
    if qval < 0.05:
        effect, se = result.get_elm_effect(elm, pheno)
        sig_results.append({
            'ELM': elm,
            'Phenotype': pheno,
            'Effect': effect,
            'SE': se,
            'P-value': pval,
            'Q-value': qval,
        })

if sig_results:
    df = pd.DataFrame(sig_results)
    print(f"\nFound {len(sig_results)} significant associations (FDR < 0.05):")
    print(df.to_string(index=False))

    # Check if we detected ELM_0 on CM phenotypes
    elm0_detected = df[df['ELM'] == 'ELM_0']
    if len(elm0_detected) > 0:
        print("\n✓ Successfully detected ELM_0 effects!")
        print(f"  Detected on: {elm0_detected['Phenotype'].tolist()}")
    else:
        print("\n✗ Failed to detect ELM_0 effects")
else:
    print("\n✗ No significant associations detected at FDR < 0.05")
    print("   This suggests the model or test is too conservative")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)