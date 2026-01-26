"""Smoke test for Dirichlet-Multinomial model comparison."""
import numpy as np

from costim_screen.dirichlet_multinomial import simulate_dm_data
from costim_screen.model_comparison import run_comparison

print("=" * 70)
print("SMOKE TEST: Dirichlet-Multinomial Model")
print("=" * 70)

# Simple synthetic effect: ELM_0 pushes cells toward CM_High / CM_Low
elm_effects = {
    "ELM_0": {"CM_High": 0.6, "CM_Low": 0.4}
}

print("\nSimulating data...")
print(f"  True effects: {elm_effects}")

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

print(f"\nData shape:")
print(f"  Observations: {data.n_obs}")
print(f"  ELMs: {data.n_elms}")
print(f"  Phenotypes: {data.n_phenotypes}")
print(f"  CCRs: {data.n_ccrs}")

# Compare only the DM first so we know it runs
print("\n" + "=" * 70)
print("Running Dirichlet-Multinomial method only...")
print("=" * 70)

metrics = run_comparison(
    data=data,
    true_effects=elm_effects,
    methods=["Dirichlet-Multinomial"],   # keep it narrow initially
)

print("\nResults:")
print(metrics.to_string())

# Check if we have reasonable TPR/FPR
if len(metrics) > 0:
    tpr = metrics.iloc[0]['tpr']
    fpr = metrics.iloc[0]['fpr']
    precision = metrics.iloc[0]['precision']

    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"True Positive Rate (Power): {tpr:.3f}")
    print(f"False Positive Rate:        {fpr:.3f}")
    print(f"Precision:                  {precision:.3f}")
    print(f"Runtime:                    {metrics.iloc[0]['runtime']:.2f}s")

    # Basic sanity checks
    print("\n" + "=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)

    checks_passed = True

    # Check 1: TPR should be reasonably high for this moderate effect size
    if tpr > 0.3:
        print(f"✓ TPR > 0.3 ({tpr:.3f}) - DM has reasonable power")
    else:
        print(f"✗ TPR <= 0.3 ({tpr:.3f}) - WARNING: Low power")
        checks_passed = False

    # Check 2: FPR should be controlled
    if fpr < 0.15:
        print(f"✓ FPR < 0.15 ({fpr:.3f}) - Type I error controlled")
    else:
        print(f"✗ FPR >= 0.15 ({fpr:.3f}) - WARNING: High false positive rate")
        checks_passed = False

    # Check 3: Precision should be decent
    if precision > 0.3:
        print(f"✓ Precision > 0.3 ({precision:.3f}) - Reasonable precision")
    else:
        print(f"✗ Precision <= 0.3 ({precision:.3f}) - WARNING: Low precision")
        checks_passed = False

    if checks_passed:
        print("\n" + "=" * 70)
        print("✓ ALL CHECKS PASSED - Ready to compare all methods!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("⚠ Some checks failed - review results above")
        print("=" * 70)
else:
    print("\n✗ ERROR: No metrics returned!")

print("\n" + "=" * 70)
print("SMOKE TEST COMPLETE")
print("=" * 70)