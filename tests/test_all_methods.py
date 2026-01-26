"""Compare all methods on synthetic DM data."""
import numpy as np
from costim_screen.dirichlet_multinomial import simulate_dm_data
from costim_screen.model_comparison import run_comparison

print("=" * 70)
print("FULL METHOD COMPARISON")
print("=" * 70)

# Same effect as smoke test
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

# Compare all methods
print("\n" + "=" * 70)
print("Running all methods...")
print("=" * 70)

metrics = run_comparison(
    data=data,
    true_effects=elm_effects,
    methods=None,  # None = all methods
)

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(metrics.to_string(index=False))

# Highlight best method for each metric
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

best_tpr_idx = metrics['tpr'].idxmax()
best_fpr_idx = metrics['fpr'].idxmin()
best_precision_idx = metrics['precision'].idxmax()
fastest_idx = metrics['runtime'].idxmin()

print(f"Best TPR (Power):      {metrics.loc[best_tpr_idx, 'method']:<30} ({metrics.loc[best_tpr_idx, 'tpr']:.3f})")
print(f"Best FPR (Type I):     {metrics.loc[best_fpr_idx, 'method']:<30} ({metrics.loc[best_fpr_idx, 'fpr']:.3f})")
print(f"Best Precision:        {metrics.loc[best_precision_idx, 'method']:<30} ({metrics.loc[best_precision_idx, 'precision']:.3f})")
print(f"Fastest Runtime:       {metrics.loc[fastest_idx, 'method']:<30} ({metrics.loc[fastest_idx, 'runtime']:.2f}s)")

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)