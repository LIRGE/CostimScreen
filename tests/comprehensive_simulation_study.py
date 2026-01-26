"""Comprehensive simulation study comparing all methods.

This script evaluates method performance across:
1. Different effect sizes (small, medium, large)
2. Different sample sizes (small, medium, large)
3. Multiple simulation replicates for robust estimates

Results are saved to results/dm_analysis/ for plotting.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import time

from costim_screen.dirichlet_multinomial import simulate_dm_data
from costim_screen.model_comparison import run_comparison

# Create output directory
# Get the repository root (parent of tests/ if running from tests/)
if Path('.').resolve().name == 'tests':
    base_path = Path('.').resolve().parent
else:
    base_path = Path('.').resolve()

output_dir = base_path / "results" / "dm_analysis"
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("COMPREHENSIVE SIMULATION STUDY")
print("=" * 80)

# Simulation scenarios
scenarios = [
    # (name, n_ccds, effect_size, description)
    ("small_n_small_effect", 100, 0.3, "N=100, effect=0.3"),
    ("small_n_medium_effect", 100, 0.6, "N=100, effect=0.6"),
    ("small_n_large_effect", 100, 0.9, "N=100, effect=0.9"),
    ("medium_n_small_effect", 200, 0.3, "N=200, effect=0.3"),
    ("medium_n_medium_effect", 200, 0.6, "N=200, effect=0.6"),
    ("medium_n_large_effect", 200, 0.9, "N=200, effect=0.9"),
    ("large_n_small_effect", 400, 0.3, "N=400, effect=0.3"),
    ("large_n_medium_effect", 400, 0.6, "N=400, effect=0.6"),
    ("large_n_large_effect", 400, 0.9, "N=400, effect=0.9"),
]

n_replicates = 5  # Number of simulation replicates per scenario
all_results = []

for scenario_idx, (scenario_name, n_ccds, effect_size, description) in enumerate(scenarios):
    print(f"\n{'=' * 80}")
    print(f"Scenario {scenario_idx + 1}/{len(scenarios)}: {description}")
    print(f"{'=' * 80}")

    for rep in range(n_replicates):
        print(f"\n  Replicate {rep + 1}/{n_replicates}...")

        # Define effect: ELM_0 pushes cells toward CM_High/CM_Low
        elm_effects = {
            "ELM_0": {"CM_High": effect_size, "CM_Low": effect_size * 0.67}
        }

        # Simulate data
        seed = scenario_idx * 100 + rep  # Unique seed per scenario-replicate
        data, truth = simulate_dm_data(
            n_ccds=n_ccds,
            n_ccrs=6,
            n_elms=25,
            elm_prevalence=0.15,
            alpha=25.0,
            elm_effects=elm_effects,
            total_count_mean=2000.0,
            total_count_cv=0.6,
            seed=seed,
        )

        # Run comparison
        start_time = time.time()
        metrics = run_comparison(
            data=data,
            true_effects=elm_effects,
            methods=None,  # All methods
        )
        total_time = time.time() - start_time

        # Add metadata
        metrics['scenario'] = scenario_name
        metrics['description'] = description
        metrics['n_ccds'] = n_ccds
        metrics['effect_size'] = effect_size
        metrics['replicate'] = rep
        metrics['total_runtime'] = total_time

        all_results.append(metrics)

        print(f"    Completed in {total_time:.1f}s")
        print(f"    DM TPR: {metrics[metrics['method'].str.contains('Dirichlet')]['tpr'].values[0]:.3f}")
        print(f"    DM FPR: {metrics[metrics['method'].str.contains('Dirichlet')]['fpr'].values[0]:.3f}")

# Combine all results
print(f"\n{'=' * 80}")
print("Combining results...")
print(f"{'=' * 80}")

combined_df = pd.concat(all_results, ignore_index=True)

# Save detailed results
output_file = output_dir / "simulation_results_detailed.csv"
combined_df.to_csv(output_file, index=False)
print(f"\n✓ Saved detailed results to: {output_file}")

# Compute summary statistics (mean ± SE across replicates)
summary = combined_df.groupby(['method', 'scenario', 'n_ccds', 'effect_size']).agg({
    'tpr': ['mean', 'std'],
    'fpr': ['mean', 'std'],
    'precision': ['mean', 'std'],
    'effect_bias': ['mean', 'std'],
    'effect_rmse': ['mean', 'std'],
    'runtime': ['mean', 'std'],
}).reset_index()

# Flatten column names
summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]

# Save summary
summary_file = output_dir / "simulation_results_summary.csv"
summary.to_csv(summary_file, index=False)
print(f"✓ Saved summary statistics to: {summary_file}")

# Print key findings
print(f"\n{'=' * 80}")
print("KEY FINDINGS")
print(f"{'=' * 80}")

for method_name in combined_df['method'].unique():
    method_data = combined_df[combined_df['method'] == method_name]

    print(f"\n{method_name}:")
    print(f"  Mean TPR (power):     {method_data['tpr'].mean():.3f} ± {method_data['tpr'].std():.3f}")
    print(f"  Mean FPR (type I):    {method_data['fpr'].mean():.3f} ± {method_data['fpr'].std():.3f}")
    print(f"  Mean precision:       {method_data['precision'].mean():.3f} ± {method_data['precision'].std():.3f}")
    print(f"  Mean effect bias:     {method_data['effect_bias'].mean():.3f}")
    print(f"  Mean runtime:         {method_data['runtime'].mean():.2f}s")

print(f"\n{'=' * 80}")
print("SIMULATION STUDY COMPLETE")
print(f"{'=' * 80}")
print(f"\nResults saved to: {output_dir}")
print(f"  - {output_file.name}")
print(f"  - {summary_file.name}")
print(f"\nNext: Run plot_simulation_results.py to visualize these results")