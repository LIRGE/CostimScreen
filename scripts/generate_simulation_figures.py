#!/usr/bin/env python3
"""Generate simulation/model comparison figures and tables from results."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Paths
BASE_PATH = Path('.').resolve()
RESULTS_PATH = BASE_PATH / 'results' / 'model_comparison'
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# Load data
summary_df = pd.read_csv(BASE_PATH / 'data' / 'simulation_results_summary.csv')
detailed_df = pd.read_csv(BASE_PATH / 'data' / 'simulation_results_detailed.csv')

# Method colors and labels
METHOD_COLORS = {
    'Dirichlet-Multinomial[pip]': '#2ecc71',  # Green
    'Joint NB-GLM': '#3498db',  # Blue
    'Mann-Whitney': '#e74c3c',  # Red
    'Per-Phenotype NB-GLM': '#9b59b6',  # Purple
}

METHOD_LABELS = {
    'Dirichlet-Multinomial[pip]': 'DM (PIP)',
    'Joint NB-GLM': 'Joint GLM',
    'Mann-Whitney': 'Mann-Whitney',
    'Per-Phenotype NB-GLM': 'Per-Pheno GLM',
}

METHODS = list(METHOD_COLORS.keys())

# ============================================================
# FIGURE 1: Effect Size RMSE
# ============================================================
print("Generating Figure 1: Effect Size RMSE...")

fig, ax = plt.subplots(figsize=(10, 6))

# Group by method
rmse_by_method = summary_df.groupby('method')['effect_rmse_mean'].mean()

x = np.arange(len(METHODS))
width = 0.6

bars = []
for i, method in enumerate(METHODS):
    if method in rmse_by_method.index:
        rmse = rmse_by_method[method]
        bar = ax.bar(i, rmse, width, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        bars.append(bar)
        ax.text(i, rmse + 0.005, f'{rmse:.3f}', ha='center', va='bottom', fontsize=10)

ax.set_xlabel('Method', fontsize=12)
ax.set_ylabel('Effect Size RMSE', fontsize=12)
ax.set_title('Effect Size Estimation Accuracy (Lower is Better)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=15, ha='right')
ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_PATH / 'fig1_effect_rmse.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {RESULTS_PATH / 'fig1_effect_rmse.png'}")

# ============================================================
# FIGURE 2: Power Curves (3 panels for N=100, 200, 400)
# ============================================================
print("Generating Figure 2: Power Curves...")

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

n_ccds_list = [100, 200, 400]
effect_sizes = [0.3, 0.6, 0.9]

for ax_idx, n_ccds in enumerate(n_ccds_list):
    ax = axes[ax_idx]

    for method in METHODS:
        powers = []
        for effect in effect_sizes:
            mask = (summary_df['method'] == method) & (summary_df['n_ccds'] == n_ccds) & (summary_df['effect_size'] == effect)
            if mask.any():
                powers.append(summary_df.loc[mask, 'tpr_mean'].values[0])
            else:
                powers.append(np.nan)

        ax.plot(effect_sizes, powers, 'o-', color=METHOD_COLORS[method],
                label=METHOD_LABELS[method], linewidth=2, markersize=8)

    ax.set_xlabel('Effect Size', fontsize=12)
    ax.set_title(f'N = {n_ccds} CCDs', fontsize=13)
    ax.set_xticks(effect_sizes)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    if ax_idx == 0:
        ax.set_ylabel('Power (True Positive Rate)', fontsize=12)

# Single legend for all panels
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02),
           ncol=4, fontsize=11, frameon=True)

fig.suptitle('Statistical Power Across Sample Sizes and Effect Sizes', fontsize=14, y=1.02)
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.savefig(RESULTS_PATH / 'fig2_power_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {RESULTS_PATH / 'fig2_power_curves.png'}")

# ============================================================
# FIGURE 3: Precision vs Power Tradeoff
# ============================================================
print("Generating Figure 3: Precision vs Power Tradeoff...")

fig, ax = plt.subplots(figsize=(10, 8))

# Plot each method with different markers for effect sizes
effect_markers = {0.3: 's', 0.6: 'o', 0.9: '^'}  # square, circle, triangle

for method in METHODS:
    method_data = summary_df[summary_df['method'] == method]

    for effect in effect_sizes:
        effect_data = method_data[method_data['effect_size'] == effect]
        if len(effect_data) > 0:
            # Average across sample sizes
            mean_tpr = effect_data['tpr_mean'].mean()
            mean_precision = effect_data['precision_mean'].mean()

            ax.scatter(mean_tpr, mean_precision,
                      color=METHOD_COLORS[method],
                      marker=effect_markers[effect],
                      s=150, alpha=0.8, edgecolors='black', linewidth=1)

# Create legend entries
method_patches = [mpatches.Patch(color=METHOD_COLORS[m], label=METHOD_LABELS[m]) for m in METHODS]
effect_handles = [plt.Line2D([0], [0], marker=effect_markers[e], color='gray',
                             linestyle='', markersize=10, label=f'Effect = {e}')
                  for e in effect_sizes]

ax.legend(handles=method_patches + effect_handles, loc='lower right', fontsize=10)

ax.set_xlabel('Power (True Positive Rate)', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision vs Power Tradeoff', fontsize=14)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.grid(alpha=0.3)

# Add diagonal reference
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='_nolegend_')

plt.tight_layout()
plt.savefig(RESULTS_PATH / 'fig3_precision_vs_power.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {RESULTS_PATH / 'fig3_precision_vs_power.png'}")

# ============================================================
# FIGURE 4: Computational Cost
# ============================================================
print("Generating Figure 4: Computational Cost...")

fig, ax = plt.subplots(figsize=(10, 6))

# Average runtime by method
runtime_by_method = summary_df.groupby('method')['runtime_mean'].mean()

x = np.arange(len(METHODS))
width = 0.6

for i, method in enumerate(METHODS):
    if method in runtime_by_method.index:
        runtime = runtime_by_method[method]
        ax.bar(i, runtime, width, color=METHOD_COLORS[method])
        ax.text(i, runtime + 0.1, f'{runtime:.2f}s', ha='center', va='bottom', fontsize=10)

ax.set_xlabel('Method', fontsize=12)
ax.set_ylabel('Average Runtime (seconds)', fontsize=12)
ax.set_title('Computational Cost Comparison', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=15, ha='right')
ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_PATH / 'fig4_computational_cost.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {RESULTS_PATH / 'fig4_computational_cost.png'}")

# ============================================================
# FIGURE 5: Type I Error Control (FPR)
# ============================================================
print("Generating Figure 5: Type I Error Control...")

fig, ax = plt.subplots(figsize=(10, 6))

# Average FPR by method
fpr_by_method = summary_df.groupby('method')['fpr_mean'].mean()

x = np.arange(len(METHODS))
width = 0.6

for i, method in enumerate(METHODS):
    if method in fpr_by_method.index:
        fpr = fpr_by_method[method]
        ax.bar(i, fpr, width, color=METHOD_COLORS[method])
        ax.text(i, fpr + 0.002, f'{fpr:.3f}', ha='center', va='bottom', fontsize=10)

# Add nominal alpha line
ax.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='Nominal α = 0.05')

ax.set_xlabel('Method', fontsize=12)
ax.set_ylabel('False Positive Rate', fontsize=12)
ax.set_title('Type I Error Control (Lower is Better, Should be ≤ 0.05)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=15, ha='right')
ax.set_ylim(0, max(0.08, ax.get_ylim()[1] * 1.15))
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_PATH / 'fig5_type1_error.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {RESULTS_PATH / 'fig5_type1_error.png'}")

# ============================================================
# TABLE 1: Summary Statistics
# ============================================================
print("Generating Table 1: Summary Statistics...")

summary_table = summary_df.groupby('method').agg({
    'tpr_mean': 'mean',
    'fpr_mean': 'mean',
    'precision_mean': 'mean',
    'effect_rmse_mean': 'mean',
    'runtime_mean': 'mean',
}).round(3)

summary_table.columns = ['Power (TPR)', 'FPR', 'Precision', 'Effect RMSE', 'Runtime (s)']
summary_table.index = summary_table.index.map(lambda x: METHOD_LABELS.get(x, x))
summary_table = summary_table.reindex([METHOD_LABELS[m] for m in METHODS if m in METHOD_LABELS])

summary_table.to_csv(RESULTS_PATH / 'table1_summary_statistics.csv')
print(f"  Saved: {RESULTS_PATH / 'table1_summary_statistics.csv'}")
print("\nTable 1: Summary Statistics")
print(summary_table.to_string())

# ============================================================
# TABLE 2: Detailed Results by Scenario
# ============================================================
print("\nGenerating Table 2: Detailed Results by Scenario...")

# Pivot table: method × scenario
scenario_order = [
    'small_n_small_effect', 'small_n_medium_effect', 'small_n_large_effect',
    'medium_n_small_effect', 'medium_n_medium_effect', 'medium_n_large_effect',
    'large_n_small_effect', 'large_n_medium_effect', 'large_n_large_effect'
]

detailed_table = summary_df.pivot_table(
    index='method',
    columns='scenario',
    values=['tpr_mean', 'fpr_mean', 'precision_mean', 'effect_rmse_mean'],
    aggfunc='first'
).round(3)

# Reorder columns
cols_order = [('tpr_mean', s) for s in scenario_order if s in detailed_table.columns.get_level_values(1)]
detailed_table = detailed_table.reindex(columns=cols_order)

detailed_table.to_csv(RESULTS_PATH / 'table2_detailed_by_scenario.csv')
print(f"  Saved: {RESULTS_PATH / 'table2_detailed_by_scenario.csv'}")

# Also create a cleaner version for power only
power_table = summary_df.pivot_table(
    index='method',
    columns=['n_ccds', 'effect_size'],
    values='tpr_mean',
    aggfunc='first'
).round(3)

power_table.index = power_table.index.map(lambda x: METHOD_LABELS.get(x, x))
power_table.to_csv(RESULTS_PATH / 'table2_power_by_scenario.csv')

print("\nTable 2: Power by Scenario (N × Effect Size)")
print(power_table.to_string())

print(f"\n{'='*60}")
print("All figures and tables generated successfully!")
print(f"Output directory: {RESULTS_PATH}")
print(f"{'='*60}")
