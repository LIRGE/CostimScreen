"""Generate comparison plots from simulation study results."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_context("paper")
sns.set_style("whitegrid")

# Load results
# Get the repository root (parent of tests/ if running from tests/)
if Path('.').resolve().name == 'tests':
    base_path = Path('.').resolve().parent
else:
    base_path = Path('.').resolve()

results_dir = base_path / "results" / "dm_analysis"
detailed_file = results_dir / "simulation_results_detailed.csv"

if not detailed_file.exists():
    raise FileNotFoundError(
        f"Results file not found: {detailed_file}\n"
        "Please run comprehensive_simulation_study.py first."
    )

print("=" * 80)
print("PLOTTING SIMULATION RESULTS")
print("=" * 80)

df = pd.read_csv(detailed_file)
print(f"\nLoaded {len(df)} rows from {detailed_file.name}")

# Clean method names for plotting
method_rename = {
    "Joint NB-GLM": "Joint GLM",
    "Mann-Whitney": "Mann-Whitney",
    "Per-Phenotype NB-GLM": "Per-Pheno GLM",
    "Dirichlet-Multinomial[pip]": "DM (PIP)",
}
df['method_short'] = df['method'].map(method_rename)

# Define color palette
colors = {
    "DM (PIP)": "#E63946",      # Red - our method
    "Joint GLM": "#457B9D",      # Blue
    "Mann-Whitney": "#2A9D8F",   # Teal
    "Per-Pheno GLM": "#F4A261",  # Orange
}

# ============================================================================
# Figure 1: Power curves (TPR vs effect size) by sample size
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for i, n_ccds in enumerate([100, 200, 400]):
    ax = axes[i]
    subset = df[df['n_ccds'] == n_ccds]

    # Compute mean and SE across replicates
    grouped = subset.groupby(['method_short', 'effect_size']).agg({
        'tpr': ['mean', 'sem']
    }).reset_index()
    grouped.columns = ['method_short', 'effect_size', 'tpr_mean', 'tpr_sem']

    for method in colors.keys():
        method_data = grouped[grouped['method_short'] == method]
        ax.errorbar(
            method_data['effect_size'],
            method_data['tpr_mean'],
            yerr=method_data['tpr_sem'],
            label=method,
            color=colors[method],
            marker='o',
            linewidth=2,
            capsize=4,
        )

    ax.axhline(0.05, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='α=0.05')
    ax.set_xlabel("Effect Size", fontsize=11)
    ax.set_title(f"N = {n_ccds} CCDs", fontsize=12, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    if i == 0:
        ax.set_ylabel("True Positive Rate (Power)", fontsize=11)
        ax.legend(frameon=True, fontsize=9)

plt.tight_layout()
power_curve_file = results_dir / "figure_power_curves.png"
plt.savefig(power_curve_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {power_curve_file.name}")
plt.close()

# ============================================================================
# Figure 2: Type I error control (FPR)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 5))

# Average FPR across all scenarios
fpr_summary = df.groupby('method_short')['fpr'].agg(['mean', 'sem']).reset_index()
fpr_summary = fpr_summary.sort_values('mean')

x_pos = np.arange(len(fpr_summary))
bars = ax.bar(
    x_pos,
    fpr_summary['mean'],
    yerr=fpr_summary['sem'],
    capsize=5,
    color=[colors[m] for m in fpr_summary['method_short']],
    edgecolor='black',
    linewidth=1.5,
)

ax.axhline(0.05, color='red', linestyle='--', linewidth=2, label='Nominal α=0.05')
ax.set_xticks(x_pos)
ax.set_xticklabels(fpr_summary['method_short'], fontsize=11)
ax.set_ylabel("False Positive Rate", fontsize=12)
ax.set_title("Type I Error Control (Mean ± SE across all scenarios)", fontsize=13, fontweight='bold')
ax.legend(frameon=True, fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(fpr_summary['mean'].max() * 1.3, 0.08))

plt.tight_layout()
fpr_file = results_dir / "figure_type_I_error.png"
plt.savefig(fpr_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {fpr_file.name}")
plt.close()

# ============================================================================
# Figure 3: Effect size estimation (RMSE)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 5))

rmse_summary = df.groupby('method_short')['effect_rmse'].agg(['mean', 'sem']).reset_index()
rmse_summary = rmse_summary.sort_values('mean')

x_pos = np.arange(len(rmse_summary))
bars = ax.bar(
    x_pos,
    rmse_summary['mean'],
    yerr=rmse_summary['sem'],
    capsize=5,
    color=[colors[m] for m in rmse_summary['method_short']],
    edgecolor='black',
    linewidth=1.5,
)

ax.set_xticks(x_pos)
ax.set_xticklabels(rmse_summary['method_short'], fontsize=11)
ax.set_ylabel("Effect Size RMSE", fontsize=12)
ax.set_title("Effect Size Estimation Accuracy (Mean ± SE across all scenarios)", fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
rmse_file = results_dir / "figure_effect_rmse.png"
plt.savefig(rmse_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {rmse_file.name}")
plt.close()

# ============================================================================
# Figure 4: Runtime comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 5))

runtime_summary = df.groupby('method_short')['runtime'].agg(['mean', 'sem']).reset_index()
runtime_summary = runtime_summary.sort_values('mean')

x_pos = np.arange(len(runtime_summary))
bars = ax.bar(
    x_pos,
    runtime_summary['mean'],
    yerr=runtime_summary['sem'],
    capsize=5,
    color=[colors[m] for m in runtime_summary['method_short']],
    edgecolor='black',
    linewidth=1.5,
)

ax.set_xticks(x_pos)
ax.set_xticklabels(runtime_summary['method_short'], fontsize=11)
ax.set_ylabel("Runtime (seconds)", fontsize=12)
ax.set_title("Computational Cost (Mean ± SE across all scenarios)", fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
runtime_file = results_dir / "figure_runtime.png"
plt.savefig(runtime_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {runtime_file.name}")
plt.close()

# ============================================================================
# Figure 5: Precision vs Power tradeoff
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 8))

# Average across all scenarios
perf_summary = df.groupby('method_short').agg({
    'tpr': 'mean',
    'precision': 'mean'
}).reset_index()

for method in colors.keys():
    method_data = perf_summary[perf_summary['method_short'] == method]
    ax.scatter(
        method_data['tpr'],
        method_data['precision'],
        s=300,
        color=colors[method],
        edgecolor='black',
        linewidth=2,
        label=method,
        alpha=0.8,
    )

ax.set_xlabel("True Positive Rate (Power)", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision vs Power Tradeoff", fontsize=13, fontweight='bold')
ax.legend(frameon=True, fontsize=11)
ax.grid(alpha=0.3)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)

# Add diagonal reference
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)

plt.tight_layout()
tradeoff_file = results_dir / "figure_precision_power_tradeoff.png"
plt.savefig(tradeoff_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {tradeoff_file.name}")
plt.close()

# ============================================================================
# Summary Table
# ============================================================================
print(f"\n{'=' * 80}")
print("SUMMARY TABLE")
print(f"{'=' * 80}")

summary_table = df.groupby('method_short').agg({
    'tpr': ['mean', 'std'],
    'fpr': ['mean', 'std'],
    'precision': ['mean', 'std'],
    'effect_rmse': ['mean', 'std'],
    'runtime': ['mean', 'std'],
}).round(3)

print(summary_table.to_string())

# Save table
table_file = results_dir / "summary_table.txt"
with open(table_file, 'w') as f:
    f.write("SIMULATION STUDY SUMMARY\n")
    f.write("=" * 80 + "\n\n")
    f.write(summary_table.to_string())
print(f"\n✓ Saved: {table_file.name}")

print(f"\n{'=' * 80}")
print("PLOTTING COMPLETE")
print(f"{'=' * 80}")
print(f"\nAll figures saved to: {results_dir}")