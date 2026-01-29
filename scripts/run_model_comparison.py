#!/usr/bin/env python3
"""
Run model comparison simulations.

This script compares four modeling approaches on synthetic data generated
from the Dirichlet-Multinomial ground truth:

1. Joint NB-GLM (src/costim_screen approach)
2. Mann-Whitney on residuals (Pooled-CAR-T-Analysis approach)
3. Per-phenotype NB-GLM (Costim-NBGLM-Pipeline approach)
4. Dirichlet-Multinomial (new principled model)

Evaluates:
- Type I error (FPR under null)
- Power (TPR with true effects)
- Effect size estimation accuracy
"""
import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from costim_screen.model_comparison import (
    run_null_simulation,
    run_power_simulation,
    run_comparison,
)
from costim_screen.dirichlet_multinomial import simulate_dm_data, PHENOTYPES

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("results/model_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_fpr_analysis():
    """Test 1: Type I Error (FPR) under null."""
    print("\n" + "="*70)
    print("TEST 1: Type I Error (FPR) Under Null")
    print("="*70)
    print()
    print("Generating data with NO true ELM effects.")
    print("A well-calibrated method should have FPR ≈ 0.05")
    print()

    # Run null simulations
    n_reps = 30
    print(f"Running {n_reps} null simulations...")

    results = run_null_simulation(
        n_replicates=n_reps,
        n_ccds=250,
        n_ccrs=6,
        n_elms=15,
    )

    # Summarize FPR by method
    print("\n--- Results ---")
    summary = results.groupby('method').agg({
        'fpr': ['mean', 'std'],
        'n_false_positives': 'mean',
        'runtime': 'mean',
    }).round(4)
    print(summary)
    print()

    # Interpretation
    print("--- Interpretation ---")
    for method in results['method'].unique():
        method_fpr = results[results['method'] == method]['fpr'].mean()
        if method_fpr > 0.10:
            status = "INFLATED (>10%)"
        elif method_fpr > 0.07:
            status = "SLIGHTLY INFLATED"
        elif method_fpr < 0.02:
            status = "CONSERVATIVE"
        else:
            status = "WELL-CALIBRATED"
        print(f"  {method}: FPR = {method_fpr:.3f} → {status}")

    return results


def run_power_analysis():
    """Test 2: Power across effect sizes."""
    print("\n" + "="*70)
    print("TEST 2: Power Across Effect Sizes")
    print("="*70)
    print()
    print("Generating data with true ELM effects on CM phenotypes.")
    print("Effect sizes tested: 0.3, 0.5, 0.7, 1.0 (log-odds scale)")
    print()

    effect_sizes = [0.3, 0.5, 0.7, 1.0]
    n_reps = 20

    print(f"Running {n_reps} simulations per effect size...")

    results = run_power_simulation(
        effect_sizes=effect_sizes,
        n_replicates=n_reps,
        n_ccds=250,
        n_ccrs=6,
        n_elms=15,
        n_active_elms=3,
    )

    # Summarize power by method and effect size
    print("\n--- Power (TPR) by Effect Size ---")
    power_summary = results.groupby(['effect_size', 'method'])['tpr'].mean().unstack()
    print(power_summary.round(3))
    print()

    # Summarize FPR (should remain controlled)
    print("--- FPR by Effect Size (should remain ~0.05) ---")
    fpr_summary = results.groupby(['effect_size', 'method'])['fpr'].mean().unstack()
    print(fpr_summary.round(3))

    return results


def run_detailed_example():
    """Test 3: Detailed single example."""
    print("\n" + "="*70)
    print("TEST 3: Detailed Single Example")
    print("="*70)
    print()

    # Create scenario with different effect types
    elm_effects = {
        'ELM_0': {'CM_High': 0.8, 'CM_Low': 0.6},  # Shift toward CM
        'ELM_1': {'EM_High': 0.7, 'EM_Low': 0.7},  # Shift toward EM
        'ELM_2': {  # Shift toward PD1_High (across T-subsets)
            'Naïve_High': 0.5, 'CM_High': 0.5, 'EM_High': 0.5,
        },
    }

    print("Ground Truth Effects (log-odds scale):")
    print("  ELM_0: +0.8 to CM_High, +0.6 to CM_Low")
    print("  ELM_1: +0.7 to EM_High, +0.7 to EM_Low")
    print("  ELM_2: +0.5 to all PD1_High phenotypes")
    print()

    data, truth = simulate_dm_data(
        n_ccds=400,
        n_ccrs=6,
        n_elms=20,
        elm_effects=elm_effects,
        alpha=25.0,
        seed=12345,
    )

    print(f"Simulated data: {data.n_obs} observations, {data.n_elms} ELMs")
    print()

    # Run all methods
    metrics_df = run_comparison(data, elm_effects)

    print("--- Summary Metrics ---")
    print(metrics_df[['method', 'tpr', 'fpr', 'precision', 'effect_rmse', 'runtime']].to_string(index=False))
    print()

    # Show detailed results for active ELMs
    from costim_screen.model_comparison import (
        run_joint_nb_glm,
        run_mann_whitney,
        run_per_phenotype_nb_glm,
        run_dirichlet_multinomial,
    )

    print("--- P-values for Active ELMs (ELM_0, ELM_1, ELM_2) ---")
    print()

    for method_name, run_func in [
        ('Joint NB-GLM', run_joint_nb_glm),
        ('Mann-Whitney', run_mann_whitney),
        ('Per-Phenotype NB-GLM', run_per_phenotype_nb_glm),
        ('Dirichlet-Multinomial', run_dirichlet_multinomial),
    ]:
        try:
            result = run_func(data)
            print(f"{method_name}:")
            for elm in ['ELM_0', 'ELM_1', 'ELM_2']:
                if elm in result.pvalues.index:
                    pvals = result.pvalues.loc[elm]
                    sig_count = result.significant.loc[elm].sum()
                    print(f"  {elm}: {sig_count}/6 phenotypes significant")
                    # Show p-values for phenotypes with true effects
                    for pheno in PHENOTYPES:
                        if elm in elm_effects and pheno in elm_effects[elm]:
                            p = pvals.get(pheno, np.nan)
                            sig = "***" if result.significant.loc[elm, pheno] else ""
                            print(f"    {pheno}: p={p:.2e} {sig} (true effect: {elm_effects[elm][pheno]})")
            print()
        except Exception as e:
            print(f"{method_name}: FAILED - {e}")
            print()

    return data, elm_effects, metrics_df


def main():
    """Run all comparison analyses."""
    print("="*70)
    print("MODEL COMPARISON: Dirichlet-Multinomial Ground Truth")
    print("="*70)
    print()
    print("Comparing four modeling approaches:")
    print("  1. Joint NB-GLM (src/costim_screen)")
    print("  2. Mann-Whitney on residuals (Pooled-CAR-T-Analysis)")
    print("  3. Per-phenotype NB-GLM (Costim-NBGLM-Pipeline)")
    print("  4. Dirichlet-Multinomial (new principled model)")
    print()
    print(f"Results will be saved to: {OUTPUT_DIR.absolute()}")
    print()

    # Run analyses
    fpr_results = run_fpr_analysis()
    power_results = run_power_analysis()
    data, elm_effects, detailed_metrics = run_detailed_example()

    # Save results
    fpr_results.to_csv(OUTPUT_DIR / "fpr_results.csv", index=False)
    power_results.to_csv(OUTPUT_DIR / "power_results.csv", index=False)
    detailed_metrics.to_csv(OUTPUT_DIR / "detailed_metrics.csv", index=False)

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print()

    # FPR summary
    print("1. TYPE I ERROR CALIBRATION (target: 0.05)")
    fpr_summary = fpr_results.groupby('method')['fpr'].mean()
    for method, fpr in fpr_summary.items():
        status = "✓" if 0.02 < fpr < 0.10 else "✗"
        print(f"   {status} {method}: FPR = {fpr:.3f}")
    print()

    # Power summary at effect=0.7
    print("2. POWER AT EFFECT SIZE = 0.7")
    power_07 = power_results[power_results['effect_size'] == 0.7].groupby('method')['tpr'].mean()
    for method, power in power_07.items():
        print(f"   {method}: Power = {power:.3f}")
    print()

    # Recommendations
    print("3. RECOMMENDATIONS")
    print()

    # Find best method
    best_fpr_method = fpr_summary.sub(0.05).abs().idxmin()
    best_power_method = power_07.idxmax()

    print(f"   Best FPR calibration: {best_fpr_method}")
    print(f"   Highest power: {best_power_method}")
    print()

    # Check if DM is best
    dm_fpr = fpr_summary.get('Dirichlet-Multinomial', np.nan)
    dm_power = power_07.get('Dirichlet-Multinomial', np.nan)

    if dm_fpr < 0.10 and dm_power > 0.5:
        print("   → Dirichlet-Multinomial is RECOMMENDED:")
        print("     - Correctly models the compositional structure")
        print("     - Well-calibrated Type I error")
        print("     - Good power for detecting true effects")
    else:
        print("   → Consider the trade-offs:")
        if dm_fpr > 0.10:
            print(f"     - DM has elevated FPR ({dm_fpr:.3f})")
        if dm_power < 0.5:
            print(f"     - DM has lower power ({dm_power:.3f})")

    print()
    print(f"Results saved to: {OUTPUT_DIR}")

    return {
        'fpr_results': fpr_results,
        'power_results': power_results,
        'detailed_metrics': detailed_metrics,
    }


if __name__ == "__main__":
    main()
