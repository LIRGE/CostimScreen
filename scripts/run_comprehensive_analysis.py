#!/usr/bin/env python3
"""
Comprehensive analysis script for CAR-T costimulatory domain screen.

Runs 4 analysis methods on multiple CCD subsets:

Subsets:
- non_gpcr: All non-GPCR CCDs (no ICD conditioning)
- non_gpcr_1, non_gpcr_2, non_gpcr_3_4: Non-GPCR by ICD ID
- gpcr: All GPCR CCDs (no ICD conditioning)
- gpcr_1, gpcr_2, gpcr_3, gpcr_4: GPCR by ICD ID

Analysis methods:
1. Mann-Whitney (mw): Pearson residuals + rank-based comparison
2. Joint NB-GLM (joint_nbglm): Single model with all phenotypes
3. Phenotype NB-GLM (phenotype_nbglm): Pooled T-subset models
4. Dirichlet-Multinomial (dm): Compositional model with PIP-based significance

Contrasts of interest:
- EM vs CM (pooled over PD1)
- Naive vs CM (pooled over PD1)
- Naive vs EM (pooled over PD1)
- PD1-Low vs PD1-High (pooled over T-subsets)

Significance criteria:
- MW, NB-GLM methods: BH-FDR q < 0.10
- DM: PIP > 0.5 AND 90% CI excludes zero

FDR Correction Note:
--------------------
FDR correction method is controlled by FDR_METHOD variable:
  - "global": BH correction across all ELM-phenotype pairs (more conservative)
  - "phenotype": BH correction within each phenotype/contrast (less conservative)

Default is "phenotype" for increased sensitivity.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add src to path
BASE_PATH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_PATH / 'src'))

import costim_screen as cs

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

DATA_PATH = BASE_PATH / 'data'
RESULTS_PATH = BASE_PATH / 'results'

# Raji column patterns (for raw count data)
# Note: Column names use "Naïve" (with accent) not "Naive"
RAJI_PHENOTYPE_PATTERNS = {
    "Naive_High": "CAR:Raji_Naïve_High",
    "Naive_Low": "CAR:Raji_Naïve_Low",
    "CM_High": "CAR:Raji_CM_High",
    "CM_Low": "CAR:Raji_CM_Low",
    "EM_High": "CAR:Raji_EM_High",
    "EM_Low": "CAR:Raji_EM_Low",
}

PHENOTYPES = ["Naive_High", "Naive_Low", "CM_High", "CM_Low", "EM_High", "EM_Low"]
TSUBSETS = ["Naive", "CM", "EM"]

# Contrasts of interest
TSUBSET_CONTRASTS = [
    ("EM", "CM"),      # EM vs CM
    ("Naive", "CM"),   # Naive vs CM
    ("Naive", "EM"),   # Naive vs EM
]

# =============================================================================
# Analysis Settings
# =============================================================================

# FDR Correction Method
# Options: "global" or "phenotype"
#   - "global": Apply BH correction across all ELM-phenotype pairs (more conservative)
#   - "phenotype": Apply BH correction within each phenotype separately (less conservative)
FDR_METHOD = "phenotype"  # Change to "global" for more conservative correction

# Minimum ELM frequency threshold
# Set to 0 to include all ELMs (no filtering)
MIN_ELM_FREQ = 0  # Include all ELMs - no filtering


def apply_fdr_correction(df, pvalue_col="pvalue", group_col="phenotype"):
    """Apply FDR correction based on FDR_METHOD setting.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing p-values
    pvalue_col : str
        Column name for p-values
    group_col : str
        Column name for grouping (used when FDR_METHOD="phenotype")

    Returns
    -------
    pd.Series
        q-values (FDR-adjusted p-values)
    """
    if len(df) == 0:
        return pd.Series(dtype=float)

    if FDR_METHOD == "global":
        # Global correction across all tests
        return cs.bh_fdr(df[pvalue_col])

    elif FDR_METHOD == "phenotype":
        # Phenotype-specific correction
        if group_col not in df.columns:
            # Fall back to global if no group column
            return cs.bh_fdr(df[pvalue_col])

        qvalues = pd.Series(index=df.index, dtype=float)
        for group in df[group_col].unique():
            mask = df[group_col] == group
            qvalues.loc[mask] = cs.bh_fdr(df.loc[mask, pvalue_col])
        return qvalues

    else:
        raise ValueError(f"Unknown FDR_METHOD: {FDR_METHOD}. Use 'global' or 'phenotype'.")


# =============================================================================
# Data Loading and Filtering
# =============================================================================

def load_and_filter_data(subset_type: str) -> tuple:
    """
    Load data and filter to specified subset.

    Parameters
    ----------
    subset_type : str
        Either 'non_gpcr' or 'gpcr_icd34'

    Returns
    -------
    counts_raji : pd.DataFrame
        Count matrix filtered to Raji samples
    counts_wide : pd.DataFrame
        Full count matrix (for NB-GLM)
    cand : pd.DataFrame
        Candidate metadata
    smeta : pd.DataFrame
        Sample metadata
    elm_df : pd.DataFrame
        ELM assignments
    all_elms : list
        List of unique ELMs
    """
    # Load data
    counts_wide = cs.load_counts_matrix(
        DATA_PATH / 'merged_counts.xlsx',
        candidate_id_col="CandidateID"
    )
    smeta = cs.load_sample_metadata(DATA_PATH / 'sample_metadata.xlsx')
    cand = cs.load_candidate_metadata(DATA_PATH / 'candidate_metadata.xlsx')
    topol = pd.read_excel(DATA_PATH / 'costim_topol_protein_families.xlsx')
    topol = topol.set_index('ID')

    # Align
    common = counts_wide.index.intersection(cand.index).intersection(topol.index)
    counts_wide = counts_wide.loc[common]
    cand = cand.loc[common]
    topol = topol.loc[common]

    # Add ICD ID to candidate metadata
    cand['ICD_ID'] = topol['ICD ID']
    cand['Gene_ICD_Mult'] = topol['Gene ICD Multiplicity']

    # Align samples
    common_samples = counts_wide.columns.intersection(smeta.index)
    counts_wide = counts_wide[common_samples]
    smeta = smeta.loc[common_samples]
    # Preserve index name after subsetting (pandas loses it during loc with Index)
    smeta.index.name = "sample_id"

    # Filter by count threshold
    counts_wide = cs.filter_domains_by_total_counts(counts_wide, min_total=50)
    cand = cand.loc[counts_wide.index]

    print(f"After count filtering: {len(counts_wide)} CCDs")

    # Apply subset filter
    # Naming convention:
    #   non_gpcr   = all non-GPCR (no ICD conditioning)
    #   non_gpcr_1 = non-GPCR with ICD ID = 1
    #   gpcr_1     = GPCR with ICD ID = 1
    if subset_type == 'non_gpcr':
        # Non-GPCR: is_gpcr == 0 (Num ICD != 4), any ICD ID
        mask = cand['is_gpcr'] == 0
        subset_ids = cand.index[mask]
        print(f"Non-GPCR CCDs (any ICD ID): {len(subset_ids)}")
    elif subset_type == 'non_gpcr_3_4':
        # Non-GPCR with ICD ID in {3, 4} (combined due to small sample sizes)
        mask = (cand['is_gpcr'] == 0) & (cand['ICD_ID'].isin([3, 4]))
        subset_ids = cand.index[mask]
        print(f"Non-GPCR CCDs with ICD ID in {{3,4}}: {len(subset_ids)}")
    elif subset_type.startswith('non_gpcr_'):
        # Non-GPCR with specific ICD ID (e.g., non_gpcr_1, non_gpcr_2, etc.)
        icd_id = int(subset_type.split('_')[-1])
        mask = (cand['is_gpcr'] == 0) & (cand['ICD_ID'] == icd_id)
        subset_ids = cand.index[mask]
        print(f"Non-GPCR CCDs with ICD ID = {icd_id}: {len(subset_ids)}")
    elif subset_type == 'gpcr':
        # GPCR: is_gpcr == 1 (Num ICD == 4), any ICD ID
        mask = cand['is_gpcr'] == 1
        subset_ids = cand.index[mask]
        print(f"GPCR CCDs (any ICD ID): {len(subset_ids)}")
    elif subset_type.startswith('gpcr_'):
        # GPCR with specific ICD ID (e.g., gpcr_1, gpcr_2, etc.)
        icd_id = int(subset_type.split('_')[-1])
        mask = (cand['is_gpcr'] == 1) & (cand['ICD_ID'] == icd_id)
        subset_ids = cand.index[mask]
        print(f"GPCR CCDs with ICD ID = {icd_id}: {len(subset_ids)}")
    elif subset_type == 'gpcr_icd34':
        # GPCR with ICD ID in {3, 4} (legacy)
        mask = (cand['is_gpcr'] == 1) & (cand['ICD_ID'].isin([3, 4]))
        subset_ids = cand.index[mask]
        print(f"GPCR CCDs with ICD ID in {{3,4}}: {len(subset_ids)}")
    else:
        raise ValueError(f"Unknown subset_type: {subset_type}")

    counts_wide = counts_wide.loc[subset_ids]
    cand = cand.loc[subset_ids]

    # Filter to Raji samples only for MW analysis
    raji_cols = [c for c in counts_wide.columns if 'CAR:Raji' in c]
    counts_raji = counts_wide[raji_cols]
    print(f"Raji samples: {len(raji_cols)}")

    # Build ELM assignments
    elm_df = cand[["ELMCategory"]].copy()
    elm_df["ELMs"] = elm_df["ELMCategory"].apply(cs.split_elm_list)

    # Get unique ELMs
    all_elms = sorted(set(elm for elms in elm_df["ELMs"] for elm in elms))
    print(f"Unique ELMs: {len(all_elms)}")

    return counts_raji, counts_wide, cand, smeta, elm_df, all_elms


# =============================================================================
# Analysis Method 1: Mann-Whitney
# =============================================================================

def run_mw_analysis(counts_raji, cand, elm_df, all_elms, output_path):
    """
    Run Mann-Whitney analysis using costim_screen.
    Uses Pearson residuals and compares CCDs with/without each ELM.
    """
    print("\n=== Running Mann-Whitney Analysis ===")
    output_path.mkdir(parents=True, exist_ok=True)

    # Compute Pearson residuals from filtered counts
    residuals_df = cs.compute_pearson_residuals(counts_raji)

    # Run MW analysis for each phenotype
    all_results = []
    for pheno, pattern in RAJI_PHENOTYPE_PATTERNS.items():
        results = cs.run_mw_analysis_for_phenotype(
            residuals_df, elm_df, pheno, pattern, elm_col="ELMs"
        )
        if len(results) > 0:
            all_results.append(results)

    if not all_results:
        print("  No results generated")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    mw_results = pd.concat(all_results, ignore_index=True)

    # Apply FDR correction (respects FDR_METHOD setting)
    mw_results["qvalue"] = apply_fdr_correction(mw_results, "pvalue", "phenotype")
    mw_results["sig_stars"] = mw_results["qvalue"].apply(cs.get_sig_stars)

    # Compute pooled T-subset effects
    pooled_tsubset = []
    for elm in all_elms:
        for ts in TSUBSETS:
            pooled_tsubset.append(cs.compute_pooled_tsubset_effect(mw_results, elm, ts))
    pooled_df = pd.DataFrame(pooled_tsubset)
    pooled_df["qvalue"] = apply_fdr_correction(pooled_df, "pvalue", "tsubset")

    # Compute T-subset contrasts
    contrast_results = compute_mw_tsubset_contrasts(pooled_df)

    # Compute PD1 contrast (pooled over T-subsets)
    pd1_contrast = compute_mw_pd1_contrast(mw_results, all_elms)

    # Save results
    mw_results.to_csv(output_path / "mw_all_results.csv", index=False)
    pooled_df.to_csv(output_path / "mw_pooled_tsubset.csv", index=False)
    contrast_results.to_csv(output_path / "mw_tsubset_contrasts.csv", index=False)
    pd1_contrast.to_csv(output_path / "mw_pd1_contrast.csv", index=False)

    # Create plots
    create_mw_plots(mw_results, pooled_df, contrast_results, pd1_contrast, output_path)

    print(f"  Saved results to {output_path}")
    return mw_results, contrast_results, pd1_contrast


def compute_mw_tsubset_contrasts(pooled_df):
    """Compute T-subset contrasts from pooled MW results."""
    results = []
    for ts1, ts2 in TSUBSET_CONTRASTS:
        df1 = pooled_df[pooled_df['tsubset'] == ts1].set_index('ELM')
        df2 = pooled_df[pooled_df['tsubset'] == ts2].set_index('ELM')
        common_elms = df1.index.intersection(df2.index)

        for elm in common_elms:
            delta1 = df1.loc[elm, 'cliff_delta']
            delta2 = df2.loc[elm, 'cliff_delta']
            diff = delta1 - delta2

            # Combine p-values using Fisher's method (rough approximation)
            p1 = df1.loc[elm, 'pvalue']
            p2 = df2.loc[elm, 'pvalue']
            # Use min p-value as conservative estimate
            combined_p = min(p1, p2) * 2  # Bonferroni-style
            combined_p = min(combined_p, 1.0)

            results.append({
                'ELM': elm,
                'contrast': f"{ts1}_vs_{ts2}",
                'effect_ts1': delta1,
                'effect_ts2': delta2,
                'diff': diff,
                'pvalue': combined_p,
            })

    df = pd.DataFrame(results)
    if len(df) > 0:
        df['qvalue'] = apply_fdr_correction(df, 'pvalue', 'contrast')
    return df


def compute_mw_pd1_contrast(mw_results, all_elms):
    """Compute PD1 High vs Low contrast pooled over T-subsets."""
    results = []
    for elm in all_elms:
        elm_data = mw_results[mw_results['ELM'] == elm]

        high_data = elm_data[elm_data['phenotype'].str.contains('High')]
        low_data = elm_data[elm_data['phenotype'].str.contains('Low')]

        if len(high_data) == 0 or len(low_data) == 0:
            continue

        # Average effect across T-subsets
        high_effect = high_data['cliff_delta'].mean()
        low_effect = low_data['cliff_delta'].mean()
        diff = low_effect - high_effect  # Low vs High

        # Combine p-values
        all_p = list(high_data['pvalue']) + list(low_data['pvalue'])
        combined_p = min(all_p) * len(all_p)
        combined_p = min(combined_p, 1.0)

        results.append({
            'ELM': elm,
            'contrast': 'PD1Low_vs_PD1High',
            'effect_low': low_effect,
            'effect_high': high_effect,
            'diff': diff,
            'pvalue': combined_p,
        })

    df = pd.DataFrame(results)
    if len(df) > 0:
        # PD1 contrast is a single contrast, so global correction is appropriate
        df['qvalue'] = cs.bh_fdr(df['pvalue'])
    return df


def create_mw_plots(mw_results, pooled_df, contrast_results, pd1_contrast, output_path):
    """Create volcano plots and heatmaps for MW results.

    Uses consistent volcano plot style matching cs.volcano_plot:
    - q_thresh=0.10, effect threshold=0.10
    - Blue/red coloring by effect direction
    - Top labels for significant hits
    """
    plots_path = output_path / 'plots'
    plots_path.mkdir(exist_ok=True)

    q_thresh = 0.10
    effect_thresh = 0.10

    # Individual volcano plots for T-subset contrasts
    for contrast_name in contrast_results['contrast'].unique():
        df = contrast_results[contrast_results['contrast'] == contrast_name].copy()
        if len(df) == 0:
            continue

        df['neglog10_q'] = -np.log10(df['qvalue'].clip(lower=1e-300))

        fig, ax = plt.subplots(figsize=(8, 6))

        # Color by effect direction (consistent with cs.volcano_plot)
        colors = np.where(df['diff'] >= 0, "#3182bd", "#e34a33")  # blue/red
        ax.scatter(df['diff'], df['neglog10_q'], c=colors, s=12, alpha=0.7)

        # Threshold lines
        ax.axhline(-np.log10(q_thresh), color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.axvline(+effect_thresh, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.axvline(-effect_thresh, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

        # Add labels for top 10 most significant
        sig_df = df.nsmallest(10, 'qvalue')
        for _, row in sig_df.iterrows():
            ax.text(row['diff'], row['neglog10_q'], row['ELM'], fontsize=8)

        ax.set_xlabel("Effect Difference (Cliff's δ)")
        ax.set_ylabel(r"$-\log_{10}$ q-value")
        ts1, ts2 = contrast_name.split('_vs_')
        ax.set_title(f"{ts1} vs {ts2}")

        plt.tight_layout()
        fig.savefig(plots_path / f"volcano_{ts1}_vs_{ts2}_pooledPD1.png", dpi=200)
        plt.close(fig)
        print(f"    Created volcano: {ts1} vs {ts2}")

    # PD1 volcano plot
    if len(pd1_contrast) > 0:
        pd1_contrast['neglog10_q'] = -np.log10(pd1_contrast['qvalue'].clip(lower=1e-300))

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = np.where(pd1_contrast['diff'] >= 0, "#3182bd", "#e34a33")
        ax.scatter(pd1_contrast['diff'], pd1_contrast['neglog10_q'], c=colors, s=12, alpha=0.7)

        ax.axhline(-np.log10(q_thresh), color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.axvline(+effect_thresh, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.axvline(-effect_thresh, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

        sig_df = pd1_contrast.nsmallest(10, 'qvalue')
        for _, row in sig_df.iterrows():
            ax.text(row['diff'], row['neglog10_q'], row['ELM'], fontsize=8)

        ax.set_xlabel("Effect Difference (Cliff's δ)")
        ax.set_ylabel(r"$-\log_{10}$ q-value")
        ax.set_title("PD1-Low vs PD1-High")

        plt.tight_layout()
        fig.savefig(plots_path / "volcano_PD1Low_vs_PD1High_pooledTsubset.png", dpi=200)
        plt.close(fig)
        print(f"    Created volcano: PD1-Low vs PD1-High")

    # Pooled heatmap (T-subsets pooled over PD1, plus PD1_Low pooled over T-subsets)
    create_mw_pooled_heatmap(pooled_df, mw_results, plots_path, "heatmap_pooled.png")

    # Assemble joint figure with all volcanos and heatmap
    try:
        cs.assemble_dm_main_figure(
            plots_path,
            out_png="figure_mw_volcanos_plus_heatmap.png",
            out_pdf=None,
        )
        print(f"    Created joint figure")
    except Exception as e:
        print(f"  Warning: joint figure failed: {e}")


def create_mw_pooled_heatmap(pooled_df, mw_results, plots_path, filename):
    """Create pooled heatmap matching cs.pooled_coef_heatmap style.

    Columns: Naive, CM, EM (pooled over PD1), PD1_Low (pooled over T-subsets)
    Values: Cliff's delta (effect size)
    """
    if len(pooled_df) == 0 and len(mw_results) == 0:
        return

    # Get all ELMs
    all_elms = sorted(pooled_df['ELM'].unique()) if len(pooled_df) > 0 else []

    # Build heatmap data with columns: Naive, CM, EM, PD1_Low
    rows = []
    for elm in all_elms:
        row = {'motif': elm}

        # T-subset columns from pooled_df (already pooled over PD1)
        for ts in ['Naive', 'CM', 'EM']:
            ts_data = pooled_df[(pooled_df['ELM'] == elm) & (pooled_df['tsubset'] == ts)]
            if len(ts_data) > 0:
                row[ts] = ts_data['cliff_delta'].iloc[0]
            else:
                row[ts] = np.nan

        # PD1_Low column: average Low effects across T-subsets
        if len(mw_results) > 0:
            low_data = mw_results[
                (mw_results['ELM'] == elm) &
                (mw_results['phenotype'].str.contains('Low'))
            ]
            if len(low_data) > 0:
                row['PD1_Low'] = low_data['cliff_delta'].mean()
            else:
                row['PD1_Low'] = np.nan
        else:
            row['PD1_Low'] = np.nan

        rows.append(row)

    data = pd.DataFrame(rows).set_index('motif')

    # Drop rows that are all NaN
    data = data.dropna(how='all')

    if len(data) == 0:
        return

    # Reorder columns
    col_order = [c for c in ['Naive', 'CM', 'EM', 'PD1_Low'] if c in data.columns]
    data = data[col_order]

    # Sort by max absolute effect
    data = data.loc[data.abs().max(axis=1).sort_values(ascending=False).index]

    # Auto-size figure
    n_rows, n_cols = data.shape
    figsize = (max(4, n_cols * 0.8 + 2), max(4, n_rows * 0.3 + 1))

    # Symmetric colormap
    abs_max = np.nanmax(np.abs(data.values))
    vmin, vmax = -abs_max, abs_max

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        data.values,
        aspect='auto',
        cmap='RdBu_r',
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest',
    )

    # Axis labels
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.index, fontsize=8)

    ax.set_xlabel('Pooled Phenotype')
    ax.set_ylabel('Motif')
    ax.set_title("MW Effect Sizes by Pooled Phenotype")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cliff's Delta")

    fig.tight_layout()
    fig.savefig(plots_path / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Also save as CSV
    data.to_csv(plots_path.parent / "heatmap_pooled_zscores.csv")
    print(f"    Created pooled heatmap")




# =============================================================================
# Analysis Method 2: Joint NB-GLM
# =============================================================================

def run_joint_nbglm_analysis(counts_wide, cand, smeta, elm_df, all_elms, output_path):
    """
    Run our joint NB-GLM analysis using costim_screen.
    Fits a single model with all phenotypes and ELM interactions.
    Following the pattern from notebooks/01_refit_joint_model.ipynb
    """
    print("\n=== Running Joint NB-GLM Analysis ===")
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Build ELM design matrix with patsy-safe names
        # For small datasets, we need stricter filtering to avoid rank-deficiency
        n_ccds = len(cand)
        min_elm_count = max(3, int(0.05 * n_ccds))  # At least 5% of CCDs or 3
        max_elm_count = int(0.95 * n_ccds)  # At most 95% of CCDs

        X_elm = cs.build_elm_design(
            cand,
            elms_col="ELMCategory",
            min_freq=MIN_ELM_FREQ
        )

        # Filter out ELMs that are too rare or too common (causes rank deficiency)
        elm_counts = X_elm.sum(axis=0)
        valid_elms = (elm_counts >= min_elm_count) & (elm_counts <= max_elm_count)
        X_elm = X_elm.loc[:, valid_elms]

        if len(X_elm.columns) == 0:
            print(f"  No ELMs pass frequency filter (need {min_elm_count}-{max_elm_count} CCDs)")
            return pd.DataFrame()

        # Make patsy-safe column names
        safe_cols, elm_mapping = cs.make_patsy_safe_columns(list(X_elm.columns))
        X_elm.columns = safe_cols
        motif_cols = list(X_elm.columns)
        print(f"  ELM features: {len(motif_cols)} (filtered from {valid_elms.sum()} valid)")

        # Convert to long format
        df = cs.counts_to_long(counts_wide, id_col="CandidateID")
        df = df.merge(smeta.reset_index(), on="sample_id", how="left")
        df = cs.add_library_size(df)
        df = df.merge(X_elm.reset_index().rename(columns={"index": "CandidateID"}),
                      on="CandidateID", how="left")

        # Create phenotype variable
        df["phenotype"] = df["Tsubset"] + "_" + df["PD1Status"]
        df["phenotype"] = df["phenotype"].apply(cs.normalize_phenotype)

        # Create CCR for clustering
        df["CCR"] = df["Donor"].astype(str) + "_" + df["Replicate"].astype(str)

        # Filter to Raji condition
        df_raji = df[df["ExpCond"].str.contains("Raji", na=False)].copy()
        print(f"  Raji observations: {len(df_raji)}")

        if len(df_raji) == 0:
            print("  No Raji data found")
            return pd.DataFrame()

        # Build and fit joint model
        formula = cs.build_joint_formula(motif_cols)
        print(f"  Fitting model...")

        # Try fitting with fallback options if standard fit fails
        fit = None
        try:
            fit = cs.fit_nb_glm_iter_alpha(
                df_raji,
                formula=formula,
                offset_col="offset",
                cluster_col="CCR"
            )
            print(f"  Dispersion (alpha): {fit.alpha:.3f}")
        except ValueError as e:
            if "NaN" in str(e) or "inf" in str(e):
                print(f"  Standard fit failed, trying with fixed alpha=1.0...")
                # Fallback: use fixed dispersion
                import statsmodels.api as sm
                import patsy
                y, X = patsy.dmatrices(formula, df_raji, return_type='dataframe')
                offset = np.log(df_raji['library_size'].values + 1)
                try:
                    model = sm.GLM(y.values.ravel(), X,
                                   family=sm.families.NegativeBinomial(alpha=1.0),
                                   offset=offset)
                    glm_result = model.fit(maxiter=100, method='bfgs')  # BFGS more stable than IRLS
                    # Create a minimal fit object
                    fit = type('Fit', (), {
                        'result': glm_result,
                        'alpha': 1.0,
                        'motif_cols': motif_cols,
                    })()
                    print(f"  Fallback fit succeeded with fixed alpha=1.0")
                except Exception as e2:
                    print(f"  Fallback fit also failed: {e2}")
                    raise
            else:
                raise

        if fit is None:
            print("  Model fitting failed completely")
            return pd.DataFrame()

        # Create plots directory
        plots_path = output_path / 'plots'
        plots_path.mkdir(exist_ok=True)

        # Compute pooled T-subset contrasts with volcano plots saved via outpath
        contrast_tables = []
        for ts_p, ts_q in TSUBSET_CONTRASTS:
            try:
                tab = cs.volcano_tsubset_pooled_pd1(
                    fit, motif_cols,
                    tsubset_p=ts_p,
                    tsubset_q=ts_q,
                    q_thresh=0.10,
                    lfc_thresh=0.10,
                    title=f"{ts_p} vs {ts_q}",
                    outpath=plots_path / f"volcano_{ts_p}_vs_{ts_q}_pooledPD1.png",
                )
                tab["contrast"] = f"{ts_p}_vs_{ts_q}"
                contrast_tables.append(tab)
                print(f"    Created volcano: {ts_p} vs {ts_q}")
            except Exception as e:
                print(f"  Warning: {ts_p} vs {ts_q} failed: {e}")
                continue

        # Compute PD1 contrast with volcano plot saved via outpath
        try:
            tab_pd1 = cs.volcano_pd1_pooled_tsubset(
                fit, motif_cols,
                pd1_high="Low",
                pd1_low="High",
                q_thresh=0.10,
                lfc_thresh=0.10,
                title="PD1-Low vs PD1-High",
                outpath=plots_path / "volcano_PD1Low_vs_PD1High_pooledTsubset.png",
            )
            tab_pd1["contrast"] = "PD1Low_vs_PD1High"
            contrast_tables.append(tab_pd1)
            print(f"    Created volcano: PD1-Low vs PD1-High")
        except Exception as e:
            print(f"  Warning: PD1 contrast failed: {e}")

        if not contrast_tables:
            print("  No contrasts computed")
            return pd.DataFrame()

        results_df = pd.concat(contrast_tables, ignore_index=True)

        # Re-apply FDR correction to respect FDR_METHOD setting
        # The cs.volcano_* functions apply global BH by default
        if FDR_METHOD == "phenotype":
            # Re-compute q-values per contrast (phenotype-specific)
            results_df['qvalue'] = apply_fdr_correction(results_df, 'pvalue', 'contrast')
            results_df['neglog10_q'] = -np.log10(results_df['qvalue'].clip(lower=1e-300))

        # Save results
        results_df.to_csv(output_path / "nbglm_contrasts.csv", index=False)

        # Create pooled heatmap using cs.pooled_coef_heatmap with outpath
        try:
            fig, ax, heatmap_data = cs.pooled_coef_heatmap(
                fit,
                motifs=motif_cols,
                title="Motif Z-scores by Pooled Phenotype",
                outpath=plots_path / "heatmap_pooled.png",
                dpi=300,
            )
            plt.close(fig)
            heatmap_data.to_csv(output_path / "heatmap_pooled_zscores.csv")
            print(f"    Created pooled heatmap")
        except Exception as e:
            print(f"  Warning: pooled heatmap failed: {e}")

        # Assemble joint figure with all volcanos and heatmap
        try:
            cs.assemble_dm_main_figure(
                plots_path,
                out_png="figure_joint_nbglm_volcanos_plus_heatmap.png",
                out_pdf=None,
            )
            print(f"    Created joint figure")
        except Exception as e:
            print(f"  Warning: joint figure failed: {e}")

        print(f"  Saved results to {output_path}")
        return results_df

    except Exception as e:
        print(f"  Error in joint NB-GLM: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


# =============================================================================
# Analysis Method 3: Phenotype NB-GLM (Pooled Approach)
# =============================================================================

def run_phenotype_nbglm_analysis(counts_wide, cand, smeta, elm_df, all_elms, output_path):
    """
    Run phenotype-specific NB-GLM analysis with POOLED approach.

    This approach:
    1. Pools High+Low for each T-subset (CM, EM, Naive) - 12 samples per model
    2. Fits separate NB-GLM per POOLED T-subset with ALL ELMs as covariates
    3. Uses fixed dispersion alpha=1.15
    4. Computes Wald tests comparing coefficients between pooled T-subsets
    5. Applies BH-FDR correction to contrast p-values
    """
    print("\n=== Running Phenotype NB-GLM Analysis (Pooled) ===")
    output_path.mkdir(parents=True, exist_ok=True)

    import statsmodels.api as sm

    try:
        # Filter to Raji samples
        raji_cols = [c for c in counts_wide.columns if 'CAR:Raji' in c]
        counts_raji = counts_wide[raji_cols]
        n_ccds = len(counts_raji)

        # Build multi-hot ELM design matrix
        elm_lists = cand['ELMCategory'].apply(cs.split_elm_list).tolist()
        all_unique_elms = sorted({elm for elms in elm_lists for elm in elms})

        X_elm_full = np.zeros((n_ccds, len(all_unique_elms)), dtype=int)
        for i, elms in enumerate(elm_lists):
            for elm in elms:
                X_elm_full[i, all_unique_elms.index(elm)] = 1

        # Filter ELMs: need at least 5% of CCDs (or 3) and at most 95%
        min_elm_count = max(3, int(0.05 * n_ccds))
        max_elm_count = int(0.95 * n_ccds)
        elm_counts = X_elm_full.sum(axis=0)
        valid_elm_mask = (elm_counts >= min_elm_count) & (elm_counts <= max_elm_count)

        unique_elms = [elm for elm, valid in zip(all_unique_elms, valid_elm_mask) if valid]
        X_elm = X_elm_full[:, valid_elm_mask]

        if len(unique_elms) == 0:
            print(f"  No ELMs pass frequency filter (need {min_elm_count}-{max_elm_count} CCDs)")
            return pd.DataFrame()

        print(f"  ELM features: {len(unique_elms)} (filtered, need {min_elm_count}-{max_elm_count} CCDs)")

        # Fit pooled models for each T-subset (pooling High+Low)
        tsubset_fits = {}

        for tsubset in TSUBSETS:
            # Match columns for this T-subset (both High and Low)
            # Use Naïve (with accent) for column matching
            tsubset_pattern = tsubset if tsubset != "Naive" else "Naïve"

            tsubset_cols = [c for c in raji_cols if tsubset_pattern in c]
            n_samples = len(tsubset_cols)

            if n_samples == 0:
                print(f"  Skipping {tsubset} (no matching columns)")
                continue

            print(f"  Fitting {tsubset}: {n_ccds} CCDs × {n_samples} samples = {n_ccds * n_samples} obs")

            # Stack counts into vector (reshape uses C-order: row0-all-cols, row1-all-cols, ...)
            y = counts_raji[tsubset_cols].to_numpy().reshape(-1)

            # Repeat design matrix to match reshape order (each row repeated n_samples times)
            X_stacked = np.repeat(X_elm, n_samples, axis=0)

            # Add intercept
            X_stacked = sm.add_constant(X_stacked, has_constant='add')

            # Fit NB-GLM with fixed alpha=1.15, with fallback methods
            res = None
            for method, alpha in [('IRLS', 1.15), ('bfgs', 1.15), ('bfgs', 1.0), ('bfgs', 0.5)]:
                try:
                    model = sm.GLM(y, X_stacked, family=sm.families.NegativeBinomial(alpha=alpha))
                    if method == 'IRLS':
                        res = model.fit(maxiter=200, tol=1e-8)
                    else:
                        res = model.fit(maxiter=200, method=method)
                    break  # Success
                except (ValueError, np.linalg.LinAlgError) as e:
                    if method == 'bfgs' and alpha == 0.5:
                        print(f"    {tsubset} fitting failed with all methods: {e}")
                    continue

            if res is None:
                print(f"    Skipping {tsubset} (all fitting methods failed)")
                continue

            tsubset_fits[tsubset] = {
                'params': res.params[1:],  # Skip intercept
                'bse': res.bse[1:],
                'pvalues': res.pvalues[1:],
                'tvalues': res.tvalues[1:],
            }

        if len(tsubset_fits) < 2:
            print("  Not enough T-subsets fitted")
            return pd.DataFrame()

        # Compute Wald tests between T-subsets
        contrast_results = []
        for ts1, ts2 in TSUBSET_CONTRASTS:
            if ts1 not in tsubset_fits or ts2 not in tsubset_fits:
                continue

            fit1 = tsubset_fits[ts1]
            fit2 = tsubset_fits[ts2]

            for i, elm in enumerate(unique_elms):
                coef1, se1 = fit1['params'][i], fit1['bse'][i]
                coef2, se2 = fit2['params'][i], fit2['bse'][i]

                diff = coef1 - coef2
                se_diff = np.sqrt(se1**2 + se2**2)
                z = diff / se_diff if se_diff > 0 else 0
                pval = 2 * stats.norm.sf(abs(z))

                contrast_results.append({
                    'motif': elm,
                    'contrast': f"{ts1}_vs_{ts2}",
                    f'{ts1}_coef': coef1,
                    f'{ts2}_coef': coef2,
                    'logFC': diff,
                    'se': se_diff,
                    'zscore': z,
                    'pvalue': pval,
                })

        # Also compute PD1 contrast (High vs Low, pooled over T-subsets)
        # Fit High and Low pooled models
        for pd1 in ['High', 'Low']:
            pd1_cols = [c for c in raji_cols if pd1 in c]
            n_samples = len(pd1_cols)

            if n_samples == 0:
                continue

            y = counts_raji[pd1_cols].to_numpy().reshape(-1)
            X_stacked = np.repeat(X_elm, n_samples, axis=0)
            X_stacked = sm.add_constant(X_stacked, has_constant='add')

            # Fit with fallback methods
            res = None
            for method, alpha in [('IRLS', 1.15), ('bfgs', 1.15), ('bfgs', 1.0), ('bfgs', 0.5)]:
                try:
                    model = sm.GLM(y, X_stacked, family=sm.families.NegativeBinomial(alpha=alpha))
                    if method == 'IRLS':
                        res = model.fit(maxiter=200, tol=1e-8)
                    else:
                        res = model.fit(maxiter=200, method=method)
                    break
                except (ValueError, np.linalg.LinAlgError):
                    continue

            if res is None:
                continue

            tsubset_fits[pd1] = {
                'params': res.params[1:],
                'bse': res.bse[1:],
            }

        # PD1 Low vs High contrast
        if 'Low' in tsubset_fits and 'High' in tsubset_fits:
            fit_low = tsubset_fits['Low']
            fit_high = tsubset_fits['High']

            for i, elm in enumerate(unique_elms):
                coef_low, se_low = fit_low['params'][i], fit_low['bse'][i]
                coef_high, se_high = fit_high['params'][i], fit_high['bse'][i]

                diff = coef_low - coef_high
                se_diff = np.sqrt(se_low**2 + se_high**2)
                z = diff / se_diff if se_diff > 0 else 0
                pval = 2 * stats.norm.sf(abs(z))

                contrast_results.append({
                    'motif': elm,
                    'contrast': 'PD1Low_vs_PD1High',
                    'Low_coef': coef_low,
                    'High_coef': coef_high,
                    'logFC': diff,
                    'se': se_diff,
                    'zscore': z,
                    'pvalue': pval,
                })

        if not contrast_results:
            print("  No contrasts computed")
            return pd.DataFrame()

        results_df = pd.DataFrame(contrast_results)

        # Apply BH-FDR correction per contrast
        results_df['qvalue'] = apply_fdr_correction(results_df, 'pvalue', 'contrast')

        # Save results
        results_df.to_csv(output_path / "nbglm_contrasts.csv", index=False)

        # Also save per-Tsubset coefficients
        tsubset_coefs = []
        for ts, fit in tsubset_fits.items():
            if ts in TSUBSETS or ts in ['High', 'Low']:
                for i, elm in enumerate(unique_elms):
                    tsubset_coefs.append({
                        'motif': elm,
                        'tsubset': ts,
                        'coef': fit['params'][i],
                        'se': fit['bse'][i],
                    })
        pd.DataFrame(tsubset_coefs).to_csv(output_path / "nbglm_tsubset_coefs.csv", index=False)

        # Create plots
        plots_path = output_path / 'plots'
        plots_path.mkdir(exist_ok=True)

        # Pooled heatmap from T-subset coefficients
        create_phenotype_nbglm_pooled_heatmap_from_tsubsets(tsubset_fits, unique_elms, plots_path, "heatmap_pooled.png")

        # Contrast volcano plots
        create_contrast_volcano_plots(results_df, plots_path, "Phenotype NB-GLM")

        # Assemble joint figure
        try:
            cs.assemble_dm_main_figure(
                plots_path,
                out_png="figure_phenotype_nbglm_volcanos_plus_heatmap.png",
                out_pdf=None,
            )
            print(f"    Created joint figure")
        except Exception as e:
            print(f"  Warning: joint figure failed: {e}")

        print(f"  Saved results to {output_path}")
        return results_df

    except Exception as e:
        print(f"  Error in phenotype NB-GLM: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def create_phenotype_nbglm_pooled_heatmap_from_tsubsets(tsubset_fits, unique_elms, plots_path, filename):
    """Create pooled heatmap from T-subset fits (pooled approach).

    Columns: Naive, CM, EM, PD1_Low
    Values: Coefficients (log-scale effects)
    """
    # Build heatmap data from tsubset_fits
    rows = []
    for i, elm in enumerate(unique_elms):
        row = {'motif': elm}

        # T-subset columns
        for ts in ['Naive', 'CM', 'EM']:
            if ts in tsubset_fits:
                row[ts] = tsubset_fits[ts]['params'][i]  # numpy array indexing
            else:
                row[ts] = np.nan

        # PD1_Low column
        if 'Low' in tsubset_fits:
            row['PD1_Low'] = tsubset_fits['Low']['params'][i]  # numpy array indexing
        else:
            row['PD1_Low'] = np.nan

        rows.append(row)

    data = pd.DataFrame(rows).set_index('motif')
    data = data.dropna(how='all')

    if len(data) == 0:
        return

    # Reorder columns
    col_order = [c for c in ['Naive', 'CM', 'EM', 'PD1_Low'] if c in data.columns]
    data = data[col_order]

    # Sort by max absolute coefficient
    data = data.loc[data.abs().max(axis=1).sort_values(ascending=False).index]

    # Auto-size figure
    n_rows, n_cols = data.shape
    figsize = (max(4, n_cols * 0.8 + 2), max(4, n_rows * 0.3 + 1))

    # Symmetric colormap
    abs_max = np.nanmax(np.abs(data.values))
    vmin, vmax = -abs_max, abs_max

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        data.values,
        aspect='auto',
        cmap='RdBu_r',
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest',
    )

    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.index, fontsize=8)

    ax.set_xlabel('Pooled Phenotype')
    ax.set_ylabel('Motif')
    ax.set_title("NB-GLM Coefficients by Pooled Phenotype")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Coefficient')

    fig.tight_layout()
    fig.savefig(plots_path / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

    data.to_csv(plots_path.parent / "heatmap_pooled_coefs.csv")
    print(f"    Created pooled heatmap")


def create_phenotype_nbglm_pooled_heatmap(results_df, plots_path, filename):
    """Create pooled heatmap matching cs.pooled_coef_heatmap style (legacy).

    Columns: Naive, CM, EM (pooled over PD1), PD1_Low (pooled over T-subsets)
    Values: Z-scores
    """
    if len(results_df) == 0:
        return

    motifs = results_df['motif'].unique()

    # Build heatmap data
    rows = []
    for m in motifs:
        row = {'motif': m}
        m_data = results_df[results_df['motif'] == m]

        # T-subset columns (pooled over PD1)
        for ts in ['Naive', 'CM', 'EM']:
            ts_data = m_data[m_data['phenotype'].str.startswith(ts)]
            if len(ts_data) > 0:
                row[ts] = ts_data['zscore'].mean()
            else:
                row[ts] = np.nan

        # PD1_Low column (pooled over T-subsets)
        low_data = m_data[m_data['phenotype'].str.contains('Low')]
        if len(low_data) > 0:
            row['PD1_Low'] = low_data['zscore'].mean()
        else:
            row['PD1_Low'] = np.nan

        rows.append(row)

    data = pd.DataFrame(rows).set_index('motif')
    data = data.dropna(how='all')

    if len(data) == 0:
        return

    # Reorder columns
    col_order = [c for c in ['Naive', 'CM', 'EM', 'PD1_Low'] if c in data.columns]
    data = data[col_order]

    # Sort by max absolute Z-score
    data = data.loc[data.abs().max(axis=1).sort_values(ascending=False).index]

    # Auto-size figure
    n_rows, n_cols = data.shape
    figsize = (max(4, n_cols * 0.8 + 2), max(4, n_rows * 0.3 + 1))

    # Symmetric colormap
    abs_max = np.nanmax(np.abs(data.values))
    vmin, vmax = -abs_max, abs_max

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        data.values,
        aspect='auto',
        cmap='RdBu_r',
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest',
    )

    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.index, fontsize=8)

    ax.set_xlabel('Pooled Phenotype')
    ax.set_ylabel('Motif')
    ax.set_title("Z-scores by Pooled Phenotype")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Z-score')

    fig.tight_layout()
    fig.savefig(plots_path / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

    data.to_csv(plots_path.parent / "heatmap_pooled_zscores.csv")
    print(f"    Created pooled heatmap")


def compute_phenotype_nbglm_contrasts(results_df):
    """Compute T-subset and PD1 contrasts from phenotype NB-GLM results."""
    contrasts = []

    motifs = results_df['motif'].unique()

    for m in motifs:
        m_data = results_df[results_df['motif'] == m]

        # T-subset contrasts (pooled over PD1)
        for ts1, ts2 in TSUBSET_CONTRASTS:
            ts1_data = m_data[m_data['phenotype'].str.startswith(ts1)]
            ts2_data = m_data[m_data['phenotype'].str.startswith(ts2)]

            if len(ts1_data) == 0 or len(ts2_data) == 0:
                continue

            effect1 = ts1_data['coef'].mean()
            effect2 = ts2_data['coef'].mean()
            se1 = np.sqrt((ts1_data['se'] ** 2).sum()) / len(ts1_data)
            se2 = np.sqrt((ts2_data['se'] ** 2).sum()) / len(ts2_data)

            diff = effect1 - effect2
            se_diff = np.sqrt(se1**2 + se2**2)
            z = diff / se_diff if se_diff > 0 else 0
            pval = 2 * (1 - stats.norm.cdf(abs(z)))

            contrasts.append({
                'motif': m,
                'contrast': f"{ts1}_vs_{ts2}",
                'logFC': diff,
                'se': se_diff,
                'zscore': z,
                'pvalue': pval,
            })

        # PD1 contrast (Low vs High, pooled over T-subsets)
        high_data = m_data[m_data['phenotype'].str.contains('High')]
        low_data = m_data[m_data['phenotype'].str.contains('Low')]

        if len(high_data) > 0 and len(low_data) > 0:
            effect_high = high_data['coef'].mean()
            effect_low = low_data['coef'].mean()
            se_high = np.sqrt((high_data['se'] ** 2).sum()) / len(high_data)
            se_low = np.sqrt((low_data['se'] ** 2).sum()) / len(low_data)

            diff = effect_low - effect_high
            se_diff = np.sqrt(se_high**2 + se_low**2)
            z = diff / se_diff if se_diff > 0 else 0
            pval = 2 * (1 - stats.norm.cdf(abs(z)))

            contrasts.append({
                'motif': m,
                'contrast': 'PD1Low_vs_PD1High',
                'logFC': diff,
                'se': se_diff,
                'zscore': z,
                'pvalue': pval,
            })

    df = pd.DataFrame(contrasts)
    if len(df) > 0:
        df['qvalue'] = apply_fdr_correction(df, 'pvalue', 'contrast')
    return df


def create_contrast_volcano_plots(contrast_results, plots_path, method_name):
    """Create volcano plots for contrasts with consistent styling.

    Uses same style as cs.volcano_plot:
    - q_thresh=0.10, lfc_thresh=0.10
    - Blue/red coloring by effect direction
    - Top labels for most significant
    """
    if len(contrast_results) == 0:
        return

    q_thresh = 0.10
    lfc_thresh = 0.10

    for contrast_name in contrast_results['contrast'].unique():
        df = contrast_results[contrast_results['contrast'] == contrast_name].copy()
        if len(df) == 0:
            continue

        df['neglog10_q'] = -np.log10(df['qvalue'].clip(lower=1e-300))

        fig, ax = plt.subplots(figsize=(8, 6))

        # Color by effect direction (consistent with cs.volcano_plot)
        colors = np.where(df['logFC'] >= 0, "#3182bd", "#e34a33")  # blue/red
        ax.scatter(df['logFC'], df['neglog10_q'], c=colors, s=12, alpha=0.7)

        # Threshold lines
        ax.axhline(-np.log10(q_thresh), color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.axvline(+lfc_thresh, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.axvline(-lfc_thresh, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

        # Top 10 labels
        sig_df = df.nsmallest(10, 'qvalue')
        for _, row in sig_df.iterrows():
            ax.text(row['logFC'], row['neglog10_q'], row['motif'], fontsize=8)

        ax.set_xlabel(r"$\log_2$ fold change")
        ax.set_ylabel(r"$-\log_{10}$ q-value")

        # Parse contrast name for title
        if '_vs_' in contrast_name:
            parts = contrast_name.split('_vs_')
            title = f"{parts[0]} vs {parts[1]}"
        else:
            title = contrast_name.replace('_', ' ')
        ax.set_title(title)

        plt.tight_layout()

        # Consistent file naming
        if 'PD1' in contrast_name:
            fig.savefig(plots_path / f"volcano_{contrast_name}_pooledTsubset.png", dpi=200)
        else:
            parts = contrast_name.split('_vs_')
            if len(parts) == 2:
                fig.savefig(plots_path / f"volcano_{parts[0]}_vs_{parts[1]}_pooledPD1.png", dpi=200)
            else:
                fig.savefig(plots_path / f"volcano_{contrast_name}.png", dpi=200)
        plt.close(fig)
        print(f"    Created volcano: {title}")


# =============================================================================
# Analysis Method 4: Dirichlet-Multinomial
# =============================================================================

def run_dm_analysis(counts_wide, cand, smeta, elm_df, all_elms, output_path):
    """
    Run Dirichlet-Multinomial analysis using costim_screen.

    Uses empirical Bayes spike-and-slab PIPs for significance detection.

    Scientific note: DM models COMPOSITIONAL effects (how ELMs shift the
    relative proportions of phenotypes). If an ELM affects proliferation
    uniformly across phenotypes, DM won't detect it (but NB-GLM will).
    Finding few/no DM hits while NB-GLM finds many suggests ELMs primarily
    affect proliferation rather than differentiation.

    CCR random effects are included to absorb batch/replicate variance.
    """
    print("\n=== Running Dirichlet-Multinomial Analysis (PIP mode) ===")
    output_path.mkdir(parents=True, exist_ok=True)

    # PIP threshold and CI level
    # Note: simulation uses pip_threshold=0.90 by default, but that's very stringent
    # We use 0.5 for a more balanced false positive/negative tradeoff
    PIP_THRESHOLD = 0.5
    CI_LEVEL = 0.90

    try:
        # Prepare DM data using the provided function
        print("  Preparing DM data...")
        smeta_with_id = smeta.reset_index()
        dm_data = cs.prepare_dm_data(
            counts_wide,
            smeta_with_id,
            cand,
            exp_cond="CAR:Raji",
            min_total=10,
            min_elm_freq=MIN_ELM_FREQ,
        )
        print(f"  DM data: {dm_data.n_obs} observations, {dm_data.n_elms} ELMs, {dm_data.n_ccrs} CCRs")

        # Fit DM model WITH CCR random effects (absorbs batch variance)
        print("  Fitting DM model (with CCR effects, finite-diff Hessian)...")
        dm_result = cs.fit_dm_model(
            dm_data,
            verbose=False,
            use_finite_diff_hessian=True,
            include_ccr_effects=True,  # Include CCR effects to absorb batch variance
        )
        print(f"  Alpha: {dm_result.alpha:.2f}")
        print(f"  Converged: {dm_result.converged}")

        # Get effects and SEs for each ELM-phenotype combination
        print("  Computing PIPs via empirical Bayes spike-and-slab...")
        effect_records = []
        se_records = []
        for elm in dm_data.elm_names:
            for pheno in dm_data.phenotype_names:
                effect, se = dm_result.get_elm_effect(elm, pheno)
                effect_records.append({"ELM": elm, "phenotype": pheno, "effect": effect})
                se_records.append({"ELM": elm, "phenotype": pheno, "se": se})

        effect_df = (
            pd.DataFrame(effect_records)
            .pivot(index="ELM", columns="phenotype", values="effect")
            .reindex(index=dm_data.elm_names, columns=dm_data.phenotype_names)
        )
        se_df = (
            pd.DataFrame(se_records)
            .pivot(index="ELM", columns="phenotype", values="se")
            .reindex(index=dm_data.elm_names, columns=dm_data.phenotype_names)
        )

        # Compute PIPs using empirical Bayes spike-and-slab
        beta_flat = effect_df.values.flatten()
        se_flat = se_df.values.flatten()
        pip_flat, pi0_hat, tau2_hat = cs.eb_spike_slab_pip(beta_flat, se_flat)
        print(f"  EB spike-slab: pi0={pi0_hat:.3f}, tau2={tau2_hat:.4f}")

        # Diagnostic: show effect/SE distributions
        valid_mask = np.isfinite(beta_flat) & np.isfinite(se_flat) & (se_flat > 0)
        if valid_mask.sum() > 0:
            valid_effects = beta_flat[valid_mask]
            valid_ses = se_flat[valid_mask]
            valid_pips = pip_flat[valid_mask]
            z_scores = valid_effects / valid_ses
            print(f"  Effects: mean={np.mean(valid_effects):.4f}, std={np.std(valid_effects):.4f}, "
                  f"max|effect|={np.max(np.abs(valid_effects)):.4f}")
            print(f"  SEs:     mean={np.mean(valid_ses):.4f}, median={np.median(valid_ses):.4f}, "
                  f"min={np.min(valid_ses):.4f}")
            print(f"  |Z|:     mean={np.mean(np.abs(z_scores)):.3f}, max={np.max(np.abs(z_scores)):.3f}, "
                  f"n(|Z|>1.96)={np.sum(np.abs(z_scores) > 1.96)}")
            print(f"  PIPs:    max={np.max(valid_pips):.4f}, n(PIP>0.5)={np.sum(valid_pips > 0.5)}, "
                  f"n(PIP>0.3)={np.sum(valid_pips > 0.3)}")

        pip_df = pd.DataFrame(
            pip_flat.reshape(effect_df.shape),
            index=effect_df.index,
            columns=effect_df.columns,
        )

        # CI exclusion rule
        z_crit = stats.norm.ppf(0.5 + CI_LEVEL / 2.0)
        ci_lo = effect_df - z_crit * se_df
        ci_hi = effect_df + z_crit * se_df
        ci_excludes_zero = (ci_lo > 0) | (ci_hi < 0)

        # Significance: PIP > threshold AND CI excludes zero
        sig_df = (pip_df > PIP_THRESHOLD) & ci_excludes_zero

        # Exclude reference phenotype
        ref = dm_result.reference_phenotype
        if ref in sig_df.columns:
            sig_df.loc[:, ref] = False

        # Save per-phenotype results with PIPs
        pheno_results = []
        for elm in dm_data.elm_names:
            for pheno in dm_data.phenotype_names:
                if pheno == ref:
                    continue
                pheno_results.append({
                    'ELM': elm,
                    'phenotype': pheno,
                    'effect': effect_df.loc[elm, pheno],
                    'se': se_df.loc[elm, pheno],
                    'pip': pip_df.loc[elm, pheno],
                    'significant': sig_df.loc[elm, pheno],
                })
        pheno_df = pd.DataFrame(pheno_results)
        pheno_df.to_csv(output_path / "dm_phenotype_results.csv", index=False)

        n_sig_pheno = pheno_df['significant'].sum()
        n_ci_excl = ci_excludes_zero.sum().sum()  # total where CI excludes 0
        n_pip_high = (pip_df > PIP_THRESHOLD).sum().sum()  # total where PIP > threshold
        print(f"  Per-phenotype: {len(pheno_df)} tests")
        print(f"    - CI excludes 0: {n_ci_excl}")
        print(f"    - PIP > {PIP_THRESHOLD}: {n_pip_high}")
        print(f"    - Both (significant): {n_sig_pheno}")

        # Now compute contrasts using the Wald approach but report PIPs as well
        contrast_tables = []
        for ts_p, ts_q in TSUBSET_CONTRASTS:
            tab = compute_dm_tsubset_contrast(dm_result, dm_data, ts_p, ts_q)
            if len(tab) > 0:
                tab["contrast"] = f"{ts_p}_vs_{ts_q}"
                contrast_tables.append(tab)

        tab_pd1 = compute_dm_pd1_contrast(dm_result, dm_data)
        if len(tab_pd1) > 0:
            tab_pd1["contrast"] = "PD1Low_vs_PD1High"
            contrast_tables.append(tab_pd1)

        if not contrast_tables:
            print("  No contrasts computed")
            return pheno_df

        results_df = pd.concat(contrast_tables, ignore_index=True)

        # Compute PIPs for contrasts too
        contrast_effects = results_df['effect'].values
        contrast_ses = results_df['se'].values
        contrast_pips, pi0_c, tau2_c = cs.eb_spike_slab_pip(contrast_effects, contrast_ses)
        results_df['pip'] = contrast_pips

        # Diagnostic for contrasts
        valid_c = np.isfinite(contrast_effects) & np.isfinite(contrast_ses) & (contrast_ses > 0)
        if valid_c.sum() > 0:
            z_c = contrast_effects[valid_c] / contrast_ses[valid_c]
            print(f"  Contrast effects: max|eff|={np.max(np.abs(contrast_effects[valid_c])):.4f}, "
                  f"max|Z|={np.max(np.abs(z_c)):.3f}")
            print(f"  Contrast PIPs: pi0={pi0_c:.3f}, tau2={tau2_c:.6f}, max_PIP={np.max(contrast_pips[valid_c]):.4f}")

        # CI exclusion for contrasts
        ci_lo_c = results_df['effect'] - z_crit * results_df['se']
        ci_hi_c = results_df['effect'] + z_crit * results_df['se']
        results_df['ci_excludes_zero'] = (ci_lo_c > 0) | (ci_hi_c < 0)
        results_df['significant_pip'] = (results_df['pip'] > PIP_THRESHOLD) & results_df['ci_excludes_zero']

        # Also keep traditional q-value for comparison
        results_df['qvalue'] = apply_fdr_correction(results_df, 'pvalue', 'contrast')

        # Save results
        results_df.to_csv(output_path / "dm_contrasts.csv", index=False)

        n_sig_pip = results_df['significant_pip'].sum()
        n_sig_bh = (results_df['qvalue'] < 0.10).sum()
        n_nominal = (results_df['pvalue'] < 0.05).sum()
        print(f"  Contrasts: {len(results_df)} tests")
        print(f"    - Nominal p<0.05: {n_nominal}")
        print(f"    - BH q<0.10: {n_sig_bh}")
        print(f"    - PIP>0.5 & CI excl 0: {n_sig_pip}")

        # Create plots
        plots_path = output_path / 'plots'
        plots_path.mkdir(exist_ok=True)

        pip_thresh = PIP_THRESHOLD
        effect_thresh = 0.10

        # Create volcano plots using PIP (y-axis = PIP, not -log10 q)
        for contrast_name in results_df['contrast'].unique():
            df = results_df[results_df['contrast'] == contrast_name].copy()
            if len(df) == 0:
                continue

            fig, ax = plt.subplots(figsize=(8, 6))

            # Color by significance (PIP > threshold AND CI excludes 0)
            colors = np.where(df['significant_pip'], "#2ca02c", "#7f7f7f")  # green/gray
            ax.scatter(df['effect'], df['pip'], c=colors, s=40, alpha=0.7)

            # Threshold lines
            ax.axhline(pip_thresh, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'PIP={pip_thresh}')
            ax.axvline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

            # Label significant points
            sig_df = df[df['significant_pip']]
            for _, row in sig_df.iterrows():
                ax.text(row['effect'], row['pip'] + 0.02, row['ELM'], fontsize=8, ha='center')

            # Also label top PIPs even if not significant
            if len(sig_df) == 0:
                top_df = df.nlargest(5, 'pip')
                for _, row in top_df.iterrows():
                    ax.text(row['effect'], row['pip'] + 0.02, row['ELM'], fontsize=8, ha='center')

            ax.set_xlabel("Log-Odds Effect")
            ax.set_ylabel("Posterior Inclusion Probability (PIP)")
            ax.set_ylim(-0.05, 1.05)
            ax.legend(loc='upper right')

            # Parse contrast name for title
            if '_vs_' in contrast_name:
                parts = contrast_name.split('_vs_')
                title = f"{parts[0]} vs {parts[1]}"
            else:
                title = contrast_name.replace('_', ' ')
            ax.set_title(f"{title} (DM PIP)")

            plt.tight_layout()

            # Consistent file naming
            if 'PD1' in contrast_name:
                fig.savefig(plots_path / f"volcano_{contrast_name}_pooledTsubset.png", dpi=200)
            else:
                parts = contrast_name.split('_vs_')
                if len(parts) == 2:
                    fig.savefig(plots_path / f"volcano_{parts[0]}_vs_{parts[1]}_pooledPD1.png", dpi=200)
                else:
                    fig.savefig(plots_path / f"volcano_{contrast_name}.png", dpi=200)
            plt.close(fig)
            print(f"    Created volcano: {title}")

        # Create pooled heatmap using PIPs
        create_dm_pooled_heatmap_pip(pip_df, effect_df, sig_df, plots_path, output_path)

        # Assemble joint figure with all volcanos and heatmap
        try:
            cs.assemble_dm_main_figure(
                plots_path,
                out_png="figure_dm_volcanos_plus_heatmap.png",
                out_pdf=None,
            )
            print(f"    Created joint figure")
        except Exception as e:
            print(f"  Warning: joint figure failed: {e}")

        print(f"  Saved results to {output_path}")
        return results_df

    except Exception as e:
        print(f"  Error in DM analysis: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def compute_dm_tsubset_contrast(dm_result, dm_data, ts_p, ts_q):
    """Compute T-subset contrast from DM model (pooled over PD1)."""
    results = []
    for elm_name in dm_data.elm_names:
        try:
            effect, se, pval = dm_result.contrast_tsubset(elm_name, ts_p, ts_q)
            z = effect / se if se > 0 else 0
            results.append({
                'ELM': elm_name,
                'effect': effect,
                'se': se,
                'zscore': z,
                'pvalue': pval,
            })
        except Exception:
            continue
    return pd.DataFrame(results)


def compute_dm_pd1_contrast(dm_result, dm_data):
    """Compute PD1 contrast from DM model (pooled over T-subsets)."""
    results = []
    for elm_name in dm_data.elm_names:
        try:
            effect, se, pval = dm_result.contrast_pd1(elm_name)
            z = effect / se if se > 0 else 0
            results.append({
                'ELM': elm_name,
                'effect': effect,
                'se': se,
                'zscore': z,
                'pvalue': pval,
            })
        except Exception:
            continue
    return pd.DataFrame(results)


def create_dm_pooled_heatmap(results_df, plots_path):
    """Create pooled heatmap matching cs.pooled_coef_heatmap style.

    Columns: Naive, CM, EM (from T-subset contrasts), PD1_Low (from PD1 contrast)
    Values: Z-scores
    """
    if len(results_df) == 0:
        return

    all_elms = results_df['ELM'].unique()

    # Build heatmap data
    rows = []
    for elm in all_elms:
        row = {'motif': elm}
        elm_data = results_df[results_df['ELM'] == elm]

        # T-subset columns: extract from contrast results
        # EM_vs_CM gives EM relative effect, Naive_vs_CM gives Naive relative effect
        # We can't directly get individual T-subset effects from contrasts,
        # so we'll use the contrast Z-scores as proxies

        # Get EM effect (from EM_vs_CM contrast, positive = EM > CM)
        em_data = elm_data[elm_data['contrast'] == 'EM_vs_CM']
        row['EM'] = em_data['zscore'].iloc[0] if len(em_data) > 0 else np.nan

        # Get Naive effect (from Naive_vs_CM contrast, positive = Naive > CM)
        naive_data = elm_data[elm_data['contrast'] == 'Naive_vs_CM']
        row['Naive'] = naive_data['zscore'].iloc[0] if len(naive_data) > 0 else np.nan

        # CM is reference (z=0)
        row['CM'] = 0.0

        # PD1_Low (from PD1Low_vs_PD1High contrast, positive = Low > High)
        pd1_data = elm_data[elm_data['contrast'] == 'PD1Low_vs_PD1High']
        row['PD1_Low'] = pd1_data['zscore'].iloc[0] if len(pd1_data) > 0 else np.nan

        rows.append(row)

    data = pd.DataFrame(rows).set_index('motif')
    data = data.dropna(how='all')

    if len(data) == 0:
        return

    # Reorder columns
    col_order = [c for c in ['Naive', 'CM', 'EM', 'PD1_Low'] if c in data.columns]
    data = data[col_order]

    # Sort by max absolute Z-score
    data = data.loc[data.abs().max(axis=1).sort_values(ascending=False).index]

    # Auto-size figure
    n_rows, n_cols = data.shape
    figsize = (max(4, n_cols * 0.8 + 2), max(4, n_rows * 0.3 + 1))

    # Symmetric colormap
    abs_max = np.nanmax(np.abs(data.values))
    vmin, vmax = -abs_max, abs_max

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        data.values,
        aspect='auto',
        cmap='RdBu_r',
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest',
    )

    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.index, fontsize=8)

    ax.set_xlabel('Pooled Phenotype')
    ax.set_ylabel('Motif')
    ax.set_title("DM Z-scores by Pooled Phenotype")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Z-score')

    fig.tight_layout()
    fig.savefig(plots_path / "heatmap_pooled.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    data.to_csv(plots_path.parent / "heatmap_pooled_zscores.csv")
    print(f"    Created pooled heatmap")


def create_dm_pooled_heatmap_pip(pip_df, effect_df, sig_df, plots_path, output_path):
    """Create pooled heatmap showing PIPs for DM analysis.

    This version shows PIP values instead of z-scores, with significant
    cells (PIP > threshold AND CI excludes 0) marked.

    Parameters
    ----------
    pip_df : pd.DataFrame
        PIPs indexed by ELM, columns are phenotypes
    effect_df : pd.DataFrame
        Effects indexed by ELM, columns are phenotypes
    sig_df : pd.DataFrame
        Boolean significance indexed by ELM, columns are phenotypes
    plots_path : Path
        Path for saving plots
    output_path : Path
        Path for saving CSV data
    """
    if len(pip_df) == 0:
        return

    # Build pooled data:
    # Columns: Naive, CM, EM (pooled over PD1), PD1_Low (pooled over T-subsets)
    # For each T-subset, average the High and Low PIPs
    # For PD1_Low, average the Low across T-subsets

    all_elms = pip_df.index.tolist()

    rows = []
    for elm in all_elms:
        row = {'motif': elm}

        # T-subset columns (average over PD1 High/Low)
        for ts in ['Naive', 'CM', 'EM']:
            ts_cols = [c for c in pip_df.columns if c.startswith(ts)]
            if ts_cols:
                row[ts] = pip_df.loc[elm, ts_cols].mean()
            else:
                row[ts] = np.nan

        # PD1_Low column (average Low phenotypes over T-subsets)
        low_cols = [c for c in pip_df.columns if c.endswith('Low')]
        if low_cols:
            row['PD1_Low'] = pip_df.loc[elm, low_cols].mean()
        else:
            row['PD1_Low'] = np.nan

        rows.append(row)

    data = pd.DataFrame(rows).set_index('motif')
    data = data.dropna(how='all')

    if len(data) == 0:
        return

    # Reorder columns
    col_order = [c for c in ['Naive', 'CM', 'EM', 'PD1_Low'] if c in data.columns]
    data = data[col_order]

    # Sort by max PIP
    data = data.loc[data.max(axis=1).sort_values(ascending=False).index]

    # Auto-size figure
    n_rows, n_cols = data.shape
    figsize = (max(4, n_cols * 0.8 + 2), max(4, n_rows * 0.3 + 1))

    # PIP-appropriate colormap (0 to 1)
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        data.values,
        aspect='auto',
        cmap='YlOrRd',  # Yellow-Orange-Red for PIPs (0=low, 1=high)
        vmin=0,
        vmax=1,
        interpolation='nearest',
    )

    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.index, fontsize=8)

    ax.set_xlabel('Pooled Phenotype')
    ax.set_ylabel('Motif')
    ax.set_title("DM Posterior Inclusion Probability (PIP) by Pooled Phenotype")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('PIP')

    fig.tight_layout()
    fig.savefig(plots_path / "heatmap_pooled.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    data.to_csv(output_path / "heatmap_pooled_pips.csv")
    print(f"    Created pooled PIP heatmap")

    # Also create an effect heatmap (for reference)
    effect_rows = []
    for elm in all_elms:
        row = {'motif': elm}

        for ts in ['Naive', 'CM', 'EM']:
            ts_cols = [c for c in effect_df.columns if c.startswith(ts)]
            if ts_cols:
                row[ts] = effect_df.loc[elm, ts_cols].mean()
            else:
                row[ts] = np.nan

        low_cols = [c for c in effect_df.columns if c.endswith('Low')]
        if low_cols:
            row['PD1_Low'] = effect_df.loc[elm, low_cols].mean()
        else:
            row['PD1_Low'] = np.nan

        effect_rows.append(row)

    effect_data = pd.DataFrame(effect_rows).set_index('motif')
    effect_data = effect_data.dropna(how='all')

    if len(effect_data) > 0:
        col_order_e = [c for c in ['Naive', 'CM', 'EM', 'PD1_Low'] if c in effect_data.columns]
        effect_data = effect_data[col_order_e]
        effect_data = effect_data.loc[effect_data.abs().max(axis=1).sort_values(ascending=False).index]

        fig2, ax2 = plt.subplots(figsize=figsize)

        abs_max = np.nanmax(np.abs(effect_data.values))
        vmin, vmax = -abs_max, abs_max

        im2 = ax2.imshow(
            effect_data.values,
            aspect='auto',
            cmap='RdBu_r',
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest',
        )

        ax2.set_xticks(range(len(col_order_e)))
        ax2.set_xticklabels(col_order_e, rotation=45, ha='right', fontsize=9)
        ax2.set_yticks(range(len(effect_data)))
        ax2.set_yticklabels(effect_data.index, fontsize=8)

        ax2.set_xlabel('Pooled Phenotype')
        ax2.set_ylabel('Motif')
        ax2.set_title("DM Effects by Pooled Phenotype")

        cbar2 = fig2.colorbar(im2, ax=ax2, shrink=0.8)
        cbar2.set_label('Effect (log-odds)')

        fig2.tight_layout()
        fig2.savefig(plots_path / "heatmap_effects.png", dpi=300, bbox_inches='tight')
        plt.close(fig2)

        effect_data.to_csv(output_path / "heatmap_pooled_effects.csv")
        print(f"    Created pooled effects heatmap")


# =============================================================================
# Summary
# =============================================================================

def count_significant_results(results_df, q_thresh=0.10):
    """Count significant results in a DataFrame.

    Returns (total_tests, n_significant)

    For DM results with PIP-based significance, uses 'significant_pip' column.
    For other methods, uses q-value threshold.
    """
    if results_df is None or len(results_df) == 0:
        return 0, 0

    n_total = len(results_df)

    # Check for PIP-based significance (DM)
    if 'significant_pip' in results_df.columns:
        n_sig = results_df['significant_pip'].sum()
        return n_total, n_sig

    # Check for per-phenotype significance column (DM phenotype results)
    if 'significant' in results_df.columns:
        n_sig = results_df['significant'].sum()
        return n_total, n_sig

    # Try different column names for q-value
    qcol = None
    for qname in ['qvalue', 'FDR', 'q']:
        if qname in results_df.columns:
            qcol = qname
            break

    n_sig = (results_df[qcol] < q_thresh).sum() if qcol else 0
    return n_total, n_sig


def print_significance_summary(results_dict, subset_name):
    """Print summary of significant results for each method."""
    print(f"\n{'='*60}")
    print(f"SIGNIFICANCE SUMMARY: {subset_name.upper()}")
    print(f"(q<0.10 for MW/NB-GLM; PIP>0.5 & CI excl. 0 for DM)")
    print(f"{'='*60}")
    print(f"{'Method':<25} {'Total Tests':>12} {'Significant':>12}")
    print(f"{'-'*50}")

    for method, results in results_dict.items():
        n_total, n_sig = count_significant_results(results)
        # Add note if DM uses PIP
        suffix = " (PIP)" if method == 'dm' else ""
        print(f"{method:<25} {n_total:>12} {n_sig:>12}{suffix}")

    print(f"{'='*60}\n")


def create_summary_excel(results_dict, output_path):
    """Create summary Excel file comparing all methods."""
    print("\n=== Creating Summary Excel ===")

    summary_path = output_path / "summary_of_methods.xlsx"

    with pd.ExcelWriter(summary_path, engine='openpyxl') as writer:
        # Overview sheet
        overview = []
        for method, results in results_dict.items():
            n_total, n_sig = count_significant_results(results)
            if n_total > 0:
                overview.append({
                    'Method': method,
                    'Total_tests': n_total,
                    'Significant_FDR10': n_sig,
                })

        if overview:
            pd.DataFrame(overview).to_excel(writer, sheet_name='Overview', index=False)

        # Individual method sheets
        for method, results in results_dict.items():
            if results is not None and len(results) > 0:
                sheet_name = method[:31]  # Excel sheet name limit
                results.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"  Saved summary to {summary_path}")


# =============================================================================
# Main
# =============================================================================

def run_all_analyses(subset_type: str):
    """Run all 5 analysis methods for a given subset."""
    print(f"\n{'='*60}")
    print(f"RUNNING ANALYSES FOR: {subset_type.upper()}")
    print(f"{'='*60}")

    # Set output path based on subset type
    output_base = RESULTS_PATH / subset_type

    output_base.mkdir(parents=True, exist_ok=True)

    # Load and filter data
    counts_raji, counts_wide, cand, smeta, elm_df, all_elms = load_and_filter_data(subset_type)

    if len(counts_raji) == 0:
        print("No data after filtering. Skipping.")
        return

    # Run all analyses
    results = {}

    import gc

    # 1. Mann-Whitney
    mw_results, mw_contrasts, mw_pd1 = run_mw_analysis(
        counts_raji, cand, elm_df, all_elms,
        output_base / 'mw_analysis'
    )
    results['mw'] = mw_results if len(mw_results) > 0 else mw_contrasts
    plt.close('all'); gc.collect()

    # 2. Joint NB-GLM
    results['joint_nbglm'] = run_joint_nbglm_analysis(
        counts_wide, cand, smeta, elm_df, all_elms,
        output_base / 'joint_nbglm_analysis'
    )
    plt.close('all'); gc.collect()

    # 3. Phenotype NB-GLM (pooled approach)
    results['phenotype_nbglm'] = run_phenotype_nbglm_analysis(
        counts_wide, cand, smeta, elm_df, all_elms,
        output_base / 'phenotype_nbglm_analysis'
    )
    plt.close('all'); gc.collect()

    # 4. Dirichlet-Multinomial
    results['dm'] = run_dm_analysis(
        counts_wide, cand, smeta, elm_df, all_elms,
        output_base / 'dm_analysis'
    )
    plt.close('all'); gc.collect()

    # Create summary
    create_summary_excel(results, output_base)

    # Print significance summary
    print_significance_summary(results, subset_type)

    print(f"\n{'='*60}")
    print(f"COMPLETED: {subset_type.upper()}")
    print(f"Results in: {output_base}")
    print(f"{'='*60}")


def main():
    """Main entry point."""
    print("="*60)
    print("COMPREHENSIVE CAR-T COSTIM SCREEN ANALYSIS")
    print("="*60)

    # Define all subset combinations
    # Naming convention:
    #   non_gpcr   = all non-GPCR (no ICD conditioning)
    #   non_gpcr_1 = non-GPCR with ICD ID = 1
    #   gpcr_1     = GPCR with ICD ID = 1
    subsets = [
        # Non-GPCR: any ICD ID
        'non_gpcr',
        # Non-GPCR conditioned on ICD ID
        'non_gpcr_1',
        'non_gpcr_2',
        'non_gpcr_3_4',  # Combined ICD ID 3 and 4 (too few CCDs separately)
        # GPCR: any ICD ID
        'gpcr',
        # GPCR conditioned on ICD ID
        'gpcr_1',
        'gpcr_2',
        'gpcr_3',
        'gpcr_4',
    ]

    import gc
    # Run for each subset
    for subset_type in subsets:
        try:
            run_all_analyses(subset_type)
        except Exception as e:
            print(f"\n*** ERROR running {subset_type}: {e} ***\n")
            import traceback
            traceback.print_exc()
        finally:
            # Close all matplotlib figures and run garbage collection
            plt.close('all')
            gc.collect()
            print(f"  [Memory cleanup after {subset_type}]")

    print("\n" + "="*60)
    print("ALL ANALYSES COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
