"""Apply Dirichlet-Multinomial model to real CAR:Raji data.

This script runs the DM analysis on real data with the accurate
finite-difference Hessian computation.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import time

import costim_screen as cs

print("=" * 80)
print("APPLY DIRICHLET-MULTINOMIAL MODEL TO REAL DATA")
print("=" * 80)

# ============================================================================
# Setup paths
# ============================================================================
# Get the repository root (parent of tests/)
if Path('.').resolve().name == 'tests':
    base_path = Path('.').resolve().parent
else:
    base_path = Path('.').resolve()

data_path = base_path / 'data'
results_path = base_path / 'results' / 'dm_analysis'
results_path.mkdir(parents=True, exist_ok=True)

print(f"\nData path: {data_path}")
print(f"Results path: {results_path}")

# ============================================================================
# Load data
# ============================================================================
print("\n" + "=" * 80)
print("Loading data...")
print("=" * 80)

counts_wide = cs.load_counts_matrix(
    data_path / 'merged_counts.xlsx',
    candidate_id_col="CandidateID"
)
smeta = cs.load_sample_metadata(data_path / 'sample_metadata.xlsx')
cand = cs.load_candidate_metadata(data_path / 'candidate_metadata.xlsx')

# Align IDs
common_candidates = counts_wide.index.intersection(cand.index)
common_samples = counts_wide.columns.intersection(smeta.index)

counts_wide = counts_wide.loc[common_candidates, common_samples]
cand = cand.loc[common_candidates]
smeta = smeta.loc[common_samples]

print(f"Counts: {counts_wide.shape[0]} CCDs × {counts_wide.shape[1]} samples")

# Filter low-count CCDs
counts_wide = cs.filter_domains_by_total_counts(counts_wide, min_total=50)
cand = cand.loc[counts_wide.index]
print(f"After filtering: {counts_wide.shape[0]} CCDs")

# ============================================================================
# Build ELM design matrix
# ============================================================================
print("\n" + "=" * 80)
print("Building ELM design matrix...")
print("=" * 80)

X_elm = cs.build_elm_category_design(
    cand.reset_index(),
    candidate_id_col="CandidateID",
    elm_col="ELMCategory",
    include_quadratic=False,
    min_freq=0.0,
)

safe_cols, mapping = cs.make_patsy_safe_columns(list(X_elm.columns), prefix="ELM_")
X_elm = X_elm.rename(columns=mapping)
elm_names = list(X_elm.columns)

# Remove GPCRs
non_gpcr_ids = cand.index[cand["is_gpcr"] == 0]
counts_wide = counts_wide.loc[non_gpcr_ids]
X_elm = X_elm.loc[non_gpcr_ids]
cand = cand.loc[non_gpcr_ids]

print(f"Number of ELMs: {len(elm_names)}")
print(f"After removing GPCRs: {counts_wide.shape[0]} CCDs")

# ============================================================================
# Prepare data for DM model
# ============================================================================
print("\n" + "=" * 80)
print("Preparing data for DM model...")
print("=" * 80)

# Sample-level derived variables
smeta = smeta.copy()
smeta.index.name = "sample_id"
smeta["Tsubset"] = smeta["Tsubset"].replace({"Naïve": "Naive"})
smeta["phenotype"] = smeta["Tsubset"].astype(str) + "_" + smeta["PD1Status"].astype(str)
smeta["CCR"] = cs.make_ccr_id(smeta)

# Long format
df = cs.counts_to_long(counts_wide, id_col="CandidateID")
df = df.merge(smeta.reset_index(), on="sample_id", how="left")
df = df.merge(X_elm.reset_index().rename(columns={"index": "CandidateID"}), on="CandidateID", how="left")

# Filter to CAR:Raji
df_raji = df[df["ExpCond"] == "CAR:Raji"].copy()
print(f"CAR:Raji observations: {len(df_raji)}")

# Prepare DM data
phenotype_order = ["Naive_High", "Naive_Low", "CM_High", "CM_Low", "EM_High", "EM_Low"]

counts_pivot = df_raji.pivot_table(
    index=["CandidateID", "CCR"],
    columns="phenotype",
    values="count",
    aggfunc="sum",
    fill_value=0
)[phenotype_order].reset_index()

counts_pivot["total"] = counts_pivot[phenotype_order].sum(axis=1)
counts_pivot = counts_pivot[counts_pivot["total"] >= 100].copy()

counts_pivot = counts_pivot.merge(
    X_elm.reset_index().rename(columns={"index": "CandidateID"}),
    on="CandidateID",
    how="left"
)

ccr_to_idx = {ccr: i for i, ccr in enumerate(counts_pivot["CCR"].unique())}
counts_pivot["CCR_idx"] = counts_pivot["CCR"].map(ccr_to_idx)

dm_data = cs.DirichletMultinomialData(
    counts=counts_pivot[phenotype_order].values,
    elm_matrix=counts_pivot[elm_names].values,
    ccr_ids=counts_pivot["CCR_idx"].values.astype(int),
    ccd_ids=counts_pivot["CandidateID"].values,
    totals=counts_pivot["total"].values,
    elm_names=elm_names,
    phenotype_names=phenotype_order,
)

print(f"\nDM data structure:")
print(f"  Observations: {dm_data.n_obs}")
print(f"  ELMs: {dm_data.n_elms}")
print(f"  Phenotypes: {dm_data.n_phenotypes}")
print(f"  CCRs: {dm_data.n_ccrs}")

# ============================================================================
# Fit DM model with accurate Hessian
# ============================================================================
print("\n" + "=" * 80)
print("Fitting Dirichlet-Multinomial model...")
print("  Using finite-difference Hessian for accurate inference")
print("  This may take ~5-10 minutes for real data")
print("=" * 80)

start_time = time.time()
dm_result = cs.fit_dm_model(dm_data, verbose=True, use_finite_diff_hessian=True)
runtime = time.time() - start_time

print(f"\n{'=' * 80}")
print("Model fit complete!")
print(f"{'=' * 80}")
print(f"  Converged: {dm_result.converged}")
print(f"  Alpha: {dm_result.alpha:.2f}")
print(f"  Log-likelihood: {dm_result.log_likelihood:.1f}")
print(f"  Runtime: {runtime:.1f}s ({runtime/60:.1f} min)")

# Save model parameters
model_params = {
    "alpha": dm_result.alpha,
    "log_likelihood": dm_result.log_likelihood,
    "converged": dm_result.converged,
    "runtime_seconds": runtime,
    "use_finite_diff_hessian": True,
}
pd.Series(model_params).to_csv(results_path / "dm_model_params.csv")

# ============================================================================
# Extract results
# ============================================================================
print("\n" + "=" * 80)
print("Extracting results...")
print("=" * 80)

results_list = []
for elm_name in dm_data.elm_names:
    pvals = dm_result.get_elm_pvalues(elm_name)

    for phenotype in dm_data.phenotype_names:
        if phenotype == "Naive_Low":
            results_list.append({
                "ELM": elm_name,
                "Phenotype": phenotype,
                "Effect": 0.0,
                "SE": 0.0,
                "Pvalue": np.nan,
                "FDR": np.nan,
            })
        else:
            effect, se = dm_result.get_elm_effect(elm_name, phenotype)
            results_list.append({
                "ELM": elm_name,
                "Phenotype": phenotype,
                "Effect": effect,
                "SE": se,
                "Pvalue": pvals[phenotype],
                "FDR": np.nan,
            })

dm_results_df = pd.DataFrame(results_list)

# Apply FDR correction (exclude reference)
non_ref_mask = dm_results_df["Phenotype"] != "Naive_Low"
dm_results_df.loc[non_ref_mask, "FDR"] = cs.bh_fdr(dm_results_df.loc[non_ref_mask, "Pvalue"].fillna(1))

# Add log2FC
dm_results_df["log2FC"] = dm_results_df["Effect"] / np.log(2)

# Save results
output_file = results_path / "dm_all_effects.csv"
dm_results_df.to_csv(output_file, index=False)
print(f"✓ Saved full results to: {output_file}")

# Show significant results
sig_results = dm_results_df[
    (dm_results_df["Phenotype"] != "Naive_Low") &
    (dm_results_df["FDR"] < 0.10)
].sort_values("Pvalue")

print(f"\n{'=' * 80}")
print(f"Significant ELM-phenotype associations (FDR < 0.10): {len(sig_results)}")
print(f"{'=' * 80}")

if len(sig_results) > 0:
    print("\nTop 20 associations:")
    print(sig_results.head(20)[["ELM", "Phenotype", "log2FC", "Pvalue", "FDR"]].to_string(index=False))
else:
    print("\nNo significant associations found at FDR < 0.10")
    print("\nTop 10 by nominal p-value:")
    top10 = dm_results_df[dm_results_df["Phenotype"] != "Naive_Low"].nsmallest(10, "Pvalue")
    print(top10[["ELM", "Phenotype", "log2FC", "Pvalue", "FDR"]].to_string(index=False))

print(f"\n{'=' * 80}")
print("ANALYSIS COMPLETE")
print(f"{'=' * 80}")
print(f"\nResults saved to: {results_path}")
print(f"  - dm_model_params.csv")
print(f"  - dm_all_effects.csv")
print(f"\nNext steps:")
print(f"  1. Review top associations")
print(f"  2. Generate volcano plots and heatmaps using the notebook")
print(f"  3. Compare with original NB-GLM results")