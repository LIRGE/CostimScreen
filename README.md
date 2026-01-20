# costim-screen

A Python toolkit for analyzing pooled CAR-T costimulatory domain screens using negative binomial GLMs with motif-phenotype interactions.

## Overview

`costim-screen` provides a complete pipeline for:

- Loading and preprocessing count matrices from pooled screens
- Building ELM (Eukaryotic Linear Motif) category design matrices
- Fitting negative binomial GLMs with iterative dispersion estimation
- Computing Wald contrasts for motif effects between phenotypes
- Generating volcano plots with FDR-corrected significance thresholds

## Installation

### Prerequisites

- Python 3.10 or higher
- conda (recommended) or pip

### From Source (Development)

1. **Create a conda environment:**

   ```bash
   conda create --name costim_screen_env jupyter python=3.12
   conda activate costim_screen_env
   ```

2. **Clone or download the repository:**

   ```bash
   # If you have the repository as a zip file:
   unzip CostimScreen.zip
   cd CostimScreen

   # Or clone from GitHub (when available):
   # git clone https://github.com/username/CostimScreen.git
   # cd CostimScreen
   ```

3. **Install in editable mode:**

   ```bash
   pip install -e .
   ```

   This installs the package along with all dependencies (numpy, pandas, statsmodels, matplotlib, etc.).

### From PyPI (Coming Soon)

```bash
pip install costim-screen
```

## Quick Start

```python
import costim_screen as cs
from pathlib import Path

# Define paths
data_path = Path("data")
results_path = Path("results")

# Load data
counts = cs.load_counts_matrix(data_path / "merged_counts.xlsx")
smeta = cs.load_sample_metadata(data_path / "sample_metadata.xlsx")
cand = cs.load_candidate_metadata(data_path / "candidate_metadata.xlsx")

# Preprocess
counts = cs.filter_domains_by_total_counts(counts, min_total=50)

# Build ELM design matrix
X_elm = cs.build_elm_category_design(
    cand.reset_index(),
    candidate_id_col="CandidateID",
    elm_col="ELMCategory",
    min_freq=0.025,
)

# Make column names patsy-safe
safe_cols, mapping = cs.make_patsy_safe_columns(list(X_elm.columns), prefix="ELM_")
X_elm = X_elm.rename(columns=mapping)

# Prepare long-format data
smeta["phenotype"] = smeta["Tsubset"] + "_" + smeta["PD1Status"]
smeta["block"] = cs.make_block_id(smeta)

df = cs.counts_to_long(counts, id_col="CandidateID")
df = df.merge(smeta.reset_index(), on="sample_id", how="left")
df = cs.add_library_size(df)
df = df.merge(X_elm.reset_index().rename(columns={"index": "CandidateID"}),
              on="CandidateID", how="left")

# Fit model
motif_cols = list(X_elm.columns)
formula = cs.build_joint_formula(motif_cols)
fit = cs.fit_nb_glm_iter_alpha(df, formula=formula, offset_col="offset", cluster_col="block")

# Compute contrasts and generate volcano plot
result = cs.motif_contrast_table(fit, motifs=motif_cols, p="EM_High", q="CM_High")
cs.volcano_plot(result, q_thresh=0.10, lfc_thresh=1.0,
                outpath=results_path / "volcano.png")
```

## Data Format

### Count Matrix (`merged_counts.xlsx`)

| CandidateID | Sample1 | Sample2 | ... |
|-------------|---------|---------|-----|
| GENE1-1     | 1234    | 5678    | ... |
| GENE2-1     | 2345    | 6789    | ... |

### Sample Metadata (`sample_metadata.xlsx`)

| sample_id | Donor | ExpCond | Tsubset | PD1Status | Replicate |
|-----------|-------|---------|---------|-----------|-----------|
| Sample1   | 1     | CAR:Raji| CM      | High      | 1         |
| Sample2   | 1     | CAR:Raji| EM      | Low       | 1         |

### Candidate Metadata (`candidate_metadata.xlsx`)

| CandidateID | ELMCategory        | ICD Num | Num ICD |
|-------------|-------------------|---------|---------|
| GENE1-1     | MOD_SUMO;LIG_SH3  | 1       | 2       |

## API Reference

See the [API Documentation](docs/_build/html/index.html) for complete details.

### Core Modules

- **`costim_screen.io`**: Data loading functions
- **`costim_screen.preprocess`**: Data transformation utilities
- **`costim_screen.features`**: ELM design matrix construction
- **`costim_screen.model`**: Negative binomial GLM fitting
- **`costim_screen.contrasts`**: Wald contrast computations
- **`costim_screen.stats`**: Statistical utilities (FDR correction)
- **`costim_screen.plots`**: Visualization functions
- **`costim_screen.pooled`**: Pooled contrast analyses

## License

MIT License

## Citation

If you use this software in your research, please cite:

> [Citation will be added upon publication]

## Authors

- Stefan Cordes (stefan@alumni.Princeton.edu)