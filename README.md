# costim-screen

A Python toolkit for analyzing pooled CAR-T costimulatory domain screens, from raw FASTQ files to statistical analysis using negative binomial GLMs, Dirichlet-multinomial models, and non-parametric tests.

📖 **Documentation**: [https://costimscreen.readthedocs.io/en/latest/](https://costimscreen.readthedocs.io/en/latest/)

## Overview

This toolkit analyzes how different **Costimulatory Cytoplasmic Domains (CCDs)** affect CAR-T cell phenotype distributions. Each CCD construct contains one or more **Eukaryotic Linear Motifs (ELMs)**—short functional sequence patterns that mediate protein interactions and signaling.

`costim-screen` provides a complete two-stage pipeline:

**Stage 1: Primary Analysis (`primary_analysis` module)**
- Parse targeted sequencing FASTQ files
- Match reads to known CCD sequences (with fuzzy matching)
- Generate count matrices for downstream analysis

**Stage 2: Statistical Analysis (`costim_screen` module)**
- Load and preprocess count matrices
- Build ELM category design matrices
- Run multiple analysis methods (NB-GLM, DM, Mann-Whitney)
- Compute contrasts and generate volcano plots

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
   git clone https://github.com/username/CostimScreen.git
   cd CostimScreen
   ```

3. **Install in editable mode:**

   ```bash
   pip install -e .
   ```

   This installs both `primary_analysis` and `costim_screen` packages along with all dependencies.

4. **Install optional dependencies:**

   ```bash
   # For development (testing, linting)
   pip install -e ".[dev]"

   # For documentation
   pip install -e ".[docs]"
   ```

## Quick Start

### Step 1: Generate Count Matrix from FASTQ Files

```python
from primary_analysis import ReadCostimFASTQ

# Configure reader
reader = ReadCostimFASTQ(
    fastq_path='/path/to/fastq',
    feature_metadata_fn='/path/to/costim_metadata.xlsx',
    sample_metadata_fn='/path/to/sample_metadata.xlsx',  # Optional
    fastq_encoding='PT_EC_RP_SS',
    n_char_features=30,
    read_error_threshold=2,
)

# Parse all FASTQ files (parallelized)
reader.read()

# Save count matrix and metadata
reader.serialize('/path/to/results')
reader.print_summary()
```

Or use the command-line interface:

```bash
read-costim \
    -f /path/to/fastq \
    --feature_metadata_fn /path/to/costim_metadata.xlsx \
    --results_path /path/to/results \
    --n_char_truncate 30
```

### Step 2: Run Statistical Analysis

```python
import costim_screen as cs
from pathlib import Path

# Load data
data_path = Path("data")
counts = cs.load_counts_matrix(data_path / "merged_counts.xlsx")
smeta = cs.load_sample_metadata(data_path / "sample_metadata.xlsx")
cand = cs.load_candidate_metadata(data_path / "candidate_metadata.xlsx")

# Preprocess
counts = cs.filter_domains_by_total_counts(counts, min_total=50)

# Build ELM design matrix
X_elm = cs.build_elm_design(cand, elms_col="ELMCategory", min_freq=0.01)

# Convert to long format for modeling
df = cs.counts_to_long(counts, id_col="CandidateID")
df = df.merge(smeta.reset_index(), on="sample_id", how="left")
df = cs.add_library_size(df)

# Fit joint NB-GLM model
formula = cs.build_joint_formula(list(X_elm.columns))
fit = cs.fit_nb_glm_iter_alpha(df, formula=formula, offset_col="offset")

# Generate volcano plot
cs.volcano_tsubset_pooled_pd1(
    fit, list(X_elm.columns),
    tsubset_p="EM", tsubset_q="CM",
    q_thresh=0.10, outpath="results/volcano_EM_vs_CM.png"
)
```

## Workflow

```
┌──────────────────────────────────────────────────────────────────────┐
│                         PRIMARY ANALYSIS                              │
│  FASTQ files → primary_analysis.ReadCostimFASTQ → merged_counts.xlsx │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       STATISTICAL ANALYSIS                            │
│  merged_counts.xlsx → costim_screen methods → volcano plots, tables  │
└──────────────────────────────────────────────────────────────────────┘
```

## Scripts (Batch Analysis)

The `scripts/` directory contains ready-to-run analysis scripts:

| Script | Description |
|--------|-------------|
| `run_comprehensive_analysis.py` | **Main analysis script** - Runs all 5 methods (4 ELM-level for individual motif effects + 1 CCD-level for combinatorial effects) on all subset combinations (GPCR/non-GPCR × ICD groupings) |
| `run_model_comparison.py` | Benchmarking script comparing method performance via simulations |
| `generate_simulation_figures.py` | Generates publication-ready figures from simulation results |

**Running the comprehensive analysis:**

```bash
cd CostimScreen
python scripts/run_comprehensive_analysis.py
```

Results are written to `results/{subset}/` with separate directories for each analysis method.

## Notebooks (Interactive Analysis)

The `notebooks/` directory contains Jupyter notebooks for interactive exploration:

| Notebook | Description |
|----------|-------------|
| `00_fastq_to_counts.ipynb` | **Primary analysis** - Parse FASTQ files and generate count matrix |
| `01_refit_joint_model.ipynb` | Step-by-step joint NB-GLM fitting with diagnostic plots |
| `02_dirichlet_multinomial_analysis.ipynb` | DM model fitting with PIP-based significance detection |
| `03_mann_whitney_analysis.ipynb` | Non-parametric analysis using Pearson residuals |

To run notebooks:

```bash
conda activate costim_screen_env
jupyter notebook notebooks/
```

## Data Format

### Feature Metadata (`costim_metadata.xlsx`)

Required columns for `primary_analysis`:

| ID | Costim |
|-----|--------|
| GENE1-1 | ATCGATCG... |
| GENE2-1 | GCTAGCTA... |

### Count Matrix (`merged_counts.xlsx`)

Output from `primary_analysis`, input for `costim_screen`:

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

| CandidateID | ELMCategory        | is_gpcr | ICD_ID |
|-------------|-------------------|---------|--------|
| GENE1-1     | MOD_SUMO;LIG_SH3  | 0       | 1      |

## Project Structure

```
CostimScreen/
├── src/
│   ├── primary_analysis/       # Stage 1: FASTQ parsing
│   │   ├── __init__.py
│   │   ├── base.py             # ReadFASTQ base class
│   │   ├── costim.py           # ReadCostimFASTQ implementation
│   │   ├── constants.py        # Flanking sequences
│   │   ├── sequencing_read.py  # Read parsing + fuzzy matching
│   │   └── read_db.py          # Count accumulation
│   │
│   └── costim_screen/          # Stage 2: Statistical analysis
│       ├── __init__.py
│       ├── io.py               # Data loading
│       ├── preprocess.py       # Data transformations
│       ├── features.py         # ELM design matrices
│       ├── model.py            # NB-GLM fitting
│       ├── contrasts.py        # Wald contrasts
│       ├── dirichlet_multinomial.py
│       ├── stats.py            # FDR correction, PIPs
│       ├── plots.py            # Visualization
│       └── pooled.py           # Pooled analyses
│
├── scripts/                    # Batch analysis scripts
├── notebooks/                  # Interactive Jupyter notebooks
├── data/                       # Input data (not in repo)
├── results/                    # Output results (not in repo)
├── docs/                       # Sphinx documentation
└── tests/                      # Unit tests
```

## Analysis Methods

### ELM-Level Analysis (Individual Motif Effects)

Tests the impact of individual ELMs by comparing all CCDs containing a given motif versus those without it. This approach asks: "Does the presence of motif X affect CAR-T behavior?" ELM-level analysis may detect effects on overall proliferation but typically does not reveal phenotype-specific effects, since the signal is averaged across diverse CCD backgrounds.

1. **Mann-Whitney (mw)**: Non-parametric comparison of Pearson residuals between CCDs with/without each ELM
2. **Joint NB-GLM (joint_nbglm)**: Single model with phenotype × ELM interactions
3. **Phenotype NB-GLM (phenotype_nbglm)**: Separate models per pooled T-subset with iterative dispersion
4. **Dirichlet-Multinomial (dm)**: Compositional model for phenotype proportions with empirical Bayes PIPs

### CCD-Level Analysis (Combinatorial Effects)

Tests individual CCD constructs, each representing a specific *combination* of ELMs. This approach asks: "Does this particular construct shift the phenotype distribution?" CCD-level analysis reveals phenotype-specific effects that emerge from specific motif combinations—effects that are diluted when averaging across all CCDs sharing a single ELM.

5. **CCD DM (ccd_dm)**: Per-construct G-test comparing each CCD's phenotype distribution to the leave-one-out population mean

## License

MIT License

## Citation

If you use this software in your research, please cite:

> [Citation will be added upon publication]

## Authors

- Stefan Cordes (stefan@alumni.princeton.edu)
- Conor Kelley (CSK70@georgetown.edu)
