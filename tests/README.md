# Tests and Analysis Scripts

This folder contains test scripts and analysis tools for the Dirichlet-Multinomial model.

## Quick Start

Run the comprehensive evaluation in this order:

```bash
# 1. Run comprehensive simulation study (may take 30-60 minutes)
python tests/comprehensive_simulation_study.py

# 2. Generate comparison plots
python tests/plot_simulation_results.py

# 3. Apply to real data (may take 5-10 minutes)
python tests/apply_dm_to_real_data.py
```

## Test Scripts

### Core Tests

**`test_dm_smoke.py`**
- Quick smoke test for the DM model
- Uses a simple synthetic dataset with known effects
- Validates that the model runs and has reasonable power

**`test_all_methods.py`**
- Compares all methods (DM, Joint NB-GLM, Mann-Whitney, Per-Phenotype GLM)
- Single scenario comparison

### Diagnostic Tests

**`test_hessian_comparison.py`**
- Compares L-BFGS-B inverse Hessian vs finite-difference Hessian
- Demonstrates that L-BFGS-B underestimates precision by 16-19×

**`test_dm_calling_modes.py`**
- Tests PIP-based calling vs BH FDR calling

**`test_dm_power.py`**
- Tests DM power with larger sample sizes

## Comprehensive Analysis

### `comprehensive_simulation_study.py`

Systematic evaluation across multiple scenarios:
- Sample sizes: 100, 200, 400 CCDs
- Effect sizes: 0.3, 0.6, 0.9
- 5 replicates per scenario = 45 total runs

**Runtime:** ~30-60 minutes

### `plot_simulation_results.py`

Generates comparison plots (requires running simulation study first)

## Real Data Analysis

### `apply_dm_to_real_data.py`

Applies DM model to real CAR:Raji data with accurate inference.

**Runtime:** ~5-10 minutes