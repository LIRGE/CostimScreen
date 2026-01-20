"""
costim-screen: Analysis toolkit for pooled CAR-T costimulatory domain screens.

This package provides tools for analyzing pooled screening data using negative
binomial generalized linear models (GLMs) with motif-phenotype interactions.

Modules
-------
io
    Data loading functions for count matrices and metadata.
preprocess
    Data transformation utilities (long format, library size, filtering).
features
    ELM design matrix construction and feature engineering.
metadata_setup
    Metadata parsing and candidate information merging.
model
    Negative binomial GLM fitting with iterative dispersion estimation.
contrasts
    Wald contrast computations for motif effects.
stats
    Statistical utilities including FDR correction.
plots
    Visualization functions (volcano plots).
pooled
    Pooled contrast analyses across phenotype groups.
diagnostics
    Model diagnostics and dispersion estimation.

Example
-------
>>> import costim_screen as cs
>>> counts = cs.load_counts_matrix("data/merged_counts.xlsx")
>>> smeta = cs.load_sample_metadata("data/sample_metadata.xlsx")
>>> cand = cs.load_candidate_metadata("data/candidate_metadata.xlsx")
>>> # ... preprocessing and model fitting ...
"""

__version__ = "0.1.0"
__author__ = "Stefan Cordes"
__email__ = "stefan@alumni.Princeton.edu"

# contrasts
from .contrasts import (
    coef_name_for_motif_phenotype,
    motif_diff_between_phenotypes,
    wald_contrast,
)

# diagnostics
from .diagnostics import (
    estimate_alpha_nb2_moments,
    per_sample_dispersion,
    poisson_overdispersion_test,
    zero_fraction,
)

# features
from .features import (
    build_elm_design,
    make_patsy_safe_columns,
    split_elm_list,
)

# io
from .io import (
    Paths,
    load_candidate_metadata,
    load_counts_matrix,
    load_sample_metadata,
    parse_samples_from_columns,
    write_sample_metadata_template,
)

# metadata_setup
from .metadata_setup import (
    CandidateMergeSpec,
    SampleParseSpec,
    build_elm_category_design,
    merge_candidate_metadata,
    sample_metadata_from_counts_xlsx,
    split_sample_id,
)

# model
from .model import (
    FitResult,
    build_joint_formula,
    fit_nb_glm_iter_alpha,
)

# plots
from .plots import (
    volcano_plot,
)

# pooled
from .pooled import (
    motif_contrast_table_pd1_pooled_tsubset,
    motif_contrast_table_tsubset_pooled_pd1,
    motif_diff_between_pd1_pooled_tsubset,
    motif_diff_between_tsubsets_pooled_pd1,
    volcano_pd1_pooled_tsubset,
    volcano_tsubset_pooled_pd1,
)

# preprocess
from .preprocess import (
    add_library_size,
    counts_to_long,
    filter_domains_by_total_counts,
    make_block_id,
)

# stats
from .stats import (
    bh_fdr,
    motif_contrast_table,
)

__all__ = [
    # contrasts
    "coef_name_for_motif_phenotype",
    "motif_diff_between_phenotypes",
    "wald_contrast",
    # diagnostics
    "estimate_alpha_nb2_moments",
    "per_sample_dispersion",
    "poisson_overdispersion_test",
    "zero_fraction",
    # features
    "build_elm_design",
    "make_patsy_safe_columns",
    "split_elm_list",
    # io
    "Paths",
    "load_candidate_metadata",
    "load_counts_matrix",
    "load_sample_metadata",
    "parse_samples_from_columns",
    "write_sample_metadata_template",
    # metadata_setup
    "CandidateMergeSpec",
    "SampleParseSpec",
    "build_elm_category_design",
    "merge_candidate_metadata",
    "sample_metadata_from_counts_xlsx",
    "split_sample_id",
    # model
    "FitResult",
    "build_joint_formula",
    "fit_nb_glm_iter_alpha",
    # plots
    "volcano_plot",
    # pooled
    "motif_contrast_table_pd1_pooled_tsubset",
    "motif_contrast_table_tsubset_pooled_pd1",
    "motif_diff_between_pd1_pooled_tsubset",
    "motif_diff_between_tsubsets_pooled_pd1",
    "volcano_pd1_pooled_tsubset",
    "volcano_tsubset_pooled_pd1",
    # preprocess
    "add_library_size",
    "counts_to_long",
    "filter_domains_by_total_counts",
    "make_block_id",
    # stats
    "bh_fdr",
    "motif_contrast_table",
]