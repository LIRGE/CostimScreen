"""
Data preprocessing utilities.

This module provides functions for transforming count data from wide to long
format, computing library sizes, filtering low-abundance candidates, and
creating CCR identifiers for mixed-effects modeling.

Functions
---------
counts_to_long
    Convert wide count matrix to long format.
add_library_size
    Compute library sizes and log offsets.
filter_domains_by_total_counts
    Filter candidates by minimum total counts.
make_ccr_id
    Create CCR identifiers for cluster-robust standard errors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_library_size(df_long: pd.DataFrame) -> pd.DataFrame:
    """Compute library sizes and log offsets for each sample.

    Adds ``lib_size`` (total counts per sample) and ``offset`` (log library size)
    columns to the long-format DataFrame. The offset is used in GLM fitting
    to normalize for sequencing depth.

    Parameters
    ----------
    df_long : pd.DataFrame
        Long-format DataFrame with columns ``sample_id`` and ``count``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added columns:
        - lib_size: total counts for each sample
        - offset: log(lib_size), clipped to minimum of 0 (lib_size >= 1)

    Examples
    --------
    >>> df_long = pd.DataFrame({
    ...     "sample_id": ["S1", "S1", "S2", "S2"],
    ...     "count": [100, 200, 150, 250]
    ... })
    >>> df_with_lib = add_library_size(df_long)
    >>> df_with_lib["lib_size"].unique()
    array([300, 400])
    """
    lib = df_long.groupby("sample_id")["count"].sum().rename("lib_size")
    out = df_long.merge(lib, on="sample_id", how="left")
    out["offset"] = np.log(out["lib_size"].clip(lower=1).astype(float))
    return out


def counts_to_long(
    counts_wide: pd.DataFrame,
    id_col: str = "CandidateID",
) -> pd.DataFrame:
    """Convert a wide count matrix to long format.

    Transforms a matrix with candidates as rows and samples as columns
    into a long-format DataFrame suitable for GLM fitting.

    Parameters
    ----------
    counts_wide : pd.DataFrame
        Wide-format count matrix with CandidateID as index and sample_id
        as column names.
    id_col : str, default "CandidateID"
        Name for the candidate ID column in the output.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        - {id_col}: candidate identifier
        - sample_id: sample identifier
        - count: read count

    Examples
    --------
    >>> counts_wide = pd.DataFrame(
    ...     {"S1": [10, 20], "S2": [15, 25]},
    ...     index=["gene1", "gene2"]
    ... )
    >>> counts_long = counts_to_long(counts_wide)
    >>> len(counts_long)
    4
    """
    df = counts_wide.stack().reset_index()
    df.columns = [id_col, "sample_id", "count"]
    return df


def filter_domains_by_total_counts(
    counts_wide: pd.DataFrame,
    min_total: int = 50,
) -> pd.DataFrame:
    """Filter candidates by minimum total counts across all samples.

    Removes candidates (rows) that have fewer than ``min_total`` counts
    summed across all samples. This helps remove noise from low-abundance
    candidates.

    Parameters
    ----------
    counts_wide : pd.DataFrame
        Wide-format count matrix with candidates as rows.
    min_total : int, default 50
        Minimum total count threshold. Candidates with fewer counts are removed.

    Returns
    -------
    pd.DataFrame
        Filtered count matrix containing only candidates meeting the threshold.

    Examples
    --------
    >>> counts = pd.DataFrame(
    ...     {"S1": [10, 100], "S2": [5, 200]},
    ...     index=["low", "high"]
    ... )
    >>> filtered = filter_domains_by_total_counts(counts, min_total=50)
    >>> list(filtered.index)
    ['high']
    """
    totals = counts_wide.sum(axis=1)
    keep = totals[totals >= min_total].index
    return counts_wide.loc[keep]


def make_ccr_id(sample_meta: pd.DataFrame) -> pd.Series:
    """Create CCR identifiers for cluster-robust standard errors.

    Constructs a CCR ID string combining donor, experimental condition,
    and replicate. This is used for clustering observations in the GLM
    to account for correlated errors within CCRs.

    Parameters
    ----------
    sample_meta : pd.DataFrame
        Sample metadata DataFrame with columns for donor, experimental
        condition, and replicate. Accepts various column name variations.

    Returns
    -------
    pd.Series
        CCR ID strings in format "Donor_ExpCond_rReplicate".

    Notes
    -----
    Handles common column name variations:
    - Donor: "Donor", "Donor;", "donor"
    - Condition: "ExpCond", "ExpCOnd", "condition"
    - Replicate: "Replicate", "rep"

    Examples
    --------
    >>> smeta = pd.DataFrame({
    ...     "Donor": [1, 1, 2],
    ...     "ExpCond": ["Raji", "Raji", "K562"],
    ...     "Replicate": [1, 2, 1]
    ... })
    >>> make_ccr_id(smeta)
    0    1_Raji_r1
    1    1_Raji_r2
    2    2_K562_r1
    dtype: object
    """
    donor_col = (
        "Donor"
        if "Donor" in sample_meta.columns
        else ("Donor;" if "Donor;" in sample_meta.columns else "donor")
    )
    cond_col = (
        "ExpCond"
        if "ExpCond" in sample_meta.columns
        else ("ExpCOnd" if "ExpCOnd" in sample_meta.columns else "condition")
    )
    rep_col = "Replicate" if "Replicate" in sample_meta.columns else "rep"
    return (
        sample_meta[donor_col].astype(str)
        + "_"
        + sample_meta[cond_col].astype(str)
        + "_r"
        + sample_meta[rep_col].astype(str)
    )