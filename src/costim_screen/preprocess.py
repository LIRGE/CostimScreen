from __future__ import annotations

import numpy as np
import pandas as pd


def add_library_size(
        df_long: pd.DataFrame
) -> pd.DataFrame:
    lib = df_long.groupby("sample_id")["count"].sum().rename("lib_size")
    out = df_long.merge(lib, on="sample_id", how="left")
    out["offset"] = np.log(out["lib_size"].clip(lower=1).astype(float))
    return out


def counts_to_long(
        counts_wide: pd.DataFrame,
        id_col: str = "CandidateID"
) -> pd.DataFrame:
    """
    counts_wide: index=CandidateID, columns=sample_id
    returns long df: CandidateID, sample_id, count
    """
    df = counts_wide.stack().reset_index()
    df.columns = [id_col, "sample_id", "count"]
    return df


def filter_domains_by_total_counts(
        counts_wide: pd.DataFrame,
        min_total: int = 50
) -> pd.DataFrame:
    totals = counts_wide.sum(axis=1)
    keep = totals[totals >= min_total].index
    return counts_wide.loc[keep]


def make_block_id(
        sample_meta: pd.DataFrame
) -> pd.Series:
    donor_col = "Donor" if "Donor" in sample_meta.columns else ("Donor;" if "Donor;" in sample_meta.columns else "donor")
    cond_col  = "ExpCond" if "ExpCond" in sample_meta.columns else ("ExpCOnd" if "ExpCOnd" in sample_meta.columns else "condition")
    rep_col   = "Replicate" if "Replicate" in sample_meta.columns else "rep"
    return (
        sample_meta[donor_col].astype(str)
        + "_"
        + sample_meta[cond_col].astype(str)
        + "_r"
        + sample_meta[rep_col].astype(str)
    )
