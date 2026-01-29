"""
Feature engineering for ELM motif design matrices.

This module provides functions for constructing one-hot encoded design matrices
from ELM (Eukaryotic Linear Motif) category annotations. These design matrices
are used as predictors in the negative binomial GLM.

Functions
---------
build_elm_design
    Build a one-hot design matrix from ELM annotations.
make_patsy_safe_columns
    Convert column names to patsy-safe identifiers.
split_elm_list
    Parse semicolon/comma/pipe-separated ELM strings.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Callable, Optional, Tuple

import pandas as pd


def build_elm_design(
    domains_df: pd.DataFrame,
    elms_col: str = "elms",
    min_freq: float = 0.01,
    collapse_fn: Optional[Callable[[str], str]] = None,
) -> pd.DataFrame:
    """Build a one-hot design matrix from ELM annotations.

    Creates a binary design matrix where each column represents an ELM
    category and each row represents a candidate domain. Values are 1
    if the candidate contains that ELM, 0 otherwise.

    Parameters
    ----------
    domains_df : pd.DataFrame
        DataFrame with domain/candidate information, indexed by domain ID.
    elms_col : str, default "elms"
        Name of the column containing ELM category strings.
    min_freq : float, default 0.01
        Minimum fraction of domains containing a feature to keep it.
        Features present in fewer domains are dropped.
    collapse_fn : callable or None, default None
        Optional function to transform ELM names (e.g., for grouping
        related motifs).

    Returns
    -------
    pd.DataFrame
        Binary design matrix with domain IDs as index and ELM features
        as columns. Values are 0 or 1.

    Examples
    --------
    >>> domains = pd.DataFrame(
    ...     {"elms": ["MOD_SUMO;LIG_SH3", "MOD_SUMO", "LIG_SH3"]},
    ...     index=["d1", "d2", "d3"]
    ... )
    >>> X = build_elm_design(domains, elms_col="elms", min_freq=0.3)
    >>> X.columns.tolist()
    ['LIG_SH3', 'MOD_SUMO']
    """
    domain_ids = domains_df.index.astype(str)
    elm_lists = []
    counter: Counter[str] = Counter()

    for did in domain_ids:
        elms = split_elm_list(
            domains_df.loc[did, elms_col] if elms_col in domains_df.columns else ""
        )
        if collapse_fn is not None:
            elms = [collapse_fn(e) for e in elms]
        elms = list(dict.fromkeys(elms))  # unique, preserve order
        elm_lists.append((did, elms))
        counter.update(elms)

    n = len(domain_ids)
    keep = {k for k, v in counter.items() if (v / max(n, 1)) >= min_freq and k != ""}

    X = pd.DataFrame(0, index=domain_ids, columns=sorted(keep), dtype=int)
    for did, elms in elm_lists:
        for e in elms:
            if e in keep:
                X.loc[did, e] = 1

    return X


def make_patsy_safe_columns(
    cols: list[str],
    prefix: str = "F_",
) -> Tuple[list[str], dict[str, str]]:
    """Convert column names to patsy-safe identifiers.

    Transforms column names to be valid Python identifiers that work with
    the patsy formula interface. Replaces special characters, ensures names
    don't start with digits, and guarantees uniqueness.

    Parameters
    ----------
    cols : list of str
        Original column names.
    prefix : str, default ``"F_"``
        Prefix to add to names that start with a digit.

    Returns
    -------
    safe_cols : list of str
        Transformed column names that are valid identifiers.
    mapping : dict
        Mapping from original names to safe names.
    """
    safe = []
    mapping = {}
    used: set[str] = set()

    for c in cols:
        s = re.sub(r"[^0-9a-zA-Z_]+", "_", str(c)).strip("_")
        if s == "":
            s = "EMPTY"
        if re.match(r"^\d", s):
            s = prefix + s

        base = s
        k = 1
        while s in used:
            k += 1
            s = f"{base}_{k}"

        used.add(s)
        safe.append(s)
        mapping[str(c)] = s

    return safe, mapping


def split_elm_list(s: str) -> list[str]:
    """Parse a delimited string of ELM categories.

    Splits a string containing ELM category names separated by semicolons,
    commas, or pipes into a list of individual category names.

    Parameters
    ----------
    s : str
        Delimited string of ELM categories (e.g., "MOD_SUMO;LIG_SH3").

    Returns
    -------
    list of str
        List of individual ELM category names. Empty list if input is
        empty or NA.

    Examples
    --------
    >>> split_elm_list("MOD_SUMO;LIG_SH3;MOD_PKA")
    ['MOD_SUMO', 'LIG_SH3', 'MOD_PKA']
    >>> split_elm_list("A|B,C")
    ['A', 'B', 'C']
    >>> split_elm_list("")
    []
    """
    if pd.isna(s) or str(s).strip() == "":
        return []
    raw = str(s).replace("|", ";").replace(",", ";")
    parts = [p.strip() for p in raw.split(";")]
    # Strip surrounding quotes (single or double) from each element
    cleaned = []
    for p in parts:
        if p:
            # Remove surrounding quotes
            p = p.strip("'\"")
            if p:
                cleaned.append(p)
    return cleaned