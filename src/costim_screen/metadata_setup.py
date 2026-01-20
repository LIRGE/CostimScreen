"""
Metadata parsing and candidate information merging.

This module provides functions and specifications for parsing sample metadata
from column names and merging candidate information from multiple Excel files.

Functions
---------
build_elm_category_design
    Build a one-hot design matrix for ELM categories with optional interactions.
merge_candidate_metadata
    Merge ELM annotations with topology information.
sample_metadata_from_counts_xlsx
    Parse sample metadata from count matrix column headers.
split_sample_id
    Split a sample ID string into metadata fields.

Classes
-------
CandidateMergeSpec
    Specification for column names when merging candidate metadata.
SampleParseSpec
    Specification for parsing sample ID strings.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .features import split_elm_list


@dataclass(frozen=True)
class CandidateMergeSpec:
    """Specification for merging candidate metadata from multiple files.

    Defines the expected input column names and desired output column names
    for the merge operation.
    """

    #: Input column name for candidate ID.
    id_col: str = "ID"
    #: Input column name for ELM vocabulary.
    elm_vocab_col: str = "elm vocab"
    #: Input column name for ICD ID.
    icd_id_col: str = "ICD ID"
    #: Input column name for ICD multiplicity.
    icd_mult_col: str = "Gene ICD Multiplicity"
    #: Output column name for candidate ID.
    out_id_col: str = "CandidateID"
    #: Output column name for ELM category.
    out_elm_col: str = "ELMCategory"
    #: Output column name for ICD number.
    out_icd_id_col: str = "ICD Num"
    #: Output column name for ICD count.
    out_icd_mult_col: str = "Num ICD"


@dataclass(frozen=True)
class SampleParseSpec:
    """Specification for parsing sample ID strings.

    Defines how to split sample ID strings into metadata fields.
    """

    #: Separator character for splitting sample IDs.
    sep: str = "_"
    #: Expected field names in order (Donor, ExpCond, Tsubset, PD1Status, Replicate).
    fields: tuple[str, ...] = ("Donor", "ExpCond", "Tsubset", "PD1Status", "Replicate")


def build_elm_category_design(
    candidates_df: pd.DataFrame,
    candidate_id_col: str = "CandidateID",
    elm_col: str = "ELMCategory",
    min_freq: float = 0.01,
    include_quadratic: bool = False,
    min_interaction_freq: float | None = None,
    max_interactions: int | None = 500,
    interaction_sep: str = "__x__",
) -> pd.DataFrame:
    """Build a one-hot design matrix for ELM categories.

    Creates a binary design matrix from ELM category annotations, with optional
    pairwise interaction terms for co-occurring motifs.

    Parameters
    ----------
    candidates_df : pd.DataFrame
        DataFrame with candidate IDs and ELM category strings.
    candidate_id_col : str, default "CandidateID"
        Column name for candidate IDs (used as index).
    elm_col : str, default "ELMCategory"
        Column containing ELM category strings (semicolon/comma/pipe separated).
    min_freq : float, default 0.01
        Minimum fraction of candidates containing a feature to keep it.
    include_quadratic : bool, default False
        If True, add pairwise interaction terms for co-occurring motifs.
    min_interaction_freq : float or None, default None
        Minimum frequency for interaction terms. Defaults to min_freq.
    max_interactions : int or None, default 500
        Maximum number of interaction terms to keep (most frequent).
    interaction_sep : str, default "__x__"
        Separator string for interaction column names.

    Returns
    -------
    pd.DataFrame
        One-hot encoded design matrix with candidate_id as index and
        ELM features (plus interactions if requested) as columns.

    Examples
    --------
    >>> X_elm = build_elm_category_design(
    ...     cand.reset_index(),
    ...     candidate_id_col="CandidateID",
    ...     elm_col="ELMCategory",
    ...     min_freq=0.025,
    ...     include_quadratic=False
    ... )
    """
    from collections import Counter

    candidate_ids = candidates_df[candidate_id_col].astype(str)
    elm_lists = []
    counter: Counter[str] = Counter()

    for idx, cid in enumerate(candidate_ids):
        elms = split_elm_list(candidates_df.iloc[idx][elm_col])
        elms = list(dict.fromkeys(elms))  # unique, preserve order
        elm_lists.append((cid, elms))
        counter.update(elms)

    n = len(candidate_ids)
    keep = {k for k, v in counter.items() if (v / max(n, 1)) >= min_freq and k != ""}

    X = pd.DataFrame(0, index=candidate_ids, columns=sorted(keep), dtype=int)
    X.index.name = candidate_id_col
    for cid, elms in elm_lists:
        for e in elms:
            if e in keep:
                X.loc[cid, e] = 1

    if include_quadratic:
        if min_interaction_freq is None:
            min_interaction_freq = min_freq

        scored = []
        for a, b in combinations(X.columns, 2):
            v = (X[a].values * X[b].values).astype(int)
            freq = float(v.mean()) if n > 0 else 0.0
            if freq >= float(min_interaction_freq):
                scored.append((freq, a, b, v))

        # keep most frequent interactions
        scored.sort(reverse=True, key=lambda t: t[0])
        if max_interactions is not None:
            scored = scored[: int(max_interactions)]

        inter_cols = {f"{a}{interaction_sep}{b}": v for (freq, a, b, v) in scored}
        if inter_cols:
            X_inter = pd.DataFrame(inter_cols, index=X.index, dtype=int)
            X = pd.concat([X, X_inter], axis=1)

    return X


def merge_candidate_metadata(
    elms_xlsx: str | Path,
    topo_xlsx: str | Path,
    out_xlsx: str | Path | None = None,
    sheet_elms: str | int = 0,
    sheet_topo: str | int = 0,
    spec: CandidateMergeSpec = CandidateMergeSpec(),
) -> pd.DataFrame:
    """Merge ELM annotations with topology information.

    Performs an inner join between ELM annotation data and topology/protein
    family data, producing a unified candidate metadata table.

    Parameters
    ----------
    elms_xlsx : str or Path
        Path to Excel file with ELM annotations (e.g., costim_normalized_elms_groupings.xlsx).
    topo_xlsx : str or Path
        Path to Excel file with topology data (e.g., costim_topol_protein_families.xlsx).
    out_xlsx : str, Path, or None, default None
        If provided, write merged data to this Excel file.
    sheet_elms : str or int, default 0
        Sheet name or index in the ELM file.
    sheet_topo : str or int, default 0
        Sheet name or index in the topology file.
    spec : CandidateMergeSpec, default CandidateMergeSpec()
        Specification for column name mapping.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with columns: CandidateID, ELMCategory, ICD Num, Num ICD.

    Raises
    ------
    ValueError
        If required columns are missing from input files.

    Examples
    --------
    >>> cand = merge_candidate_metadata(
    ...     "data/costim_normalized_elms_groupings.xlsx",
    ...     "data/costim_topol_protein_families.xlsx",
    ...     out_xlsx="data/candidate_metadata.xlsx"
    ... )
    """
    elms_xlsx = Path(elms_xlsx)
    topo_xlsx = Path(topo_xlsx)

    a = pd.read_excel(elms_xlsx, sheet_name=sheet_elms)
    b = pd.read_excel(topo_xlsx, sheet_name=sheet_topo)

    # normalize column names (strip whitespace)
    a.columns = _normalize_excel_columns(a.columns)
    b.columns = _normalize_excel_columns(b.columns)

    needed_a = {spec.id_col, spec.elm_vocab_col}
    needed_b = {spec.id_col, spec.icd_id_col, spec.icd_mult_col}

    missing_a = needed_a - set(a.columns)
    missing_b = needed_b - set(b.columns)
    if missing_a:
        raise ValueError(f"{elms_xlsx} missing columns: {sorted(missing_a)}")
    if missing_b:
        raise ValueError(f"{topo_xlsx} missing columns: {sorted(missing_b)}")

    a = a.loc[:, [spec.id_col, spec.elm_vocab_col]].copy()
    b = b.loc[:, [spec.id_col, spec.icd_id_col, spec.icd_mult_col]].copy()

    merged = a.merge(b, on=spec.id_col, how="inner")

    merged = merged.rename(
        columns={
            spec.id_col: spec.out_id_col,
            spec.elm_vocab_col: spec.out_elm_col,
            spec.icd_id_col: spec.out_icd_id_col,
            spec.icd_mult_col: spec.out_icd_mult_col,
        }
    )

    # Remove surrounding brackets from ELMCategory values
    merged[spec.out_elm_col] = (
        merged[spec.out_elm_col]
        .astype(str)
        .str.replace(r"^\[", "", regex=True)
        .str.replace(r"\]$", "", regex=True)
    )

    if out_xlsx is not None:
        out_xlsx = Path(out_xlsx)
        out_xlsx.parent.mkdir(parents=True, exist_ok=True)
        merged.to_excel(out_xlsx, index=False, sheet_name="candidate_metadata")

    return merged


def _normalize_excel_columns(cols: Iterable) -> list[str]:
    """Strip whitespace from Excel column names."""
    return [str(c).strip() for c in cols]


def sample_metadata_from_counts_xlsx(
    counts_xlsx: str | Path,
    out_xlsx: str | Path | None = "data/sample_metadata.xlsx",
    sheet_name: str | int = 0,
    spec: SampleParseSpec = SampleParseSpec(),
    domain_id_col: str | None = None,
    skip_first_col_if_unknown: bool = True,
    strict: bool = True,
) -> pd.DataFrame:
    """Parse sample metadata from count matrix column headers.

    Reads the header row of a counts Excel file and parses sample metadata
    from the column names by splitting on a separator character.

    Parameters
    ----------
    counts_xlsx : str or Path
        Path to the counts Excel file.
    out_xlsx : str, Path, or None, default "data/sample_metadata.xlsx"
        If provided, write parsed metadata to this Excel file.
    sheet_name : str or int, default 0
        Sheet name or index to read.
    spec : SampleParseSpec, default SampleParseSpec()
        Specification for parsing sample IDs.
    domain_id_col : str or None, default None
        Name of the domain/candidate ID column to exclude. If None and
        skip_first_col_if_unknown is True, the first column is excluded.
    skip_first_col_if_unknown : bool, default True
        If domain_id_col is not found, skip the first column.
    strict : bool, default True
        If True, raise an error if sample IDs don't parse correctly.

    Returns
    -------
    pd.DataFrame
        Sample metadata DataFrame indexed by sample_id with columns
        for each field in spec.fields.

    Examples
    --------
    >>> smeta = sample_metadata_from_counts_xlsx(
    ...     "data/merged_counts.xlsx",
    ...     out_xlsx="data/sample_metadata.xlsx"
    ... )
    """
    counts_xlsx = Path(counts_xlsx)

    header = pd.read_excel(counts_xlsx, sheet_name=sheet_name, nrows=0)
    cols = _normalize_excel_columns(header.columns)

    sample_cols = cols[:]
    if domain_id_col is not None and domain_id_col in sample_cols:
        sample_cols = [c for c in sample_cols if c != domain_id_col]
    elif skip_first_col_if_unknown and len(sample_cols) > 0:
        sample_cols = sample_cols[1:]

    rows = [split_sample_id(c, spec=spec, strict=strict) for c in sample_cols]
    smeta = pd.DataFrame(rows).set_index("sample_id")
    smeta.index.name = "sample_id"

    if out_xlsx is not None:
        out_xlsx = Path(out_xlsx)
        out_xlsx.parent.mkdir(parents=True, exist_ok=True)
        smeta.to_excel(out_xlsx, index=True, sheet_name="sample_metadata")

    return smeta


def split_sample_id(
    sample_id: str,
    spec: SampleParseSpec = SampleParseSpec(),
    strict: bool = True,
) -> dict[str, object]:
    """Split a sample ID string into metadata fields.

    Parses a sample ID like "1_CAR:Raji_CM_High_1" into its component
    fields (Donor, ExpCond, Tsubset, PD1Status, Replicate).

    Parameters
    ----------
    sample_id : str
        The sample ID string to parse.
    spec : SampleParseSpec, default SampleParseSpec()
        Specification for parsing (separator and field names).
    strict : bool, default True
        If True, raise ValueError if the number of tokens doesn't match
        the expected number of fields.

    Returns
    -------
    dict
        Dictionary with 'sample_id' key plus one key per field in spec.fields.

    Raises
    ------
    ValueError
        If strict=True and the number of tokens doesn't match spec.fields.

    Examples
    --------
    >>> split_sample_id("1_CAR:Raji_CM_High_1")
    {'sample_id': '1_CAR:Raji_CM_High_1', 'Donor': '1', 'ExpCond': 'CAR:Raji',
     'Tsubset': 'CM', 'PD1Status': 'High', 'Replicate': '1'}
    """
    sid = str(sample_id).strip()
    toks = sid.split(spec.sep)

    if strict and len(toks) != len(spec.fields):
        raise ValueError(
            f"Sample ID '{sid}' split by '{spec.sep}' yielded {len(toks)} tokens "
            f"(expected {len(spec.fields)}). Tokens={toks}"
        )

    # best-effort if non-strict
    toks = (toks + [""] * len(spec.fields))[: len(spec.fields)]
    out: dict[str, object] = {"sample_id": sid}
    out.update({k: v for k, v in zip(spec.fields, toks)})

    return out