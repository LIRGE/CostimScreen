"""
Input/output functions for loading screen data.

This module provides functions for reading Excel files containing count matrices,
sample metadata, and candidate metadata. It handles common formatting issues and
normalizes column names.

Functions
---------
load_counts_matrix
    Load a count matrix from Excel with candidate IDs as index.
load_sample_metadata
    Load sample metadata from Excel with sample IDs as index.
load_candidate_metadata
    Load candidate metadata including ELM categories.
parse_samples_from_columns
    Parse sample metadata from column naming conventions.
write_sample_metadata_template
    Write a blank sample metadata template CSV.

Classes
-------
Paths
    Data class holding paths to data and results directories.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class Paths:
    """Container for data and results directory paths."""

    #: Path to the directory containing input data files.
    data_path: Path
    #: Path to the directory for output results.
    results_path: Path


def load_candidate_metadata(
    candidate_meta_xlsx: str | Path,
    sheet_name: str | int = 0,
    candidate_id_col: str = "CandidateID",
) -> pd.DataFrame:
    """Load candidate metadata from an Excel file.

    Reads a candidate metadata Excel file and returns a DataFrame indexed by
    candidate ID. Computes a derived ``is_gpcr`` column based on the number
    of intracellular domains.

    Parameters
    ----------
    candidate_meta_xlsx : str or Path
        Path to the Excel file containing candidate metadata.
    sheet_name : str or int, default 0
        Sheet name or index to read.
    candidate_id_col : str, default "CandidateID"
        Name of the column containing candidate identifiers.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by CandidateID with columns:
        - ELMCategory: semicolon-separated ELM motif categories
        - ICD Num: intracellular domain number
        - Num ICD: number of intracellular domains
        - is_gpcr: 1 if Num ICD > 4, else 0

    Raises
    ------
    ValueError
        If required columns are missing from the input file.

    Examples
    --------
    >>> cand = load_candidate_metadata("data/candidate_metadata.xlsx")
    >>> cand.head()
    """
    candidate_meta_xlsx = Path(candidate_meta_xlsx)
    cand = pd.read_excel(candidate_meta_xlsx, sheet_name=sheet_name)
    cand = _norm_cols(cand)

    if candidate_id_col not in cand.columns:
        raise ValueError(f"{candidate_meta_xlsx} missing '{candidate_id_col}' column.")

    cand[candidate_id_col] = cand[candidate_id_col].astype(str)
    cand = cand.set_index(candidate_id_col)
    cand.index.name = "CandidateID"

    # Tolerate alternate spellings/spaces
    rename = {"ICD Num": "ICD Num", "Num ICD": "Num ICD"}
    cand = cand.rename(columns={k: v for k, v in rename.items() if k in cand.columns})

    required = ["ELMCategory", "ICD Num", "Num ICD"]
    missing = [c for c in required if c not in cand.columns]
    if missing:
        raise ValueError(f"{candidate_meta_xlsx} missing required columns: {missing}")

    cand["Num ICD"] = pd.to_numeric(cand["Num ICD"], errors="coerce")
    cand["ICD Num"] = pd.to_numeric(cand["ICD Num"], errors="coerce")
    cand["is_gpcr"] = (cand["Num ICD"].fillna(0) > 4).astype(int)

    return cand


def load_counts_matrix(
    counts_xlsx: str | Path,
    sheet_name: str | int = 0,
    candidate_id_col: Optional[str] = "CandidateID",
    skip_first_col_if_unknown: bool = True,
) -> pd.DataFrame:
    """Load a count matrix from an Excel file.

    Reads a count matrix where rows are candidates and columns are samples.
    Values are expected to be integer counts.

    Parameters
    ----------
    counts_xlsx : str or Path
        Path to the Excel file containing the count matrix.
    sheet_name : str or int, default 0
        Sheet name or index to read.
    candidate_id_col : str or None, default "CandidateID"
        Name of the column containing candidate IDs. If None, attempts
        to infer from the data.
    skip_first_col_if_unknown : bool, default True
        If candidate_id_col is not found, treat the first column as the ID column.

    Returns
    -------
    pd.DataFrame
        Count matrix with CandidateID as index and sample_id as columns.
        Values are integers.

    Raises
    ------
    ValueError
        If the candidate ID column cannot be determined.

    Examples
    --------
    >>> counts = load_counts_matrix("data/merged_counts.xlsx")
    >>> counts.shape
    (1000, 36)
    """
    counts_xlsx = Path(counts_xlsx)
    df = pd.read_excel(counts_xlsx, sheet_name=sheet_name)
    df = _norm_cols(df)

    if candidate_id_col is not None and candidate_id_col in df.columns:
        id_col = candidate_id_col
    elif skip_first_col_if_unknown:
        id_col = df.columns[0]
    else:
        raise ValueError(f"Could not determine CandidateID column in {counts_xlsx}.")

    df[id_col] = df[id_col].astype(str)
    df = df.set_index(id_col)
    df.index.name = "CandidateID"

    # coerce to int raw counts
    df = df.apply(pd.to_numeric, errors="raise").fillna(0).astype(int)

    return df


def load_sample_metadata(
    sample_meta_xlsx: str | Path,
    sheet_name: str | int = 0,
    sample_id_col: str = "sample_id",
) -> pd.DataFrame:
    """Load sample metadata from an Excel file.

    Reads sample metadata and returns a DataFrame indexed by sample ID.
    Handles common column name variations and typos.

    Parameters
    ----------
    sample_meta_xlsx : str or Path
        Path to the Excel file containing sample metadata.
    sheet_name : str or int, default 0
        Sheet name or index to read.
    sample_id_col : str, default "sample_id"
        Name of the column containing sample identifiers.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by sample_id with columns:
        - Donor: donor identifier
        - ExpCond: experimental condition
        - Tsubset: T-cell subset (e.g., CM, EM, Naive)
        - PD1Status: PD1 expression level (High/Low)
        - Replicate: replicate number

    Raises
    ------
    ValueError
        If required columns are missing from the input file.

    Examples
    --------
    >>> smeta = load_sample_metadata("data/sample_metadata.xlsx")
    >>> smeta["Tsubset"].unique()
    array(['CM', 'EM', 'Naive'], dtype=object)
    """
    sample_meta_xlsx = Path(sample_meta_xlsx)
    smeta = pd.read_excel(sample_meta_xlsx, sheet_name=sheet_name)
    smeta = _norm_cols(smeta)

    # tolerate your current quirks
    rename = {
        "Donor;": "Donor",
        "ExpCOnd": "ExpCond",
        "ExpCond;": "ExpCond",
        "Tsubset;": "Tsubset",
        "PD1Status;": "PD1Status",
        "Replicate;": "Replicate",
    }
    smeta = smeta.rename(columns={k: v for k, v in rename.items() if k in smeta.columns})

    # Find the sample_id column - check for common names
    id_col_found = None
    if sample_id_col in smeta.columns:
        id_col_found = sample_id_col
    else:
        # Check for unnamed index column that pandas creates when reading Excel with index
        for candidate in ["Unnamed: 0", "index"]:
            if candidate in smeta.columns:
                id_col_found = candidate
                break

    if id_col_found is None:
        raise ValueError(f"{sample_meta_xlsx} missing '{sample_id_col}' column.")

    smeta[id_col_found] = smeta[id_col_found].astype(str)
    smeta = smeta.set_index(id_col_found)
    smeta.index.name = "sample_id"

    required = ["Donor", "ExpCond", "Tsubset", "PD1Status", "Replicate"]
    missing = [c for c in required if c not in smeta.columns]
    if missing:
        raise ValueError(f"{sample_meta_xlsx} missing required columns: {missing}")

    return smeta


def _norm_col(c: object) -> str:
    """Strip whitespace from a column name."""
    return str(c).strip()


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize all column names by stripping whitespace."""
    df = df.copy()
    df.columns = [_norm_col(c) for c in df.columns]
    return df


def parse_samples_from_columns(
    sample_ids: list[str],
    allowed_conditions: tuple[str, ...] = ("Alone", "Activated", "K562", "Raji"),
    allowed_memory: tuple[str, ...] = ("Naive", "CM", "EM"),
) -> pd.DataFrame:
    """Parse sample metadata from column naming conventions.

    Attempts to extract donor, condition, memory subset, PD1 status, and
    replicate information from sample ID strings using pattern matching.

    Parameters
    ----------
    sample_ids : list of str
        List of sample ID strings to parse.
    allowed_conditions : tuple of str, default ("Alone", "Activated", "K562", "Raji")
        Valid experimental condition names.
    allowed_memory : tuple of str, default ("Naive", "CM", "EM")
        Valid T-cell memory subset names.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by sample_id with parsed metadata columns:
        donor, condition, memory, pd1, rep. Values may be None if parsing fails.

    Examples
    --------
    >>> sample_ids = ["D1_Raji_CM_PD1high_rep1", "D2_K562_EM_PD1low_rep2"]
    >>> smeta = parse_samples_from_columns(sample_ids)
    """
    cond_map = {c.lower(): c for c in allowed_conditions}
    mem_map = {
        "naive": "Naive",
        "nai": "Naive",
        "cm": "CM",
        "centralmemory": "CM",
        "central": "CM",
        "em": "EM",
        "effectormemory": "EM",
        "effector": "EM",
    }

    rows = []
    for sid in sample_ids:
        toks = _tokenize_sample_id(sid.lower())

        donor = None
        rep = None
        condition = None
        memory = None
        pd1 = None

        # donor
        m = re.search(r"(?:donor|d)(\d+)", sid.lower())
        if m:
            donor = int(m.group(1))

        # rep
        m = re.search(r"(?:rep|r)(\d+)", sid.lower())
        if m:
            rep = int(m.group(1))

        # condition
        for t in toks:
            if t in cond_map:
                condition = cond_map[t]
                break

        # memory
        for t in toks:
            if t in mem_map:
                memory = mem_map[t]
                break

        # PD1
        # look for pd1hi/pd1high/pd1lo/pd1low or standalone hi/low if pd1 present
        if any("pd1" in t for t in toks):
            if any(t.endswith(("hi", "high")) or t == "hi" or t == "high" for t in toks):
                pd1 = "High"
            if any(t.endswith(("lo", "low")) or t == "lo" or t == "low" for t in toks):
                pd1 = "Low"
        else:
            # fallback: explicit tokens like pd1high/pd1low without pd1 split
            if "pd1high" in toks or "pd1hi" in toks:
                pd1 = "High"
            elif "pd1low" in toks or "pd1lo" in toks:
                pd1 = "Low"

        rows.append(
            {
                "sample_id": sid,
                "donor": donor,
                "condition": condition,
                "memory": memory,
                "pd1": pd1,
                "rep": rep,
            }
        )

    smeta = pd.DataFrame(rows).set_index("sample_id")

    # If parsing failed for some samples, user can fill manually.
    return smeta


def _tokenize_sample_id(sample_id: str) -> list[str]:
    """Split a sample ID string into tokens."""
    toks = re.split(r"[\s_\-\.]+", sample_id.strip())
    return [t for t in toks if t]


def write_sample_metadata_template(sample_ids: list[str], out_csv: str | Path) -> None:
    """Write a blank sample metadata template CSV.

    Creates a CSV file with sample IDs and empty columns for manual
    metadata entry.

    Parameters
    ----------
    sample_ids : list of str
        List of sample ID strings.
    out_csv : str or Path
        Output path for the CSV file.

    Examples
    --------
    >>> write_sample_metadata_template(["sample1", "sample2"], "template.csv")
    """
    out_csv = Path(out_csv)
    df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "donor": pd.NA,
            "condition": pd.NA,
            "memory": pd.NA,
            "pd1": pd.NA,
            "rep": pd.NA,
        }
    )
    df.to_csv(out_csv, index=False)