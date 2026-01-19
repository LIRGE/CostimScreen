from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class Paths:
    data_path: Path
    results_path: Path


def load_candidate_metadata(
        candidate_meta_xlsx: str | Path,
        sheet_name: str | int = 0,
        candidate_id_col: str = "CandidateID",
) -> pd.DataFrame:
    """
    Reads data/candidate_metadata.xlsx.

    Expected columns:
      CandidateID, ELMCategory, ICD Num, Num ICD
    Also computes:
      is_gpcr = (Num ICD > 4)
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
    """
    Reads data/merged_counts.xlsx.

    Expected:
      - one column holding candidate IDs (default 'CandidateID').
      - remaining columns are sample_ids, values are raw integer counts.
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
    """
    Reads data/sample_metadata.xlsx.

    Expected columns (with some tolerated typos):
      sample_id, Donor, ExpCond, Tsubset, PD1Status, Replicate
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
    # strip whitespace; preserve internal chars
    return str(c).strip()


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_norm_col(c) for c in df.columns]
    return df


def parse_samples_from_columns(
    sample_ids: list[str],
    allowed_conditions: tuple[str, ...] = ("Alone", "Activated", "K562", "Raji"),
    allowed_memory: tuple[str, ...] = ("Naive", "CM", "EM"),
) -> pd.DataFrame:
    """
    Best-effort parser for sample IDs.

    Expected to find:
      donor: D1 / Donor1 / donor-1 etc.
      condition: one of allowed_conditions
      memory: Naive/CM/EM (or CentralMemory/EffectorMemory)
      pd1: PD1hi/PD1high/hi/high or PD1lo/low
      rep: rep1 / r1 / R1 etc.
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
    toks = re.split(r"[\s_\-\.]+", sample_id.strip())
    return [t for t in toks if t]


def write_sample_metadata_template(sample_ids: list[str], out_csv: str | Path) -> None:
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
