# costim_screen/preprocessing.py
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .features import split_elm_list


@dataclass(frozen=True)
class CandidateMergeSpec:
    # input columns
    id_col: str = "ID"
    elm_vocab_col: str = "elm vocab"
    icd_id_col: str = "ICD ID"
    icd_mult_col: str = "Gene ICD Multiplicity"
    # output columns
    out_id_col: str = "CandidateID"
    out_elm_col: str = "ELMCategory"
    out_icd_id_col: str = "ICD Num"
    out_icd_mult_col: str = "Num ICD"


@dataclass(frozen=True)
class SampleParseSpec:
    sep: str = "_"
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
    """
    Build a one-hot design matrix for ELM categories.

    Parameters
    ----------
    candidates_df : pd.DataFrame
        DataFrame with candidate IDs and ELM category strings.
    candidate_id_col : str
        Column name for candidate IDs (used as index).
    elm_col : str
        Column containing ELM category strings (semicolon/comma/pipe separated).
    min_freq : float
        Minimum fraction of candidates containing a feature to keep it.
    include_quadratic : bool
        If True, add pairwise interaction terms.
    min_interaction_freq : float | None
        Minimum frequency for interaction terms; defaults to min_freq.
    max_interactions : int | None
        Maximum number of interaction terms to keep (by frequency).
    interaction_sep : str
        Separator for interaction column names.

    Returns
    -------
    pd.DataFrame
        One-hot encoded design matrix, index=candidate_id, columns=features.
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
    """
    Inner join on 'ID' between:
      - data/costim_normalized_elms_groupings.xlsx: ['ID', 'elm vocab']
      - data/costim_topol_protein_families.xlsx: ['ID', 'ICD ID', 'ICD Multiplicity']

    Renames:
      ID -> CandidateID
      elm vocab -> ELMCategory
      ICD ID -> ICD Num
      ICD Multiplicity -> Num ICD

    Writes to out_xlsx if provided, and returns merged DataFrame.
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

    # Remove surrounding brackets from ELMCategory values (e.g., "[elm1, elm2]" -> "elm1, elm2")
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
    """
    Reads ONLY the header of the counts matrix workbook and parses sample metadata
    from column names by splitting on spec.sep.

    Assumptions:
      - One column is the domain/candidate ID column.
      - All other columns are sample IDs encoding metadata.

    domain_id_col:
      - If provided and present, it's excluded from sample columns.
      - Otherwise, if skip_first_col_if_unknown=True, the first column is excluded.

    Writes to out_xlsx if provided, and returns the metadata DataFrame indexed by sample_id.
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
    """
    Split a sample_id like:
      Donor_ExpCond_Tsubset_PD1Status_Replicate
    using spec.sep into spec.fields.

    If strict=True, raises if the number of tokens != len(fields).
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
    out = {"sample_id": sid}
    out.update({k: v for k, v in zip(spec.fields, toks)})

    return out
