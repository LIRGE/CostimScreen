from __future__ import annotations

from collections import Counter
import pandas as pd
import re
from typing import Callable, Optional, Tuple


def build_elm_design(
    domains_df: pd.DataFrame,
    elms_col: str = "elms",
    min_freq: float = 0.01,
    collapse_fn: Optional[Callable[[str], str]] = None,
) -> pd.DataFrame:
    """
    Returns X_elm: index=domain_id, columns=elm_feature, values in {0,1}.
    - min_freq is fraction of domains containing the feature to keep it.
    - collapse_fn can map raw ELM names -> grouped feature names (e.g., partner-based grouping).
    """
    domain_ids = domains_df.index.astype(str)
    elm_lists = []
    counter = Counter()

    for did in domain_ids:
        elms = split_elm_list(domains_df.loc[did, elms_col] if elms_col in domains_df.columns else "")
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
        prefix: str = "F_"
) -> Tuple[list[str], dict[str, str]]:
    """
    Returns (safe_cols, mapping original->safe), guaranteeing uniqueness.
    """
    safe = []
    mapping = {}
    used = set()

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
    if pd.isna(s) or str(s).strip() == "":
        return []
    raw = str(s).replace("|", ";").replace(",", ";")
    parts = [p.strip() for p in raw.split(";")]
    return [p for p in parts if p]
