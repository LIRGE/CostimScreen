from __future__ import annotations

import re
import unicodedata


def normalize_phenotype(s: str) -> str:
    """Normalize phenotype-ish strings to a canonical ASCII form.

    - Removes diacritics (Naïve -> Naive)
    - Strips whitespace
    - Normalizes separators to underscore
    - Canonicalizes common tokens:
        * naive/nai -> Naive
        * cm/centralmemory -> CM
        * em/effectormemory -> EM
        * high/hi/pd1high -> High
        * low/lo/pd1low -> Low

    This is intentionally conservative: if the string doesn't look like a
    (Tsubset, PD1) or token we recognize, it just returns the cleaned ASCII.
    """
    # ASCII fold (remove combining marks)
    normalized = unicodedata.normalize("NFKD", str(s))
    ascii_str = "".join(
        c for c in normalized if not unicodedata.category(c).startswith("M")
    ).strip()

    if ascii_str == "":
        return ascii_str

    # Normalize separators
    ascii_str = re.sub(r"[\s\-\/]+", "_", ascii_str)      # spaces/dashes/slashes -> _
    ascii_str = re.sub(r"_+", "_", ascii_str).strip("_")  # collapse __ -> _

    def _canon_token(tok: str) -> str:
        t = tok.strip()
        tl = t.lower()

        # T-subsets
        if tl in {"naive", "nai", "naïve"}:
            return "Naive"
        if tl in {"cm", "centralmemory", "central_memory", "central"}:
            return "CM"
        if tl in {"em", "effectormemory", "effector_memory", "effector"}:
            return "EM"

        # PD1 status (handle common encodings)
        if tl in {"high", "hi", "pd1high", "pd1_hi", "pd1hi"}:
            return "High"
        if tl in {"low", "lo", "pd1low", "pd1_lo", "pd1lo"}:
            return "Low"

        # Leave everything else alone (but keep original casing)
        return t

    toks = ascii_str.split("_")
    toks = [_canon_token(t) for t in toks if t != ""]

    return "_".join(toks)
