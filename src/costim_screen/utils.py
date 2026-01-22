"""
Utility functions for the costim_screen package.

This module provides common utilities used across the package,
including text normalization for phenotype labels.

Functions
---------
normalize_phenotype
    Normalize phenotype strings to canonical ASCII form.
"""
from __future__ import annotations

import unicodedata


def normalize_phenotype(s: str) -> str:
    """Normalize phenotype string to canonical ASCII form.

    Converts Unicode characters to their closest ASCII equivalents
    (e.g., 'Naïve' → 'Naive') to ensure consistent matching regardless
    of how phenotype labels were entered.

    Parameters
    ----------
    s : str
        Phenotype string, possibly containing Unicode characters.

    Returns
    -------
    str
        Normalized ASCII phenotype string.

    Examples
    --------
    >>> normalize_phenotype("Naïve_High")
    'Naive_High'
    >>> normalize_phenotype("Naive_High")
    'Naive_High'
    >>> normalize_phenotype("CM_Low")
    'CM_Low'
    """
    # NFKD decomposition splits characters like 'ï' into 'i' + combining diaeresis
    # Then we filter out combining characters (category 'M')
    normalized = unicodedata.normalize("NFKD", str(s))
    ascii_str = "".join(c for c in normalized if not unicodedata.category(c).startswith("M"))
    return ascii_str
