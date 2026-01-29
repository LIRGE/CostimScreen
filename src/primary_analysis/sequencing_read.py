"""
Sequencing read parsing and error correction.

Provides utilities for extracting variable regions from sequencing reads
using flanking constant sequences, and fuzzy matching to known barcodes.
"""

import logging
from typing import List, Optional, Tuple

from rapidfuzz.distance import Hamming
from rapidfuzz.process import extractOne

logger = logging.getLogger(__name__)


class SequencingRead:
    """
    Parse and error-correct a single sequencing read.

    Extracts the variable region from a read by locating flanking constant
    sequences (upstream or downstream anchors). Supports fuzzy matching
    to a whitelist of known sequences.

    Parameters
    ----------
    seq : str
        Raw sequencing read (lowercase).
    n_char : int
        Number of characters to extract from the variable region.
    downstream_const_seq : str, optional
        Constant sequence downstream (3') of the variable region.
        If found, extracts n_char bases starting after this sequence.
    upstream_const_seq : str, optional
        Constant sequence upstream (5') of the variable region.
        If found, extracts n_char bases ending before this sequence.

    Attributes
    ----------
    raw_read : str or None
        Extracted variable region, or None if anchors not found.

    Notes
    -----
    If both downstream and upstream sequences are provided, downstream
    takes precedence. At least one must be provided.

    Examples
    --------
    >>> read = SequencingRead(
    ...     seq="acgtTARGETSEQtggaggatccgccrest",
    ...     n_char=9,
    ...     downstream_const_seq="tggaggatccgcc",
    ...     upstream_const_seq=None
    ... )
    >>> read.raw_read
    'targetseq'  # lowercase
    """

    def __init__(
        self,
        seq: str,
        n_char: int,
        downstream_const_seq: Optional[str] = None,
        upstream_const_seq: Optional[str] = None,
    ):
        self.raw_read: Optional[str] = None

        # Ensure sequence is lowercase for matching
        seq = seq.lower()

        if downstream_const_seq is not None and downstream_const_seq in seq:
            # Extract region AFTER downstream constant (reads left-to-right)
            start_idx = seq.index(downstream_const_seq) + len(downstream_const_seq)
            self.raw_read = seq[start_idx:][:n_char]

        elif upstream_const_seq is not None and upstream_const_seq in seq:
            # Extract region BEFORE upstream constant (last n_char bases)
            end_idx = seq.index(upstream_const_seq)
            self.raw_read = seq[:end_idx][-n_char:]

    def empty(self) -> bool:
        """Check if the read extraction failed (no anchors found)."""
        return self.raw_read is None

    def best_match_to_whitelist(
        self,
        whitelist: List[str],
        blacklist: Optional[List[str]] = None,
        read_error_threshold: int = 2,
    ) -> Tuple[Optional[str], int]:
        """
        Find the best matching sequence from a whitelist.

        Uses Hamming distance for fuzzy matching. Returns the corrected
        sequence if within the error threshold, otherwise None.

        Parameters
        ----------
        whitelist : list of str
            Valid sequences to match against.
        blacklist : list of str, optional
            Sequences to explicitly reject (e.g., ambiguous mappings).
        read_error_threshold : int, default 2
            Maximum allowed Hamming distance for a valid match.

        Returns
        -------
        corrected_read : str or None
            Best matching sequence, or None if no valid match.
        edit_distance : int
            Hamming distance to the best match (or threshold+1 if rejected).

        Examples
        --------
        >>> read = SequencingRead(seq="...", n_char=10, ...)
        >>> read.raw_read = "actgactgac"
        >>> corrected, dist = read.best_match_to_whitelist(
        ...     whitelist=["actgactgac", "aaaaaaagac"],
        ...     read_error_threshold=2
        ... )
        >>> corrected
        'actgactgac'
        >>> dist
        0
        """
        if self.raw_read is None:
            return None, read_error_threshold + 1

        blacklist = blacklist or []

        # Check blacklist first
        if self.raw_read in blacklist:
            return None, read_error_threshold + 1

        # Check exact match
        if self.raw_read in whitelist:
            return self.raw_read, 0

        # Fuzzy match using Hamming distance
        result = extractOne(
            query=self.raw_read,
            choices=whitelist,
            scorer=Hamming.distance,
        )

        if result is None:
            return None, read_error_threshold + 1

        corrected_read, edit_distance, _ = result

        if edit_distance > read_error_threshold:
            return None, edit_distance

        return corrected_read, edit_distance


def parse_read_with_anchors(
    seq: str,
    n_char: int,
    downstream_const_seq: Optional[str] = None,
    upstream_const_seq: Optional[str] = None,
) -> Optional[str]:
    """
    Convenience function to extract variable region from a sequence.

    Parameters
    ----------
    seq : str
        Raw sequencing read.
    n_char : int
        Number of characters to extract.
    downstream_const_seq : str, optional
        Downstream flanking sequence.
    upstream_const_seq : str, optional
        Upstream flanking sequence.

    Returns
    -------
    str or None
        Extracted variable region, or None if anchors not found.
    """
    read = SequencingRead(
        seq=seq,
        n_char=n_char,
        downstream_const_seq=downstream_const_seq,
        upstream_const_seq=upstream_const_seq,
    )
    return read.raw_read
