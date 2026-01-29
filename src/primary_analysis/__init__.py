"""
Primary Analysis - FASTQ parsing for targeted sequencing.

This module provides tools for parsing targeted sequencing FASTQ files
from CAR-T costimulatory domain screens and related assays.

Main Classes
------------
ReadCostimFASTQ
    Parse costimulatory domain targeted sequencing FASTQ files.

ReadFASTQ
    Abstract base class for FASTQ parsing (for subclassing).

SequencingRead
    Parse and error-correct individual sequencing reads.

ReadDB
    In-memory database for accumulating read counts.

Constants
---------
COSTIM_LEFT, COSTIM_RIGHT
    Flanking constant sequences for costim parsing.

Examples
--------
>>> from primary_analysis import ReadCostimFASTQ
>>> reader = ReadCostimFASTQ(
...     fastq_path='/path/to/fastq',
...     feature_metadata_fn='/path/to/costim_metadata.xlsx',
... )
>>> reader.read()
>>> reader.serialize('/path/to/results')
>>> print(reader.merged_counts_df.shape)
(1200, 48)
"""

from .base import ReadFASTQ, FASTQParseError
from .costim import ReadCostimFASTQ
from .constants import (
    COSTIM_LEFT,
    COSTIM_RIGHT,
    SGRNA_LEFT,
    SGRNA_RIGHT,
    LTI_LEFT,
    LTI_RIGHT,
    FASTQ_NAMING_SCHEME_PT_EC_RP_SS,
    FASTQ_NAMING_SCHEME_PT_EC_RP_EX,
    FASTQ_NAMING_SCHEME_MQ_CT_T,
)
from .read_db import ReadDB
from .sequencing_read import SequencingRead, parse_read_with_anchors

__all__ = [
    # Main classes
    "ReadCostimFASTQ",
    "ReadFASTQ",
    "FASTQParseError",
    # Utility classes
    "SequencingRead",
    "ReadDB",
    # Convenience functions
    "parse_read_with_anchors",
    # Constants
    "COSTIM_LEFT",
    "COSTIM_RIGHT",
    "SGRNA_LEFT",
    "SGRNA_RIGHT",
    "LTI_LEFT",
    "LTI_RIGHT",
    "FASTQ_NAMING_SCHEME_PT_EC_RP_SS",
    "FASTQ_NAMING_SCHEME_PT_EC_RP_EX",
    "FASTQ_NAMING_SCHEME_MQ_CT_T",
]

__version__ = "0.1.0"
