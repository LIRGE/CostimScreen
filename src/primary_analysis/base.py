"""
Base class for FASTQ parsing.

Provides the foundation for targeted sequencing FASTQ readers with
multiprocessing support, metadata handling, and count aggregation.
"""

import gzip
import logging
import os
import re
from abc import ABC, abstractmethod
from multiprocessing import Pool, cpu_count
from os import PathLike
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import pandas as pd

from .read_db import ReadDB
from .sequencing_read import SequencingRead

logger = logging.getLogger(__name__)


class FASTQParseError(Exception):
    """Raised when FASTQ parsing encounters an error."""
    pass


class ReadFASTQ(ABC):
    """
    Abstract base class for FASTQ parsing with multiprocessing support.

    Provides infrastructure for:
    - Loading and validating FASTQ files
    - Parsing feature and sample metadata
    - Parallel read processing across multiple cores
    - Count aggregation and serialization

    Subclasses must implement:
    - `_parse_feature_metadata()`: Load feature whitelist/metadata
    - `_parse_sample_metadata()`: Load or infer sample metadata

    Parameters
    ----------
    fastq_path : PathLike or str
        Directory containing FASTQ files (*.fastq.gz).
    feature_metadata_fn : PathLike or str, optional
        Path to feature metadata Excel file.
    sample_metadata_fn : PathLike or str, optional
        Path to sample metadata Excel file. If None, metadata is
        inferred from FASTQ filenames.
    debug : bool, default False
        Enable debug mode (limits reads processed, extra logging).
    downstream_const_seq : str, optional
        Constant sequence downstream of variable region.
    upstream_const_seq : str, optional
        Constant sequence upstream of variable region.
    fastq_encoding : str, optional
        Filename encoding scheme for inferring sample metadata.
    fastq_type : {'gDNA', 'mRNA'}, default 'gDNA'
        Type of sequencing data.
    n_char_features : int, default 10
        Number of characters to extract from variable region.
    num_cores : int, optional
        Number of CPU cores for parallel processing.
        Defaults to (available cores - 2).
    read_error_threshold : int, default 2
        Maximum edit distance for fuzzy matching.

    Attributes
    ----------
    merged_counts_df : pd.DataFrame
        Aggregated count matrix after calling `read()`.
    feature_metadata_df : pd.DataFrame
        Feature metadata after parsing.
    sample_metadata_df : pd.DataFrame
        Sample metadata after parsing.
    """

    def __init__(
        self,
        fastq_path: Union[PathLike, str],
        feature_metadata_fn: Optional[Union[PathLike, str]] = None,
        sample_metadata_fn: Optional[Union[PathLike, str]] = None,
        debug: bool = False,
        downstream_const_seq: Optional[str] = None,
        upstream_const_seq: Optional[str] = None,
        fastq_encoding: Optional[str] = None,
        fastq_type: Literal['gDNA', 'mRNA'] = 'gDNA',
        n_char_features: int = 10,
        num_cores: Optional[int] = None,
        read_error_threshold: int = 2,
    ):
        self._debug = debug
        self._fastq_path = Path(fastq_path)
        self._fastq_type = fastq_type
        self._n_char_features = n_char_features
        self._read_error_threshold = read_error_threshold

        # Constant sequences (lowercase for matching)
        self._downstream_const_seq = downstream_const_seq.lower() if downstream_const_seq else None
        self._upstream_const_seq = upstream_const_seq.lower() if upstream_const_seq else None

        # Validate at least one anchor is provided
        if self._downstream_const_seq is None and self._upstream_const_seq is None:
            raise ValueError("At least one of downstream_const_seq or upstream_const_seq must be provided")

        # Debug directory
        if self._debug:
            logger.info("Running in DEBUG mode")
            self._debug_dir = Path.cwd() / "debug"
            self._debug_dir.mkdir(exist_ok=True)
        else:
            self._debug_dir = None

        # Determine number of cores
        if num_cores is None:
            if hasattr(os, 'sched_getaffinity'):
                self._num_cores = max(1, len(os.sched_getaffinity(0)) - 2)
            else:
                self._num_cores = max(1, cpu_count() - 2)
        else:
            self._num_cores = num_cores
        logger.info(f"Using {self._num_cores} cores for parallel processing")

        # Find FASTQ files
        self._find_fastq_files()

        # Initialize metadata placeholders
        self.feature_metadata_df = pd.DataFrame()
        self.sample_metadata_df = pd.DataFrame()
        self.merged_counts_df = pd.DataFrame()

        # Feature whitelist/blacklist (populated by subclass)
        self._feature_whitelist: List[str] = []
        self._feature_blacklist: List[str] = []
        self._seq_to_feature: Dict[str, str] = {}

        # Sample name mapping
        self._r1_path_to_sample: Dict[str, str] = {}

        # Parse metadata (subclass implementations)
        self._parse_feature_metadata(feature_metadata_fn)
        self._parse_sample_metadata(sample_metadata_fn, fastq_encoding)

    def _find_fastq_files(self) -> None:
        """Locate R1 and R2 FASTQ files in the specified directory."""
        if not self._fastq_path.exists():
            raise FileNotFoundError(f"FASTQ path does not exist: {self._fastq_path}")

        # Find R1 files
        self._r1_file_list = sorted([
            str(f) for f in self._fastq_path.glob("*_R1.fastq.gz")
        ])

        if not self._r1_file_list:
            # Try alternative pattern
            self._r1_file_list = sorted([
                str(f) for f in self._fastq_path.glob("*_R1_*.fastq.gz")
            ])

        if not self._r1_file_list:
            raise FASTQParseError(f"No R1 FASTQ files found in {self._fastq_path}")

        logger.info(f"Found {len(self._r1_file_list)} R1 FASTQ files")

        # Find matching R2 files (may not exist for single-end)
        self._r2_file_list = []
        for r1_path in self._r1_file_list:
            r2_path = r1_path.replace('_R1.', '_R2.').replace('_R1_', '_R2_')
            if os.path.exists(r2_path):
                self._r2_file_list.append(r2_path)
            else:
                self._r2_file_list.append(None)

    @abstractmethod
    def _parse_feature_metadata(
        self,
        feature_metadata_fn: Optional[Union[PathLike, str]] = None,
    ) -> None:
        """
        Parse feature metadata and build whitelist.

        Must populate:
        - self.feature_metadata_df
        - self._feature_whitelist
        - self._feature_blacklist
        - self._seq_to_feature
        """
        pass

    @abstractmethod
    def _parse_sample_metadata(
        self,
        sample_metadata_fn: Optional[Union[PathLike, str]] = None,
        fastq_encoding: Optional[str] = None,
    ) -> None:
        """
        Parse or infer sample metadata.

        Must populate:
        - self.sample_metadata_df
        - self._r1_path_to_sample
        """
        pass

    def read(self) -> None:
        """
        Read all FASTQ files and aggregate counts.

        Uses multiprocessing to parse files in parallel. Results are
        merged into `self.merged_counts_df`.
        """
        # Build argument list for parallel processing
        arguments = []
        for r1_fn, r2_fn in zip(self._r1_file_list, self._r2_file_list):
            sample_name = self._r1_path_to_sample.get(r1_fn, 'Unknown')
            arguments.append((
                r1_fn,
                r2_fn,
                self._feature_blacklist,
                self._feature_whitelist,
                sample_name,
                self._debug,
                self._downstream_const_seq,
                self._upstream_const_seq,
                self._n_char_features,
                self._read_error_threshold,
            ))

        logger.info(f"Processing {len(arguments)} FASTQ files...")

        if self._fastq_type == 'gDNA':
            with Pool(processes=self._num_cores) as pool:
                sample_counts = pool.starmap(self._read_from_gdna, arguments)
        elif self._fastq_type == 'mRNA':
            with Pool(processes=self._num_cores) as pool:
                sample_counts = pool.starmap(self._read_from_mrna, arguments)
        else:
            raise ValueError(f"Unknown FASTQ type: {self._fastq_type}")

        # Merge results
        logger.info("Merging counts from all samples...")
        valid_counts = [c for c in sample_counts if c is not None and len(c) > 0]

        if not valid_counts:
            logger.warning("No counts obtained from any sample")
            self.merged_counts_df = pd.DataFrame()
            return

        self.merged_counts_df = pd.concat(valid_counts, axis=1)

        # Sort columns (samples) and rows (features)
        self.merged_counts_df = self.merged_counts_df.sort_index(axis=1)

        # Map sequences to feature names
        self.merged_counts_df = self.merged_counts_df.rename(index=self._seq_to_feature)
        self.merged_counts_df.index = self.merged_counts_df.index.astype(str)
        self.merged_counts_df = self.merged_counts_df.sort_index(axis=0)

        # Fill NaN with 0
        self.merged_counts_df = self.merged_counts_df.fillna(0).astype(int)

        logger.info(f"Merged counts: {self.merged_counts_df.shape[0]} features x {self.merged_counts_df.shape[1]} samples")

    @staticmethod
    def _read_from_gdna(
        r1_fn: str,
        r2_fn: Optional[str],
        feature_blacklist: List[str],
        feature_whitelist: List[str],
        sample: str,
        debug: bool,
        downstream_const_seq: Optional[str],
        upstream_const_seq: Optional[str],
        n_char_features: int,
        read_error_threshold: int,
    ) -> pd.DataFrame:
        """
        Parse a single gDNA FASTQ file and return counts.

        This is a static method to enable multiprocessing.
        """
        read_db = ReadDB(feature_whitelist=feature_whitelist, sample_list=[sample])

        try:
            with gzip.open(r1_fn, 'rt') as f:
                for line_num, line in enumerate(f):
                    # Log progress every 100k reads
                    if line_num % 400000 == 0 and line_num > 0:
                        logger.info(f"{sample}: {line_num // 4:,} reads processed...")

                    # Sequence line is every 4th line starting at 1
                    if line_num % 4 == 1:
                        seq = line.rstrip().lower()

                        read = SequencingRead(
                            seq=seq,
                            n_char=n_char_features,
                            downstream_const_seq=downstream_const_seq,
                            upstream_const_seq=upstream_const_seq,
                        )

                        if not read.empty():
                            corrected, edit_dist = read.best_match_to_whitelist(
                                whitelist=feature_whitelist,
                                blacklist=feature_blacklist,
                                read_error_threshold=read_error_threshold,
                            )
                            if corrected is not None:
                                read_db.increment_count(corrected, sample)

                    # Debug mode: limit reads
                    if debug and line_num >= 400000:
                        break

            logger.info(f"{sample}: Complete - {read_db.total_counts:,} counts")
            return read_db.counts()

        except Exception as e:
            logger.error(f"Error processing {r1_fn}: {e}")
            return pd.DataFrame()

    @staticmethod
    def _read_from_mrna(
        r1_fn: str,
        r2_fn: Optional[str],
        feature_blacklist: List[str],
        feature_whitelist: List[str],
        sample: str,
        debug: bool,
        downstream_const_seq: Optional[str],
        upstream_const_seq: Optional[str],
        n_char_features: int,
        read_error_threshold: int,
    ) -> pd.DataFrame:
        """
        Parse a single mRNA FASTQ file and return counts.

        For mRNA, may need UMI deduplication (not yet implemented).
        """
        # For now, treat same as gDNA
        # TODO: Add UMI handling for mRNA
        return ReadFASTQ._read_from_gdna(
            r1_fn, r2_fn, feature_blacklist, feature_whitelist,
            sample, debug, downstream_const_seq, upstream_const_seq,
            n_char_features, read_error_threshold
        )

    def serialize(
        self,
        results_path: Union[PathLike, str],
        format: Literal['excel', 'csv'] = 'excel',
    ) -> None:
        """
        Save results to disk.

        Parameters
        ----------
        results_path : PathLike or str
            Directory to save results.
        format : {'excel', 'csv'}, default 'excel'
            Output format.
        """
        results_path = Path(results_path)
        results_path.mkdir(parents=True, exist_ok=True)

        if format == 'excel':
            self.feature_metadata_df.to_excel(results_path / 'feature_metadata.xlsx', index=False)
            self.sample_metadata_df.to_excel(results_path / 'sample_metadata.xlsx', index=False)
            self.merged_counts_df.to_excel(results_path / 'merged_counts.xlsx')
        else:
            self.feature_metadata_df.to_csv(results_path / 'feature_metadata.csv', index=False)
            self.sample_metadata_df.to_csv(results_path / 'sample_metadata.csv', index=False)
            self.merged_counts_df.to_csv(results_path / 'merged_counts.csv')

        logger.info(f"Results saved to {results_path}")

    def print_summary(self) -> None:
        """Print a summary of the parsed data."""
        print(f"FASTQ Path: {self._fastq_path}")
        print(f"R1 Files: {len(self._r1_file_list)}")
        print(f"Features in whitelist: {len(self._feature_whitelist)}")
        print(f"Features in blacklist: {len(self._feature_blacklist)}")
        if len(self.merged_counts_df) > 0:
            print(f"Count matrix: {self.merged_counts_df.shape[0]} features x {self.merged_counts_df.shape[1]} samples")
            print(f"Total counts: {self.merged_counts_df.sum().sum():,}")
