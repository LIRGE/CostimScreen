"""
Costimulatory domain FASTQ reader.

Specialized FASTQ parser for CAR-T costimulatory domain targeted sequencing.
Handles feature metadata containing costim sequences and maps reads to
candidate IDs.
"""

import argparse
import logging
from os import PathLike
from os.path import basename
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import pandas as pd

from .base import ReadFASTQ, FASTQParseError
from .constants import (
    COSTIM_LEFT,
    COSTIM_RIGHT,
    FASTQ_NAMING_SCHEME_PT_EC_RP_EX,
    FASTQ_NAMING_SCHEME_PT_EC_RP_SS,
)

logger = logging.getLogger(__name__)


class ReadCostimFASTQ(ReadFASTQ):
    """
    FASTQ reader for costimulatory domain targeted sequencing.

    Parses FASTQ files containing costimulatory domain sequences, matching
    reads to a whitelist of known costim sequences and aggregating counts
    per sample.

    Parameters
    ----------
    fastq_path : PathLike or str
        Directory containing FASTQ files.
    feature_metadata_fn : PathLike or str
        Excel file with costim metadata. Must contain columns:
        - 'ID': Candidate identifier (e.g., 'GENE-1')
        - 'Costim': Full costim DNA sequence
    sample_metadata_fn : PathLike or str, optional
        Excel file with sample metadata. If None, metadata is inferred
        from FASTQ filenames using fastq_encoding.
    debug : bool, default False
        Enable debug mode (limits reads, extra logging).
    downstream_const_seq : str, default COSTIM_LEFT
        Constant sequence downstream of costim variable region.
    upstream_const_seq : str, default COSTIM_RIGHT
        Constant sequence upstream of costim variable region.
    fastq_encoding : {'PT_EC_RP_SS', 'PT_EC_RP_EX'}, default 'PT_EC_RP_SS'
        Filename encoding scheme for inferring sample metadata.
    fastq_type : {'gDNA', 'mRNA'}, default 'gDNA'
        Type of sequencing data.
    n_char_features : int, default 50
        Number of costim bases to use for matching.
    max_feature_length : int, default 60
        Maximum length of costim sequences to consider.
    num_cores : int, optional
        Number of CPU cores for parallel processing.
    read_error_threshold : int, default 2
        Maximum edit distance for fuzzy matching.

    Attributes
    ----------
    ambiguous_sequences : list of dict
        Records of truncated sequences that map to multiple features.

    Examples
    --------
    >>> reader = ReadCostimFASTQ(
    ...     fastq_path='/path/to/fastq',
    ...     feature_metadata_fn='/path/to/costim_metadata.xlsx',
    ...     sample_metadata_fn='/path/to/sample_metadata.xlsx',
    ... )
    >>> reader.read()
    >>> reader.serialize('/path/to/results')
    """

    def __init__(
        self,
        fastq_path: Union[PathLike, str],
        feature_metadata_fn: Union[PathLike, str],
        sample_metadata_fn: Optional[Union[PathLike, str]] = None,
        debug: bool = False,
        downstream_const_seq: str = COSTIM_LEFT,
        upstream_const_seq: str = COSTIM_RIGHT,
        fastq_encoding: Literal['PT_EC_RP_SS', 'PT_EC_RP_EX'] = 'PT_EC_RP_SS',
        fastq_type: Literal['gDNA', 'mRNA'] = 'gDNA',
        n_char_features: int = 50,
        max_feature_length: int = 60,
        num_cores: Optional[int] = None,
        read_error_threshold: int = 2,
    ):
        # Store costim-specific parameters before calling super().__init__
        self._feature_metadata_fn = feature_metadata_fn
        self._sample_metadata_fn = sample_metadata_fn
        self._fastq_encoding = fastq_encoding
        self.max_feature_length = max_feature_length
        self.ambiguous_sequences: List[Dict] = []

        # Validate feature metadata exists
        if not Path(feature_metadata_fn).exists():
            raise FileNotFoundError(f"Feature metadata not found: {feature_metadata_fn}")

        # Validate sample metadata if provided
        if sample_metadata_fn is not None and not Path(sample_metadata_fn).exists():
            raise FileNotFoundError(f"Sample metadata not found: {sample_metadata_fn}")

        # Call parent constructor
        super().__init__(
            fastq_path=fastq_path,
            feature_metadata_fn=feature_metadata_fn,
            sample_metadata_fn=sample_metadata_fn,
            debug=debug,
            downstream_const_seq=downstream_const_seq,
            upstream_const_seq=upstream_const_seq,
            fastq_encoding=fastq_encoding,
            fastq_type=fastq_type,
            n_char_features=min(n_char_features, max_feature_length),
            num_cores=num_cores,
            read_error_threshold=read_error_threshold,
        )

    def _parse_feature_metadata(
        self,
        feature_metadata_fn: Optional[Union[PathLike, str]] = None,
    ) -> None:
        """
        Parse costim feature metadata and build sequence whitelist.

        Reads the feature metadata Excel file, truncates costim sequences,
        and builds the mapping from truncated sequence to feature ID.
        Handles ambiguous sequences (same truncation maps to multiple features).
        """
        if feature_metadata_fn is None:
            feature_metadata_fn = self._feature_metadata_fn

        # Load metadata
        df = pd.read_excel(
            feature_metadata_fn,
            sheet_name=0,
            dtype=str,
            header=0,
            engine='openpyxl',
        )

        # Validate required columns
        required_cols = {'ID', 'Costim'}
        missing = required_cols - set(df.columns)
        if missing:
            raise FASTQParseError(f"Feature metadata missing columns: {missing}")

        # Parse additional info from ID (e.g., 'GENE-1' -> Num='1')
        df['Num'] = df['ID'].apply(lambda x: x.rsplit('-', 1)[-1] if '-' in x else '')

        # Sort by ID
        df = df.sort_values('ID').reset_index(drop=True)

        logger.info(f"Feature metadata contains {len(df)} costim domains")

        # Build whitelist and sequence-to-feature mapping
        feature_blacklist = []
        feature_whitelist = []
        seq_to_feature = {}

        for _, row in df.iterrows():
            costim_seq = row['Costim']
            feature_id = row['ID']

            # Truncate sequence based on anchor position
            n_trunc = min(len(costim_seq), self._n_char_features)

            if self._downstream_const_seq is not None:
                # Take first n_trunc bases (downstream anchor means we read left-to-right)
                truncated = costim_seq[:n_trunc].lower()
            elif self._upstream_const_seq is not None:
                # Take last n_trunc bases (upstream anchor means we read right-to-left)
                truncated = costim_seq[-n_trunc:].lower()
            else:
                truncated = costim_seq[:n_trunc].lower()

            # Check for ambiguous mapping
            if truncated in seq_to_feature:
                old_feature = seq_to_feature[truncated]
                logger.warning(
                    f"Ambiguous sequence '{truncated}' maps to both "
                    f"'{old_feature}' and '{feature_id}'"
                )
                self.ambiguous_sequences.append({
                    'sequence': truncated,
                    'feature_old': old_feature,
                    'feature_new': feature_id,
                })
                # Remove from mapping and add to blacklist
                del seq_to_feature[truncated]
                feature_blacklist.append(truncated)
            else:
                seq_to_feature[truncated] = feature_id

        # Build final whitelist (sequences not in blacklist)
        feature_whitelist = list(seq_to_feature.keys())

        logger.info(f"Whitelist: {len(feature_whitelist)} unambiguous sequences")
        logger.info(f"Blacklist: {len(feature_blacklist)} ambiguous sequences")

        # Store results
        self.feature_metadata_df = df
        self._feature_whitelist = feature_whitelist
        self._feature_blacklist = feature_blacklist
        self._seq_to_feature = seq_to_feature

    def _parse_sample_metadata(
        self,
        sample_metadata_fn: Optional[Union[PathLike, str]] = None,
        fastq_encoding: Optional[str] = None,
    ) -> None:
        """
        Parse or infer sample metadata.

        If sample_metadata_fn is provided, loads from Excel.
        Otherwise, infers metadata from FASTQ filenames using the
        encoding scheme.
        """
        if sample_metadata_fn is None:
            sample_metadata_fn = self._sample_metadata_fn

        if fastq_encoding is None:
            fastq_encoding = self._fastq_encoding

        if sample_metadata_fn is None:
            # Infer metadata from filenames
            self._infer_sample_metadata(fastq_encoding)
        else:
            # Load from file
            self._load_sample_metadata(sample_metadata_fn)

    def _infer_sample_metadata(self, fastq_encoding: str) -> None:
        """Infer sample metadata from FASTQ filenames."""
        # Get encoding scheme
        if fastq_encoding == 'PT_EC_RP_SS':
            encoding_dict = FASTQ_NAMING_SCHEME_PT_EC_RP_SS
            logger.info("Using Patient_ExpCond_Replicate_Tsubset encoding")
        elif fastq_encoding == 'PT_EC_RP_EX':
            encoding_dict = FASTQ_NAMING_SCHEME_PT_EC_RP_EX
            logger.info("Using Patient_ExpCond_Replicate_SortCat encoding")
        else:
            raise ValueError(f"Unknown FASTQ encoding: {fastq_encoding}")

        # Build metadata from filenames
        metadata = {
            'Sample': [],
            'R1_Path': [],
        }
        for field in encoding_dict.values():
            metadata[field] = []

        for r1_path in self._r1_file_list:
            filename = basename(r1_path).replace('_R1.fastq.gz', '')
            parts = filename.split('_')

            metadata['Sample'].append(filename)
            metadata['R1_Path'].append(r1_path)

            for idx, field in encoding_dict.items():
                if idx < len(parts):
                    metadata[field].append(parts[idx])
                else:
                    metadata[field].append('')

        df = pd.DataFrame(metadata)
        df = df.sort_values(list(encoding_dict.values())).reset_index(drop=True)

        self.sample_metadata_df = df
        self._r1_path_to_sample = dict(zip(df['R1_Path'], df['Sample']))

    def _load_sample_metadata(self, sample_metadata_fn: Union[PathLike, str]) -> None:
        """Load sample metadata from Excel file."""
        df = pd.read_excel(
            sample_metadata_fn,
            sheet_name=0,
            dtype=str,
            header=0,
            engine='openpyxl',
        )

        # Get sample names from R1 filenames
        sample_names = [
            basename(r1).replace('_R1.fastq.gz', '')
            for r1 in self._r1_file_list
        ]

        # Check all samples are in metadata
        if 'ID' in df.columns:
            metadata_ids = set(df['ID'].tolist())
            missing = [s for s in sample_names if s not in metadata_ids]
            if missing:
                logger.warning(f"Samples not in metadata: {missing[:5]}...")

            # Filter to found samples
            df = df[df['ID'].isin(sample_names)].copy()

            # Generate abbreviated sample name
            name_cols = ['Hlthy Vol', 'Exp Cond', 'T subset', 'PD-1 Status', 'Repl']
            available_cols = [c for c in name_cols if c in df.columns]
            if available_cols:
                df['Name'] = df[available_cols].apply(lambda row: '_'.join(row), axis=1)
            else:
                df['Name'] = df['ID']

            # Build path-to-name mapping
            id_to_name = dict(zip(df['ID'], df['Name']))
            self._r1_path_to_sample = {
                r1_path: id_to_name.get(
                    basename(r1_path).replace('_R1.fastq.gz', ''),
                    'Unknown'
                )
                for r1_path in self._r1_file_list
            }
        else:
            # Fallback: use filename as sample name
            self._r1_path_to_sample = {
                r1: basename(r1).replace('_R1.fastq.gz', '')
                for r1 in self._r1_file_list
            }

        # Sort metadata
        sort_cols = ['Hlthy Vol', 'Exp Cond', 'Repl', 'T subset']
        available_sort = [c for c in sort_cols if c in df.columns]
        if available_sort:
            df = df.sort_values(available_sort).reset_index(drop=True)

        self.sample_metadata_df = df

    def serialize(
        self,
        results_path: Union[PathLike, str],
        format: Literal['excel', 'csv'] = 'excel',
    ) -> None:
        """
        Save results to disk, including ambiguous sequence report.

        Parameters
        ----------
        results_path : PathLike or str
            Directory to save results.
        format : {'excel', 'csv'}, default 'excel'
            Output format.
        """
        # Call parent serialize
        super().serialize(results_path, format=format)

        # Save ambiguous sequences report
        if self.ambiguous_sequences:
            amb_df = pd.DataFrame(self.ambiguous_sequences)
            results_path = Path(results_path)
            if format == 'excel':
                amb_df.to_excel(results_path / 'ambiguous_costim_report.xlsx', index=False)
            else:
                amb_df.to_csv(results_path / 'ambiguous_costim_report.csv', index=False)
            logger.info(f"Saved {len(self.ambiguous_sequences)} ambiguous sequence records")


def main():
    """Command-line interface for ReadCostimFASTQ."""
    parser = argparse.ArgumentParser(
        description='Parse costimulatory domain targeted sequencing FASTQ files'
    )

    parser.add_argument(
        '-f', '--fastq_path',
        required=True,
        help='Path to directory containing FASTQ files',
    )
    parser.add_argument(
        '--feature_metadata_fn',
        required=True,
        help='Path to costim metadata Excel file',
    )
    parser.add_argument(
        '--sample_metadata_fn',
        default=None,
        help='Path to sample metadata Excel file (optional)',
    )
    parser.add_argument(
        '--results_path',
        required=True,
        help='Path to output directory',
    )
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help='Enable debug mode',
    )
    parser.add_argument(
        '--fastq_encoding',
        choices=['PT_EC_RP_SS', 'PT_EC_RP_EX'],
        default='PT_EC_RP_SS',
        help='FASTQ filename encoding scheme',
    )
    parser.add_argument(
        '--fastq_type',
        choices=['gDNA', 'mRNA'],
        default='gDNA',
        help='Type of sequencing data',
    )
    parser.add_argument(
        '--n_char_truncate',
        type=int,
        default=30,
        help='Number of characters to truncate features',
    )
    parser.add_argument(
        '--num_cores',
        type=int,
        default=None,
        help='Number of CPU cores',
    )
    parser.add_argument(
        '--read_error_threshold',
        type=int,
        default=2,
        help='Maximum edit distance for fuzzy matching',
    )
    parser.add_argument(
        '--format',
        choices=['excel', 'csv'],
        default='excel',
        help='Output format',
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    # Create reader and process
    reader = ReadCostimFASTQ(
        fastq_path=args.fastq_path,
        feature_metadata_fn=args.feature_metadata_fn,
        sample_metadata_fn=args.sample_metadata_fn,
        debug=args.debug,
        fastq_encoding=args.fastq_encoding,
        fastq_type=args.fastq_type,
        n_char_features=args.n_char_truncate,
        num_cores=args.num_cores,
        read_error_threshold=args.read_error_threshold,
    )

    reader.read()
    reader.serialize(args.results_path, format=args.format)
    reader.print_summary()


if __name__ == '__main__':
    main()
