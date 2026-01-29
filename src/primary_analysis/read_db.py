"""
Read database for accumulating counts from FASTQ parsing.

Provides in-memory storage for feature counts across samples during
parallel FASTQ processing.
"""

from typing import List, Optional

import pandas as pd


class ReadDB:
    """
    In-memory database for accumulating read counts.

    Stores counts of feature occurrences per sample during FASTQ parsing.
    Thread-safe for use in multiprocessing contexts (each worker gets its own instance).

    Parameters
    ----------
    feature_whitelist : list of str, optional
        Pre-defined list of valid features. If provided, only these features
        will be counted.
    sample_list : list of str, optional
        Pre-defined list of sample names. If provided, initializes count
        structure for these samples.

    Examples
    --------
    >>> db = ReadDB(feature_whitelist=['GENE1', 'GENE2'], sample_list=['Sample1'])
    >>> db.increment_count('GENE1', 'Sample1')
    >>> db.increment_count('GENE1', 'Sample1')
    >>> counts = db.counts()
    >>> counts.loc['GENE1', 'Sample1']
    2
    """

    def __init__(
        self,
        feature_whitelist: Optional[List[str]] = None,
        sample_list: Optional[List[str]] = None,
    ):
        self._feature_whitelist = feature_whitelist or []
        self._sample_list = sample_list or []

        # Initialize count dictionary
        # Structure: {feature: {sample: count}}
        self._counts: dict[str, dict[str, int]] = {}

        # Pre-populate with whitelist features if provided
        for feature in self._feature_whitelist:
            self._counts[feature] = {s: 0 for s in self._sample_list}

    def increment_count(self, feature: str, sample: str, count: int = 1) -> None:
        """
        Increment the count for a feature-sample pair.

        Parameters
        ----------
        feature : str
            Feature identifier (e.g., gene name, barcode).
        sample : str
            Sample identifier.
        count : int, default 1
            Amount to increment by.
        """
        if feature not in self._counts:
            self._counts[feature] = {}

        if sample not in self._counts[feature]:
            self._counts[feature][sample] = 0

        self._counts[feature][sample] += count

    def get_count(self, feature: str, sample: str) -> int:
        """
        Get the count for a feature-sample pair.

        Parameters
        ----------
        feature : str
            Feature identifier.
        sample : str
            Sample identifier.

        Returns
        -------
        int
            Count value, or 0 if not found.
        """
        return self._counts.get(feature, {}).get(sample, 0)

    def counts(self) -> pd.DataFrame:
        """
        Export counts as a DataFrame.

        Returns
        -------
        pd.DataFrame
            Count matrix with features as rows and samples as columns.
            Missing values are filled with 0.
        """
        if not self._counts:
            return pd.DataFrame()

        # Convert nested dict to DataFrame
        df = pd.DataFrame.from_dict(self._counts, orient='index')

        # Fill NaN with 0 and convert to int
        df = df.fillna(0).astype(int)

        # Sort index and columns
        df = df.sort_index(axis=0).sort_index(axis=1)

        return df

    @property
    def n_features(self) -> int:
        """Number of unique features with counts."""
        return len(self._counts)

    @property
    def n_samples(self) -> int:
        """Number of unique samples with counts."""
        all_samples = set()
        for sample_counts in self._counts.values():
            all_samples.update(sample_counts.keys())
        return len(all_samples)

    @property
    def total_counts(self) -> int:
        """Total count across all features and samples."""
        total = 0
        for sample_counts in self._counts.values():
            total += sum(sample_counts.values())
        return total
