#!/usr/bin/env python3
"""
RNA-seq count matrix normalization utilities.

Each public function implements one independent mathematical step from the
normalization specification. Shared parsing and validation logic is isolated
in helper functions to keep the pipeline DRY.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet


def _validate_integer_counts(counts_df: pd.DataFrame) -> None:
    """Validate that all count values are numeric non-negative integers.

    Args:
        counts_df: Gene-by-sample count matrix.

    Raises:
        ValueError: If values are non-numeric, negative, or non-integer.
    """
    numeric = counts_df.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("Counts matrix contains non-numeric values.")
    if (numeric < 0).any().any():
        raise ValueError("Counts matrix contains negative values.")
    if not np.allclose(np.mod(numeric.to_numpy(dtype=float), 1.0), 0.0):
        raise ValueError("Counts matrix contains non-integer values.")


def _infer_separator(input_path: str) -> str:
    """Infer input delimiter from file extension.

    Args:
        input_path: Path to counts matrix file.

    Returns:
        A delimiter string for pandas read/write.

    Raises:
        ValueError: If extension is not supported.
    """
    suffix = Path(input_path).suffix.lower()
    if suffix == ".csv":
        return ","
    if suffix in {".tsv", ".txt"}:
        return "\t"
    raise ValueError("Unsupported input extension. Use .csv, .tsv, or .txt.")


def load_counts_matrix(input_path: str) -> pd.DataFrame:
    """Load raw counts matrix into gene-by-sample DataFrame.

    Args:
        input_path: Path to counts matrix file with gene_id in first column.

    Returns:
        DataFrame indexed by gene IDs with sample columns of integer counts.
    """
    separator = _infer_separator(input_path)
    matrix_df = pd.read_csv(input_path, sep=separator, index_col=0)
    matrix_df = matrix_df.dropna(axis=1, how="all")
    _validate_integer_counts(matrix_df)
    return matrix_df.astype(np.int64)


def save_counts_matrix(counts_df: pd.DataFrame, output_path: str) -> None:
    """Persist matrix to TSV with gene index included.

    Args:
        counts_df: Matrix to write.
        output_path: Destination file path.
    """
    counts_df.to_csv(output_path, sep="\t", index=True)


def count_biological_zero_genes(counts_df: pd.DataFrame, min_count: int) -> int:
    """Count genes silent in some samples but high in others.

    Args:
        counts_df: Raw counts matrix, genes x samples.
        min_count: Threshold defining a high-expression event.

    Returns:
        Number of biological-zero genes.
    """
    has_zero = (counts_df == 0).any(axis=1)
    has_high = (counts_df >= min_count).any(axis=1)
    biological_mask = has_zero & has_high
    return int(biological_mask.sum())


def filter_genes(
    counts_df: pd.DataFrame, min_count: int, min_replicates: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Filter genes with the N-replicate rule.

    Args:
        counts_df: Raw integer counts matrix, genes x samples.
        min_count: Minimum count threshold X.
        min_replicates: Minimum sample count meeting X (R).

    Returns:
        A tuple of (filtered counts DataFrame, pass mask by gene).
    """
    replicate_mask = (counts_df >= min_count).sum(axis=1) >= min_replicates
    filtered_df = counts_df.loc[replicate_mask]
    return filtered_df, replicate_mask


def _geometric_mean_per_gene(counts_df: pd.DataFrame) -> pd.Series:
    """Compute DESeq2-style gene geometric means.

    Args:
        counts_df: Gene-by-sample integer counts.

    Returns:
        Series of geometric means indexed by gene ID.
    """
    values = counts_df.to_numpy(dtype=float)
    has_zero = (values == 0).any(axis=1)
    safe_values = np.where(values > 0, values, 1.0)
    geometric = np.exp(np.mean(np.log(safe_values), axis=1))
    geometric[has_zero] = 0.0
    return pd.Series(geometric, index=counts_df.index, dtype=float)


def calculate_size_factors(counts_df: pd.DataFrame) -> pd.Series:
    """Calculate sample size factors with median-of-ratios.

    Args:
        counts_df: Filtered raw integer count matrix.

    Returns:
        Sample-wise size factors indexed by sample name.

    Raises:
        ValueError: If no valid genes remain or any size factor is non-positive.
    """
    geo_means = _geometric_mean_per_gene(counts_df)
    valid_genes = geo_means[geo_means > 0].index
    if len(valid_genes) == 0:
        raise ValueError("No genes left for size-factor estimation.")

    subset = counts_df.loc[valid_genes]
    geo_subset = geo_means.loc[valid_genes]
    ratios = subset.div(geo_subset, axis=0)
    size_factors = ratios.median(axis=0)

    if (size_factors <= 0).any():
        raise ValueError("Computed a non-positive size factor.")
    return size_factors


def apply_scaling(counts_df: pd.DataFrame, size_factors: pd.Series) -> pd.DataFrame:
    """Scale counts by dividing each sample by its size factor.

    Args:
        counts_df: Raw or filtered count matrix.
        size_factors: Sample-wise size factors.

    Returns:
        Float DataFrame of scaled counts.

    Raises:
        ValueError: If size factors are missing for matrix columns.
    """
    aligned = size_factors.reindex(counts_df.columns)
    if aligned.isna().any():
        raise ValueError("Size factors missing for one or more sample columns.")
    return counts_df.div(aligned, axis=1)


def load_condition_metadata(condition_path: str, sample_names: pd.Index) -> pd.DataFrame:
    """Load line-based condition metadata and align to sample names.

    Args:
        condition_path: Path to condition file (one condition per line).
        sample_names: Expected sample order from count matrix columns.

    Returns:
        Metadata DataFrame with one column `condition`.

    Raises:
        ValueError: If file is empty or line count mismatches sample count.
    """
    with open(condition_path, "r", encoding="utf-8") as cond_file:
        conditions = [line.strip() for line in cond_file if line.strip()]

    if not conditions:
        raise ValueError("Condition metadata file is empty.")
    if len(conditions) != len(sample_names):
        raise ValueError(
            "Condition count does not match number of sample columns: "
            f"{len(conditions)} vs {len(sample_names)}."
        )
    return pd.DataFrame({"condition": conditions}, index=sample_names)


def _counts_genes_by_samples_to_pydeseq2(counts_df: pd.DataFrame) -> pd.DataFrame:
    """Transpose matrix to samples-by-genes format."""
    return counts_df.T


def _counts_pydeseq2_to_genes_by_samples(counts_df: pd.DataFrame) -> pd.DataFrame:
    """Transpose matrix back to genes-by-samples format."""
    return counts_df.T


def apply_vst(filtered_counts_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Apply pydeseq2 variance-stabilizing transformation.

    Args:
        filtered_counts_df: Filtered raw integer counts in genes x samples.
        metadata_df: Sample metadata with `condition` column and sample index.

    Returns:
        VST-transformed matrix in genes x samples orientation.

    Raises:
        ValueError: If metadata is missing required fields or sample alignment.
    """
    _validate_integer_counts(filtered_counts_df)
    if "condition" not in metadata_df.columns:
        raise ValueError("Metadata must contain a 'condition' column.")

    # Convert to pydeseq2 orientation and enforce integer type.
    counts_for_dds = _counts_genes_by_samples_to_pydeseq2(filtered_counts_df).astype(np.int64)

    # Reindex metadata to counts order and fail if any sample is unmatched.
    aligned_metadata = metadata_df.reindex(counts_for_dds.index)
    if aligned_metadata.isna().any().any():
        raise ValueError("Condition metadata is missing one or more samples.")

    # Fit DESeq2 model components and compute VST output layer.
    dds = DeseqDataSet(
        counts=counts_for_dds,
        metadata=aligned_metadata,
        design_factors="condition",
        quiet=False,
        low_memory=True,
        n_cpus=4
    )
    dds.deseq2()
    dds.vst()

    vst_samples_by_genes = dds.layers["vst_counts"]
    if not isinstance(vst_samples_by_genes, pd.DataFrame):
        vst_samples_by_genes = pd.DataFrame(
            vst_samples_by_genes, index=counts_for_dds.index, columns=counts_for_dds.columns
        )
    return _counts_pydeseq2_to_genes_by_samples(vst_samples_by_genes)


def apply_global_minmax(vst_df: pd.DataFrame) -> pd.DataFrame:
    """Scale full matrix to [0, 1] using global minimum and maximum.

    Args:
        vst_df: VST-transformed matrix.

    Returns:
        Min-max normalized matrix.

    Raises:
        ValueError: If all matrix values are constant.
    """
    v_min = float(vst_df.to_numpy().min())
    v_max = float(vst_df.to_numpy().max())
    denominator = v_max - v_min
    if denominator == 0.0:
        raise ValueError("Cannot min-max scale a constant matrix.")
    return (vst_df - v_min) / denominator
