#!/usr/bin/env python3
"""
Data loading utilities for the genomic expression Transformer pipeline.

Handles TSV/CSV ingestion, comment-line skipping, and column validation
for expression datasets with tissue and sequence fields.
"""

from __future__ import annotations

import pandas as pd

from normalization_utils import _infer_separator

REQUIRED_COLUMNS: list[str] = [
    "tissue",
    "promoter_sequence",
    "utr_5_sequence",
    "vst_expression",
]


def load_expression_dataframe(data_path: str) -> pd.DataFrame:
    """Load and validate an expression dataset from a TSV or CSV file.

    Comment lines starting with ``#`` are skipped. Rows with missing required
    fields are dropped.

    Args:
        data_path: Path to the input dataset file.

    Returns:
        Cleaned DataFrame with required columns and float ``vst_expression``.

    Raises:
        ValueError: If required columns are missing.
    """
    separator = _infer_separator(data_path)
    df = pd.read_csv(data_path, sep=separator, comment="#")
    df.columns = df.columns.str.strip()

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=REQUIRED_COLUMNS)
    df["vst_expression"] = df["vst_expression"].astype(float)
    return df


def extract_unique_tissues(df: pd.DataFrame) -> list[str]:
    """Extract alphabetically sorted unique tissue labels from the dataset.

    Args:
        df: Expression DataFrame containing a ``tissue`` column.

    Returns:
        Sorted list of unique tissue name strings.
    """
    tissues = df["tissue"].astype(str).unique().tolist()
    return sorted(tissues)
