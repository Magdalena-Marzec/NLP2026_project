#!/usr/bin/env python3
"""
CLI driver for RNA-seq normalization and transformation pipeline.

The script executes independent processing steps from normalization_utils and
saves each stage to a file with a shared output prefix.
"""

from __future__ import annotations

import argparse
import sys

from normalization_utils import (
    apply_global_minmax,
    apply_scaling,
    apply_vst,
    calculate_size_factors,
    count_biological_zero_genes,
    filter_genes,
    load_condition_metadata,
    load_counts_matrix,
    save_counts_matrix,
)


def parse_arguments() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(description="RNA-seq count normalization pipeline.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to raw counts matrix (.csv, .tsv, or .txt).",
    )
    parser.add_argument(
        "--condition-file",
        required=True,
        help="Path to condition.txt (one condition per line, same order as sample columns).",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=10,
        help="Filtering threshold X for replicate rule (default: 10).",
    )
    parser.add_argument(
        "--min-replicates",
        type=int,
        default=4,
        help="Minimum sample replicates R for filtering (default: 4).",
    )
    parser.add_argument(
        "--scale-to-unity",
        action="store_true",
        help="Apply global min-max scaling to [0, 1] on VST output.",
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Prefix for output artifacts.",
    )
    return parser.parse_args()


def _output_path(prefix: str, suffix: str) -> str:
    """Build output file path from prefix and suffix."""
    return f"{prefix}_{suffix}.tsv"


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the complete pipeline in strict step order.

    Args:
        args: Parsed CLI arguments.
    """
    print(f"Loading count matrix: {args.input}")
    raw_counts = load_counts_matrix(args.input)

    total_genes = raw_counts.shape[0]
    print(f"Detected genes: {total_genes}")

    biological_zero_count = count_biological_zero_genes(raw_counts, args.min_count)
    print(
        "Biological zeros detected: "
        f"{biological_zero_count} "
        "(genes with zero in some samples and high expression in others)"
    )

    # Apply N-replicate filtering and derive rejected noise-zero count.
    filtered_counts, replicate_mask = filter_genes(
        raw_counts, min_count=args.min_count, min_replicates=args.min_replicates
    )
    noise_zero_rejected = int((~replicate_mask).sum())
    print(f"Rejected as noise zeros: {noise_zero_rejected}")
    print(f"Genes retained after filtering: {filtered_counts.shape[0]}")

    filtered_path = _output_path(args.output_prefix, "filtered")
    save_counts_matrix(filtered_counts, filtered_path)
    print(f"Saved filtered counts: {filtered_path}")

    size_factors = calculate_size_factors(filtered_counts)
    scaled_counts = apply_scaling(filtered_counts, size_factors)
    scaled_path = _output_path(args.output_prefix, "scaled")
    save_counts_matrix(scaled_counts, scaled_path)
    print(f"Saved scaled counts: {scaled_path}")

    metadata = load_condition_metadata(args.condition_file, filtered_counts.columns)
    vst_counts = apply_vst(filtered_counts, metadata)
    vst_path = _output_path(args.output_prefix, "vst")
    save_counts_matrix(vst_counts, vst_path)
    print(f"Saved VST matrix: {vst_path}")

    if args.scale_to_unity:
        final_01 = apply_global_minmax(vst_counts)
        final_path = _output_path(args.output_prefix, "final_01")
        save_counts_matrix(final_01, final_path)
        print(f"Saved [0,1]-scaled matrix: {final_path}")

    print("Pipeline completed successfully.")


def main() -> None:
    """Run the pipeline and convert known failures into CLI errors."""
    args = parse_arguments()
    if not args.condition_file:
        raise ValueError("--condition-file is required.")
    try:
        run_pipeline(args)
    except (ValueError, OSError) as error:
        print(f"Pipeline error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
