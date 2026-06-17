#!/usr/bin/env python3
"""
Integer vocabulary for the genomic expression Transformer.

Maps special tokens (PAD, UNK), nucleotide characters, and dataset-specific
tissue tokens to contiguous integer IDs used by the embedding layer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Vocabulary:
    """Bidirectional token-to-id mapping with padding metadata.

    Attributes:
        token_to_id: Maps token strings to integer indices.
        id_to_token: Inverse mapping from indices to token strings.
        pad_id: Integer ID of the padding token.
        vocab_size: Total number of tokens in the vocabulary.
    """

    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    pad_id: int
    vocab_size: int


def build_tissue_token_names(sorted_tissues: list[str]) -> list[str]:
    """Build tissue placeholder token names in 1-based index order.

    Args:
        sorted_tissues: Alphabetically sorted tissue names from the dataset.

    Returns:
        List of strings like ``[TISSUE_1]``, ``[TISSUE_2]``, ...
    """
    names: list[str] = []
    for i in range(1, len(sorted_tissues) + 1):
        names.append(f"[TISSUE_{i}]")
    return names


def build_vocabulary(unique_tissues: list[str], config: dict) -> Vocabulary:
    """Construct the full token vocabulary from config and dataset tissues.

    Token order: PAD, UNK, nucleotides (A/C/T/G), then tissue tokens.

    Args:
        unique_tissues: Tissue names observed in the dataset (any order).
        config: Hyperparameter dict containing ``vocab_pad``, ``vocab_unk``,
            and ``nucleotides``.

    Returns:
        Populated ``Vocabulary`` instance.
    """
    pad_token = config["vocab_pad"]
    unk_token = config["vocab_unk"]
    nucleotides = config["nucleotides"]

    sorted_tissues = sorted(unique_tissues)
    tissue_tokens = build_tissue_token_names(sorted_tissues)

    all_tokens = [pad_token, unk_token] + nucleotides + tissue_tokens
    token_to_id = {tok: idx for idx, tok in enumerate(all_tokens)}
    id_to_token = {idx: tok for tok, idx in token_to_id.items()}
    pad_id = token_to_id[pad_token]

    return Vocabulary(
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        pad_id=pad_id,
        vocab_size=len(all_tokens),
    )


def encode_dna_sequence(
    sequence: str, vocabulary: Vocabulary, config: dict
) -> list[int]:
    """Convert a DNA string into a list of nucleotide integer token IDs.

    Unknown characters are mapped to the UNK token (``N``).

    Args:
        sequence: Raw DNA sequence (promoter + UTR concatenation).
        vocabulary: Built vocabulary with nucleotide and UNK entries.
        config: Hyperparameter dict containing ``vocab_unk`` and ``nucleotides``.

    Returns:
        List of integer token IDs, one per base.
    """
    unk_token = config["vocab_unk"]
    unk_id = vocabulary.token_to_id[unk_token]
    nucleotide_set = set(config["nucleotides"])

    ids: list[int] = []
    for ch in sequence.upper():
        if ch in vocabulary.token_to_id and ch in nucleotide_set:
            ids.append(vocabulary.token_to_id[ch])
        else:
            ids.append(unk_id)
    return ids


def encode_tissue(
    tissue_name: str, vocabulary: Vocabulary, sorted_tissues: list[str]
) -> int:
    """Map a tissue name to its dedicated ``[TISSUE_i]`` integer token ID.

    Args:
        tissue_name: Tissue label from the dataset row.
        vocabulary: Built vocabulary containing tissue tokens.
        sorted_tissues: Alphabetically sorted tissue list used during vocab build.

    Returns:
        Integer token ID for the tissue.

    Raises:
        ValueError: If ``tissue_name`` is not present in ``sorted_tissues``.
    """
    try:
        tissue_index = sorted_tissues.index(tissue_name)
    except ValueError as exc:
        raise ValueError(f"Unknown tissue name: {tissue_name}") from exc

    tissue_token = f"[TISSUE_{tissue_index + 1}]"
    return vocabulary.token_to_id[tissue_token]
