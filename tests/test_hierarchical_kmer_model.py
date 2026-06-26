#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from model import ExpressionTransformer, number_of_chunks  # noqa: E402


def make_config(
    *,
    max_seq_len: int = 2196,
    dropout: float = 0.0,
) -> dict:
    return {
        "d_model": 32,
        "n_heads": 4,
        "n_layers": 4,
        "local_n_layers": 1,
        "global_n_layers": 1,
        "d_mlp": 64,
        "dropout": dropout,
        "max_seq_len": max_seq_len,
        "chunk_size_tokens": 32,
        "chunk_stride_tokens": 24,
    }


def make_kmer_batch(
    raw_dna_lengths: list[int],
    *,
    kmer_size: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create [tissue] + overlapping stride-1 k-mer token IDs."""
    pad_id = 0
    tissue_id = 4098
    token_lengths = [
        max(0, raw_length - kmer_size + 1)
        for raw_length in raw_dna_lengths
    ]
    max_token_length = max(token_lengths, default=0)

    tokens = torch.full(
        (len(raw_dna_lengths), max_token_length + 1),
        fill_value=pad_id,
        dtype=torch.long,
    )
    tokens[:, 0] = tissue_id

    for row, token_length in enumerate(token_lengths):
        if token_length:
            tokens[row, 1 : token_length + 1] = torch.randint(
                low=1,
                high=4097,
                size=(token_length,),
                dtype=torch.long,
            )

    return tokens, tokens.eq(pad_id)


@pytest.mark.parametrize(
    ("token_length", "expected"),
    [
        (0, 0),
        (1, 1),
        (32, 1),
        (33, 2),
        (495, 21),   # 500 nt with overlapping 6-mers
        (795, 33),   # 800 nt
        (2195, 92),  # 2200 nt
    ],
)
def test_number_of_chunks(token_length: int, expected: int) -> None:
    assert number_of_chunks(token_length, 32, 24) == expected


def test_2200_nt_forward_uses_92_chunks() -> None:
    model = ExpressionTransformer(
        vocab_size=4104,
        pad_id=0,
        config=make_config(max_seq_len=2196),
    )
    tokens, padding_mask = make_kmer_batch([2200])

    prediction, aux = model(tokens, padding_mask, return_aux=True)

    assert tokens.shape == (1, 2196)
    assert prediction.shape == (1,)
    assert aux["chunk_representations"].shape == (1, 92, 32)
    assert aux["valid_chunk_mask"].sum().item() == 92
    assert model.max_chunks == 92


def test_mixed_length_batch_has_per_sample_chunk_counts() -> None:
    model = ExpressionTransformer(
        vocab_size=4104,
        pad_id=0,
        config=make_config(),
    )
    tokens, padding_mask = make_kmer_batch([500, 800, 2200])

    _, aux = model(tokens, padding_mask, return_aux=True)

    assert aux["required_chunk_counts"].tolist() == [21, 33, 92]
    assert aux["valid_chunk_mask"].sum(dim=1).tolist() == [21, 33, 92]


def test_prediction_does_not_depend_on_longer_batch_member() -> None:
    torch.manual_seed(7)
    model = ExpressionTransformer(
        vocab_size=4104,
        pad_id=0,
        config=make_config(dropout=0.0),
    )
    model.eval()

    short_tokens, short_mask = make_kmer_batch([500])
    mixed_tokens, mixed_mask = make_kmer_batch([500, 2200])
    mixed_tokens[0, : short_tokens.shape[1]] = short_tokens[0]

    with torch.no_grad():
        prediction_alone = model(short_tokens, short_mask)[0]
        prediction_mixed = model(mixed_tokens, mixed_mask)[0]

    torch.testing.assert_close(
        prediction_alone,
        prediction_mixed,
        rtol=1e-5,
        atol=1e-6,
    )


def test_zero_complete_kmers_falls_back_to_tissue_only() -> None:
    model = ExpressionTransformer(
        vocab_size=4104,
        pad_id=0,
        config=make_config(),
    )
    tokens, padding_mask = make_kmer_batch([4, 5])

    prediction, aux = model(tokens, padding_mask, return_aux=True)

    assert tokens.shape == (2, 1)
    assert prediction.shape == (2,)
    assert aux["chunk_representations"].shape == (2, 0, 32)
    assert aux["valid_chunk_mask"].shape == (2, 0)


def test_backward_reaches_local_and_global_encoders() -> None:
    model = ExpressionTransformer(
        vocab_size=4104,
        pad_id=0,
        config=make_config(),
    )
    tokens, padding_mask = make_kmer_batch([800, 800])
    target = torch.randn(2)

    prediction = model(tokens, padding_mask)
    loss = torch.nn.functional.mse_loss(prediction, target)
    loss.backward()

    assert model.chunk_token.grad is not None
    assert model.token_embedding.weight.grad is not None
    assert model.local_blocks[0].attention.in_proj_weight.grad is not None
    assert model.global_blocks[0].attention.in_proj_weight.grad is not None


def test_eight_token_overlap_covers_every_three_kmer_motif_span() -> None:
    """An 8-nt motif is represented by three consecutive 6-mer tokens."""
    token_length = 2195
    chunk_size = 32
    stride = 24
    starts = range(0, token_length, stride)

    for motif_start in range(token_length - 3 + 1):
        motif_end = motif_start + 3
        assert any(
            chunk_start <= motif_start
            and motif_end <= chunk_start + chunk_size
            for chunk_start in starts
        )


def test_rejects_input_beyond_configured_token_budget() -> None:
    model = ExpressionTransformer(
        vocab_size=4104,
        pad_id=0,
        config=make_config(max_seq_len=100),
    )
    tokens, padding_mask = make_kmer_batch([200])

    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        model(tokens, padding_mask)
