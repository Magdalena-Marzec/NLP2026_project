#!/usr/bin/env python3

"""
Hierarchical local-global Transformer for genomic expression regression.

This implementation is designed for the ``kmer-tokenizer-ablation`` branch.
It preserves the existing public interface::

    model = ExpressionTransformer(vocab_size, pad_id, config)
    predictions = model(tokens, key_padding_mask)

Input layout remains unchanged: ``tokens[:, 0]`` is the tissue token and
``tokens[:, 1:]`` contains DNA tokens. In the target branch those DNA tokens
are overlapping 6-mers.

The DNA-token sequence is divided into overlapping local chunks. A learned
``[CHUNK]`` token summarizes each chunk with a local Transformer. A global
Transformer then processes::

    [TISSUE], [CHUNK_1], ..., [CHUNK_N]

The scalar prediction is read from the final tissue representation.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def number_of_chunks(
    token_length: int,
    chunk_size_tokens: int = 32,
    chunk_stride_tokens: int = 24,
) -> int:
    """Return the number of right-padded chunks covering ``token_length``.

    ``token_length`` counts encoded DNA tokens, not raw nucleotides. For the
    branch default (overlapping 6-mers with stride 1), 2200 nt become 2195 DNA
    tokens and therefore 92 chunks for size 32 / stride 24.
    """
    if token_length < 0:
        raise ValueError("token_length must be non-negative.")
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be positive.")
    if chunk_stride_tokens <= 0:
        raise ValueError("chunk_stride_tokens must be positive.")
    if chunk_stride_tokens > chunk_size_tokens:
        raise ValueError(
            "chunk_stride_tokens cannot exceed chunk_size_tokens because "
            "DNA-token positions would be skipped."
        )
    if token_length == 0:
        return 0
    if token_length <= chunk_size_tokens:
        return 1
    return 1 + math.ceil(
        (token_length - chunk_size_tokens) / chunk_stride_tokens
    )


class TransformerBlock(nn.Module):
    """Single Pre-LayerNorm Transformer block."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_mlp: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_mlp, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply self-attention and MLP residual sublayers."""
        norm_x = self.layer_norm1(x)
        attention_output, _ = self.attention(
            norm_x,
            norm_x,
            norm_x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attention_output
        x = x + self.mlp(self.layer_norm2(x))
        return x


class ExpressionTransformer(nn.Module):
    """Tissue-conditioned hierarchical Transformer regressor.

    Additional configuration keys relative to the flat model:

    ``local_n_layers``
        Number of local Transformer blocks. Default: 2.
    ``global_n_layers``
        Number of global Transformer blocks. Default: 2.
    ``chunk_size_tokens``
        Number of encoded DNA tokens in one local chunk. Default: 32.
    ``chunk_stride_tokens``
        Distance between consecutive chunk starts in encoded-token units.
        Default: 24.

    The legacy ``n_layers`` key may remain in the project configuration for
    checkpoint metadata and comparison with the flat model, but this class
    uses the explicit local/global layer counts.
    """

    def __init__(self, vocab_size: int, pad_id: int, config: dict) -> None:
        super().__init__()

        d_model = int(config["d_model"])
        n_heads = int(config["n_heads"])
        d_mlp = int(config["d_mlp"])
        dropout = float(config["dropout"])
        max_seq_len = int(config["max_seq_len"])

        local_n_layers = int(config.get("local_n_layers", 2))
        global_n_layers = int(config.get("global_n_layers", 2))
        chunk_size_tokens = int(config.get("chunk_size_tokens", 32))
        chunk_stride_tokens = int(config.get("chunk_stride_tokens", 24))

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if not 0 <= pad_id < vocab_size:
            raise ValueError("pad_id must be a valid vocabulary index.")
        if d_model <= 0 or d_model % n_heads != 0:
            raise ValueError(
                "d_model must be positive and divisible by n_heads."
            )
        if local_n_layers <= 0 or global_n_layers <= 0:
            raise ValueError(
                "local_n_layers and global_n_layers must be positive."
            )
        if max_seq_len < 1:
            raise ValueError("max_seq_len must allow at least the tissue token.")
        if chunk_size_tokens <= 0 or chunk_stride_tokens <= 0:
            raise ValueError("Chunk size and stride must be positive.")
        if chunk_stride_tokens > chunk_size_tokens:
            raise ValueError(
                "chunk_stride_tokens cannot exceed chunk_size_tokens."
            )

        self.pad_id = int(pad_id)
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_stride_tokens = chunk_stride_tokens
        self.chunk_overlap_tokens = (
            chunk_size_tokens - chunk_stride_tokens
        )

        # max_seq_len includes the tissue token at index 0.
        max_dna_tokens = max_seq_len - 1
        self.max_chunks = number_of_chunks(
            max_dna_tokens,
            chunk_size_tokens=chunk_size_tokens,
            chunk_stride_tokens=chunk_stride_tokens,
        )

        # The vocabulary table remains shared by tissue and k-mer tokens,
        # exactly as in kmer-tokenizer-ablation.
        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_id,
        )

        # Local sequence: [CHUNK] + chunk_size_tokens DNA tokens.
        self.chunk_token = nn.Parameter(torch.empty(1, 1, d_model))
        self.local_positional_embedding = nn.Embedding(
            chunk_size_tokens + 1,
            d_model,
        )
        self.local_blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_mlp, dropout)
                for _ in range(local_n_layers)
            ]
        )
        self.local_final_layer_norm = nn.LayerNorm(d_model)

        # Global sequence: [TISSUE] + one representation per valid chunk.
        self.global_positional_embedding = nn.Embedding(
            self.max_chunks + 1,
            d_model,
        )
        self.global_blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_mlp, dropout)
                for _ in range(global_n_layers)
            ]
        )
        self.global_final_layer_norm = nn.LayerNorm(d_model)

        # Unchanged from the flat branch model.
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self._reset_hierarchical_parameters()

    def _reset_hierarchical_parameters(self) -> None:
        nn.init.normal_(self.chunk_token, mean=0.0, std=0.02)
        nn.init.normal_(
            self.local_positional_embedding.weight,
            mean=0.0,
            std=0.02,
        )
        nn.init.normal_(
            self.global_positional_embedding.weight,
            mean=0.0,
            std=0.02,
        )

    def _validate_inputs(
        self,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> None:
        if tokens.ndim != 2:
            raise ValueError(
                f"tokens must have shape (B, L), got {tuple(tokens.shape)}."
            )
        if tokens.dtype != torch.long:
            raise TypeError("tokens must have dtype torch.long.")
        if key_padding_mask.shape != tokens.shape:
            raise ValueError(
                "key_padding_mask must have the same shape as tokens."
            )
        if key_padding_mask.dtype != torch.bool:
            raise TypeError("key_padding_mask must have dtype torch.bool.")
        if tokens.shape[1] < 1:
            raise ValueError("The input must contain a tissue token.")
        if tokens.shape[1] > self.max_seq_len:
            raise ValueError(
                f"Input length {tokens.shape[1]} exceeds "
                f"max_seq_len={self.max_seq_len}."
            )
        if key_padding_mask[:, 0].any():
            raise ValueError("The tissue token at position 0 cannot be padding.")

    def _required_chunk_counts(
        self,
        dna_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute each sample's true chunk count from its non-padding prefix."""
        dna_lengths = (~dna_padding_mask).sum(dim=1)
        extra_chunks = torch.div(
            torch.clamp(dna_lengths - self.chunk_size_tokens, min=0)
            + self.chunk_stride_tokens
            - 1,
            self.chunk_stride_tokens,
            rounding_mode="floor",
        )
        positive_counts = 1 + extra_chunks
        return torch.where(
            dna_lengths > 0,
            positive_counts,
            torch.zeros_like(positive_counts),
        )

    def _pad_dna_for_windows(
        self,
        dna_tokens: torch.Tensor,
        dna_padding_mask: torch.Tensor,
        n_chunks: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if n_chunks == 0:
            return dna_tokens, dna_padding_mask

        covered_length = (
            (n_chunks - 1) * self.chunk_stride_tokens
            + self.chunk_size_tokens
        )
        right_padding = covered_length - dna_tokens.shape[1]
        if right_padding < 0:
            raise RuntimeError("Internal chunk-coverage calculation failed.")
        if right_padding == 0:
            return dna_tokens, dna_padding_mask

        return (
            F.pad(dna_tokens, (0, right_padding), value=self.pad_id),
            F.pad(dna_padding_mask, (0, right_padding), value=True),
        )

    def encode_chunks(
        self,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Encode DNA into one vector per local chunk.

        The method is public so chunk-level representations can be inspected
        in later mechanistic-interpretability experiments.
        """
        self._validate_inputs(tokens, key_padding_mask)

        batch_size = tokens.shape[0]
        dna_tokens = tokens[:, 1:]
        dna_padding_mask = key_padding_mask[:, 1:]

        required_chunk_counts = self._required_chunk_counts(
            dna_padding_mask
        )
        n_chunks = int(required_chunk_counts.max().item())

        if n_chunks > self.max_chunks:
            raise ValueError(
                f"Input requires {n_chunks} chunks, but the model supports "
                f"at most {self.max_chunks}."
            )

        if n_chunks == 0:
            empty_representations = self.chunk_token.new_zeros(
                batch_size,
                0,
                self.d_model,
            )
            empty_mask = torch.zeros(
                batch_size,
                0,
                dtype=torch.bool,
                device=tokens.device,
            )
            empty_starts = torch.zeros(
                0,
                dtype=torch.long,
                device=tokens.device,
            )
            return {
                "chunk_representations": empty_representations,
                "valid_chunk_mask": empty_mask,
                "chunk_starts": empty_starts,
                "required_chunk_counts": required_chunk_counts,
            }

        dna_tokens, dna_padding_mask = self._pad_dna_for_windows(
            dna_tokens,
            dna_padding_mask,
            n_chunks,
        )

        dna_windows = dna_tokens.unfold(
            dimension=1,
            size=self.chunk_size_tokens,
            step=self.chunk_stride_tokens,
        ).contiguous()
        window_padding_mask = dna_padding_mask.unfold(
            dimension=1,
            size=self.chunk_size_tokens,
            step=self.chunk_stride_tokens,
        ).contiguous()

        chunk_indices = torch.arange(
            n_chunks,
            device=tokens.device,
        ).unsqueeze(0)
        valid_chunk_mask = (
            chunk_indices < required_chunk_counts.unsqueeze(1)
        )

        # In a mixed-length batch, windows beyond a sample's own count can
        # overlap its padded suffix. Mask them entirely so that the sample's
        # prediction cannot depend on lengths of other batch members.
        window_padding_mask = (
            window_padding_mask | ~valid_chunk_mask.unsqueeze(-1)
        )

        window_embeddings = self.token_embedding(dna_windows)
        chunk_tokens = self.chunk_token.expand(
            batch_size,
            n_chunks,
            -1,
            -1,
        )
        local_x = torch.cat(
            [chunk_tokens, window_embeddings],
            dim=2,
        )

        local_positions = torch.arange(
            self.chunk_size_tokens + 1,
            device=tokens.device,
        )
        local_x = (
            local_x
            + self.local_positional_embedding(local_positions)[
                None, None, :, :
            ]
        )

        local_x = local_x.reshape(
            batch_size * n_chunks,
            self.chunk_size_tokens + 1,
            self.d_model,
        )

        chunk_token_mask = torch.zeros(
            batch_size,
            n_chunks,
            1,
            dtype=torch.bool,
            device=tokens.device,
        )
        local_padding_mask = torch.cat(
            [chunk_token_mask, window_padding_mask],
            dim=-1,
        ).reshape(
            batch_size * n_chunks,
            self.chunk_size_tokens + 1,
        )

        for block in self.local_blocks:
            local_x = block(local_x, local_padding_mask)

        local_x = self.local_final_layer_norm(local_x)
        chunk_representations = local_x[:, 0, :].reshape(
            batch_size,
            n_chunks,
            self.d_model,
        )
        chunk_representations = chunk_representations.masked_fill(
            ~valid_chunk_mask.unsqueeze(-1),
            0.0,
        )

        chunk_starts = (
            torch.arange(n_chunks, device=tokens.device)
            * self.chunk_stride_tokens
        )

        return {
            "chunk_representations": chunk_representations,
            "valid_chunk_mask": valid_chunk_mask,
            "chunk_starts": chunk_starts,
            "required_chunk_counts": required_chunk_counts,
        }

    def forward(
        self,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Predict one scalar expression value per sample."""
        chunk_data = self.encode_chunks(tokens, key_padding_mask)
        chunk_representations = chunk_data["chunk_representations"]
        valid_chunk_mask = chunk_data["valid_chunk_mask"]

        batch_size, n_chunks, _ = chunk_representations.shape
        tissue_representation = self.token_embedding(tokens[:, :1])
        global_x = torch.cat(
            [tissue_representation, chunk_representations],
            dim=1,
        )

        global_positions = torch.arange(
            n_chunks + 1,
            device=tokens.device,
        )
        global_x = (
            global_x
            + self.global_positional_embedding(global_positions)[
                None, :, :
            ]
        )

        tissue_padding_mask = torch.zeros(
            batch_size,
            1,
            dtype=torch.bool,
            device=tokens.device,
        )
        global_padding_mask = torch.cat(
            [tissue_padding_mask, ~valid_chunk_mask],
            dim=1,
        )

        for block in self.global_blocks:
            global_x = block(global_x, global_padding_mask)

        global_x = self.global_final_layer_norm(global_x)
        tissue_output = global_x[:, 0, :]
        prediction = self.regression_head(tissue_output).squeeze(-1)

        if not return_aux:
            return prediction

        aux: dict[str, Any] = {
            **chunk_data,
            "global_outputs": global_x,
            "global_padding_mask": global_padding_mask,
            "tissue_representation": tissue_output,
        }
        return prediction, aux
