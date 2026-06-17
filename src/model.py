#!/usr/bin/env python3
"""
Minimalist Pre-LayerNorm Transformer for genomic expression regression.

Maps a tissue-prepended DNA token sequence to a scalar VST expression value
via a shallow 4-layer stack designed for mechanistic interpretability.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """Single Pre-LN transformer block with multi-head self-attention and MLP.

    Residual connections wrap both the attention and feed-forward sub-layers.
    """

    def __init__(
        self, d_model: int, n_heads: int, d_mlp: int, dropout: float
    ) -> None:
        """Initialize layer norms, attention, and feed-forward MLP.

        Args:
            d_model: Model embedding dimension.
            n_heads: Number of attention heads.
            d_mlp: Hidden dimension of the feed-forward network.
            dropout: Dropout probability applied in attention and MLP.
        """
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
        self, x: torch.Tensor, key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply Pre-LN self-attention and MLP with residual connections.

        Args:
            x: Input tensor of shape (B, L, D).
            key_padding_mask: Boolean mask of shape (B, L); True marks padding.

        Returns:
            Output tensor of shape (B, L, D).
        """
        norm_x = self.layer_norm1(x)
        attn_out, _ = self.attention(
            norm_x, norm_x, norm_x, key_padding_mask=key_padding_mask
        )
        # Residual after attention
        x = x + attn_out

        norm_x2 = self.layer_norm2(x)
        mlp_out = self.mlp(norm_x2)
        # Residual after MLP
        x = x + mlp_out

        # Shape: (B, L, D)
        return x


class ExpressionTransformer(nn.Module):
    """Transformer encoder that regresses VST expression from genomic tokens.

    A tissue token prepended at index 0 is routed through a regression head
    after the final layer norm.
    """

    def __init__(self, vocab_size: int, pad_id: int, config: dict) -> None:
        """Build embeddings, transformer stack, and regression head.

        Args:
            vocab_size: Total number of tokens in the vocabulary.
            pad_id: Padding token index for the embedding layer.
            config: Hyperparameter dict with model architecture keys.
        """
        super().__init__()
        d_model = config["d_model"]
        n_heads = config["n_heads"]
        n_layers = config["n_layers"]
        d_mlp = config["d_mlp"]
        dropout = config["dropout"]
        max_seq_len = config["max_seq_len"]

        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_id
        )
        self.positional_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_mlp, dropout)
                for _ in range(n_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self, tokens: torch.Tensor, key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode token sequence and predict scalar VST expression per sample.

        Args:
            tokens: Integer token IDs of shape (B, L).
            key_padding_mask: Boolean padding mask of shape (B, L).

        Returns:
            Predicted expression values of shape (B,).
        """
        batch_size, seq_len = tokens.shape

        # Token embeddings — Shape: (B, L, D)
        x = self.token_embedding(tokens)

        # Absolute position indices, vectorized — Shape: (B, L)
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(
            batch_size, seq_len
        )
        x = x + self.positional_embedding(positions)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, key_padding_mask)

        x = self.final_layer_norm(x)

        # Route tissue token at index 0 — Shape: (B, D)
        cls_token = x[:, 0, :]

        # Regression head — Shape: (B,) after squeeze
        out = self.regression_head(cls_token).squeeze(-1)
        return out
