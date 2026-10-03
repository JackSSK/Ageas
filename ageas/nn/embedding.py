#!/usr/bin/env python3
"""Per-modality embedding stage of :class:`~ageas.nn.NN_Classifier`.

The input is ``(batch, n_modalities, n_genes)``: one row per data layer
(e.g. spliced and unspliced counts, or RNA and ATAC), genes in ``var``
order. Genes are split into ``seq_len`` chunks; each modality embeds each
chunk into one token, and token *t* is the sum over modalities of their
chunk-*t* tokens. Summing in a shared embedding space is what lets the
modalities be fused token by token.
"""
import math

import torch
import torch.nn as nn

#: Embedder types a config can select with ``embedder``.
EMBEDDERS = ('linear', 'mlp')


class Chunk_Embedder(nn.Module):
    """One modality's embedder: each gene chunk gets its own weights.

    ``'linear'`` maps a chunk with a :class:`~torch.nn.Linear`; ``'mlp'``
    adds a normalisation layer and a ReLU (``Linear -> Norm -> ReLU``).
    Chunks never share weights: genes in different chunks are different
    genes.
    """

    def __init__(
        self,
        seq_len: int,
        chunk_size: int,
        token_dim: int,
        embedder: str = 'mlp',
        norm_layer=nn.LayerNorm,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if embedder not in EMBEDDERS:
            raise ValueError(f"embedder must be one of {EMBEDDERS}, got {embedder!r}")
        self.chunks = nn.ModuleList(
            nn.Linear(chunk_size, token_dim, bias=bias) for _ in range(seq_len)
        )
        self.norm = norm_layer(token_dim) if embedder == 'mlp' else None
        self.activation = nn.ReLU(inplace=True) if embedder == 'mlp' else None

    def forward(self, chunks: torch.Tensor) -> torch.Tensor:
        """``(batch, seq_len, chunk_size)`` -> ``(batch, seq_len, token_dim)``."""
        tokens = torch.stack(
            [linear(chunks[:, t]) for t, linear in enumerate(self.chunks)], dim=1
        )
        if self.norm is not None:
            tokens = self.activation(self.norm(tokens))
        return tokens


class Modality_Embedding(nn.Module):
    """Turn ``(batch, n_modalities, len_in)`` into ``(batch, seq_len, token_dim)``.

    Genes are split, in order, into ``seq_len`` chunks of
    ``ceil(len_in / seq_len)`` genes; the last chunk is zero-padded. Each
    modality has its own :class:`Chunk_Embedder`, and the modalities'
    tokens are summed position by position.

    With ``token_dim=None`` there is no embedder: each token is the raw
    chunk (summed over modalities), and :attr:`token_dim` is the chunk size.

    Attributes:
        seq_len: Number of tokens.
        chunk_size: Genes per chunk.
        token_dim: Size of each output token.
        embedders: One :class:`Chunk_Embedder` per modality, or ``None``.
    """

    def __init__(
        self,
        n_modalities: int,
        len_in: int,
        seq_len: int = 1,
        token_dim: int = None,
        embedder: str = 'mlp',
        norm_layer=nn.LayerNorm,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.chunk_size = math.ceil(len_in / seq_len)
        self.padding = self.chunk_size * seq_len - len_in
        self.token_dim = self.chunk_size if token_dim is None else token_dim
        self.embedders = None if token_dim is None else nn.ModuleList(
            Chunk_Embedder(
                seq_len, self.chunk_size, token_dim,
                embedder=embedder, norm_layer=norm_layer, bias=bias,
            )
            for _ in range(n_modalities)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(batch, n_modalities, len_in)`` -> ``(batch, seq_len, token_dim)``."""
        if self.padding:
            x = nn.functional.pad(x, (0, self.padding))
        chunks = x.reshape(x.shape[0], x.shape[1], self.seq_len, self.chunk_size)
        if self.embedders is None:
            return chunks.sum(dim=1)
        return sum(
            embedder(chunks[:, m]) for m, embedder in enumerate(self.embedders)
        )
