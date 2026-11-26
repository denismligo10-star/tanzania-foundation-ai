
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        embedding_dim=256,
        num_heads=8,
        dropout=0.1,
        max_sequence_length=512,
    ):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads"
            )

        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.qkv = nn.Linear(
            embedding_dim,
            3 * embedding_dim
        )

        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(
                    max_sequence_length,
                    max_sequence_length,
                    dtype=torch.bool
                )
            ),
            persistent=False,
        )

    def forward(self, x):
        batch_size, sequence_length, channels = x.shape

        if sequence_length > self.causal_mask.size(0):
            raise ValueError(
                "Input exceeds maximum sequence length."
            )

        qkv = self.qkv(x)

        qkv = qkv.reshape(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim
        )

        qkv = qkv.permute(2, 0, 3, 1, 4)

        query, key, value = qkv.unbind(0)

        scores = (
            query @ key.transpose(-2, -1)
        ) * (self.head_dim ** -0.5)

        mask = self.causal_mask[
            :sequence_length,
            :sequence_length
        ]

        scores = scores.masked_fill(
            ~mask,
            torch.finfo(scores.dtype).min
        )

        weights = F.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)

        output = weights @ value

        output = output.transpose(1, 2).contiguous()

        output = output.reshape(
            batch_size,
            sequence_length,
            channels
        )

        return self.residual_dropout(
            self.output_projection(output)
        )


class FeedForward(nn.Module):
    def __init__(self, embedding_dim=256, dropout=0.1):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim=256,
        num_heads=8,
        dropout=0.1,
        max_sequence_length=512,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embedding_dim)

        self.attention = CausalSelfAttention(
            embedding_dim,
            num_heads,
            dropout,
            max_sequence_length,
        )

        self.norm2 = nn.LayerNorm(embedding_dim)

        self.feed_forward = FeedForward(
            embedding_dim,
            dropout
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))

        return x
