
import torch
import torch.nn as nn

from .embedding import TokenPositionEmbedding
from .attention import TransformerBlock


class TanzaniaFoundationModel(nn.Module):
    """
    A decoder-only Transformer language model.
    """

    def __init__(
        self,
        vocab_size=16000,
        embedding_dim=256,
        num_layers=6,
        num_heads=8,
        max_sequence_length=512,
        dropout=0.1,
    ):
        super().__init__()

        self.embedding = TokenPositionEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_sequence_length=max_sequence_length,
            dropout=dropout,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                dropout=dropout,
                max_sequence_length=max_sequence_length,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embedding_dim)

        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size,
            bias=False
        )

        # Share input and output token embeddings.
        self.lm_head.weight = self.embedding.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(self, input_ids):
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape (batch_size, sequence_length)"
            )

        x = self.embedding(input_ids)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits
