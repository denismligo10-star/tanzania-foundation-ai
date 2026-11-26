
import torch
import torch.nn as nn


class TokenPositionEmbedding(nn.Module):
    """
    Converts token IDs into token and positional embeddings.
    """

    def __init__(
        self,
        vocab_size: int = 16000,
        embedding_dim: int = 256,
        max_sequence_length: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.max_sequence_length = max_sequence_length

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.position_embedding = nn.Embedding(
            max_sequence_length,
            embedding_dim
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = input_ids.shape

        if sequence_length > self.max_sequence_length:
            raise ValueError(
                "Input sequence exceeds maximum sequence length."
            )

        positions = torch.arange(
            sequence_length,
            device=input_ids.device
        )

        token_vectors = self.token_embedding(input_ids)
        position_vectors = self.position_embedding(positions)

        embeddings = token_vectors + position_vectors

        return self.dropout(embeddings)
