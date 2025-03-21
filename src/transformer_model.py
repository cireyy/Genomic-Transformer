import torch
import torch.nn as nn
import torch.nn.functional as F
from .embedding import ChromosomeWiseEmbedding

class MetaGenoTransformer(nn.Module):
    def __init__(self,
                 embedding_dim=512,
                 attention_dim=512,
                 num_heads=8,
                 num_layers=2,
                 num_tasks=6,
                 dropout=0.1):
        super().__init__()

        self.embedding_layer = ChromosomeWiseEmbedding(
            num_alleles=3,
            embedding_dim=embedding_dim,
            attention_dim=attention_dim
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=attention_dim,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(attention_dim),
            nn.Linear(attention_dim, attention_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(attention_dim, num_tasks)
        )

    def forward(self, x, family_history):
        """
        Args:
            x: Tensor of shape [B, 22, m] with SNP values

        Returns:
            logits: Tensor of shape [B, num_tasks]
        """
        x = self.embedding_layer(x)                       # [B, 22, 512]
        x = self.transformer_encoder(x)                  # [B, 22, 512]
        x = x.mean(dim=1)                                # [B, 512] global summary
        logits = self.classifier(x)                      # [B, 6] for 6 tasks
        return logits
