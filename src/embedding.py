import torch
import torch.nn as nn
import torch.nn.functional as F

class ChromosomeWiseEmbedding(nn.Module):
    def __init__(self, num_alleles=3, embedding_dim=64, attention_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_alleles, embedding_dim=embedding_dim)

        # Attention projections
        self.query_proj = nn.Linear(embedding_dim, attention_dim)
        self.key_proj = nn.Linear(embedding_dim, attention_dim)
        self.value_proj = nn.Linear(embedding_dim, attention_dim)

        self.attention_dim = attention_dim

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, 22, m] with SNP values (0,1,2)

        Returns:
            Tensor of shape [batch_size, 22, attention_dim] (chromosome-level embeddings)
        """
        batch_size, num_chromosomes, num_snps = x.shape
        outputs = []

        for chr_idx in range(num_chromosomes):
            chr_snps = x[:, chr_idx, :]                    # [B, m]
            embedded = self.embedding(chr_snps)            # [B, m, D]

            Q = self.query_proj(embedded)                  # [B, m, d]
            K = self.key_proj(embedded)                    # [B, m, d]
            V = self.value_proj(embedded)                  # [B, m, d]

            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.attention_dim ** 0.5)  # [B, m, m]
            attn_weights = F.softmax(attn_scores, dim=-1)                                    # [B, m, m]

            attn_output = torch.matmul(attn_weights, V)   # [B, m, d]
            pooled = attn_output.mean(dim=1)              # [B, d]
            outputs.append(pooled)

        final_output = torch.stack(outputs, dim=1)         # [B, 22, d]
        return final_output
