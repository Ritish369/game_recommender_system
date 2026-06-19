# model.py — TwoTowerModel for v1 game recommendation.
# Item tower: item_id_embedding + static features → 128-dim L2-norm.
# User tower: history → MultiHeadAttention → mean pool → concat with user_id_embedding → 128-dim.
# Shared embedding space where dot product = cosine similarity.

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EMBEDDING_DIM


class TwoTowerModel(nn.Module):
    """Two-tower neural retrieval model with shared item vocabulary.

    Item tower: concat(item_id_emb, static_features) → MLP → L2-norm.
    User tower: history items → MultiHeadAttention → mean pool →
        concat(user_id_emb, sequence_vec) → MLP → L2-norm.
    
    Item bias learns broad appeal (positive bias = everyone likes this item).
    Temperature is a learnable parameter initialized at log(0.07).
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        item_feature_dim: int,
        padding_idx: int,
        embedding_dim: int = EMBEDDING_DIM,
        num_heads: int = 4,
        dropout: float = 0.2,
        ablation: str = "no_user_id",
    ):
        super().__init__()
        self.num_items = num_items
        self.padding_idx = padding_idx
        self.embedding_dim = embedding_dim
        self.ablation = ablation

        # ── Embeddings ──
        # Shared item_id_embedding used by BOTH item tower and user tower (history encoding).
        self.item_id_embedding = nn.Embedding(
            num_items + 1,  # +1 for padding index
            embedding_dim,
            padding_idx=padding_idx,
        )
        self.user_id_embedding = nn.Embedding(num_users, embedding_dim)

        # ── User tower: MultiHeadAttention over history ──
        # 4 heads, 128-dim each — captures pairwise relationships between history items.
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # LayerNorm after residual connection for training stability.
        self.attn_norm = nn.LayerNorm(embedding_dim)

        # ── User tower projection (after concat user_id_emb + sequence_vec) ──
        user_input_dim = embedding_dim * 2
        self.user_projection = nn.Sequential(
            nn.Linear(user_input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim),
        )

        # ── Item tower projection ──
        self.item_projection = nn.Sequential(
            nn.Linear(embedding_dim + item_feature_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim),
        )

        # ── Item bias: learnable broad-appeal scalar per item ──
        self.item_bias = nn.Embedding(num_items, 1)
        nn.init.zeros_(self.item_bias.weight)

        # ── Learnable temperature for InfoNCE ──
        # Initialized at log(0.07) so effective starting temperature is 0.07.
        # Lower temp → sharper distribution, harder negatives.
        self.log_temperature = nn.Parameter(torch.tensor(math.log(0.07)))

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier init for linear layers, normal init for embeddings."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def temperature(self) -> torch.Tensor:
        """Exponentiate log_temperature to ensure positivity."""
        return torch.exp(self.log_temperature)

    def encode_item(
        self,
        item_ids: torch.Tensor,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        """Encode items via item_id_embedding + static features → L2-norm.
        
        Args:
            item_ids: [B] item indices
            item_features: [B, item_feature_dim] precomputed static features
        Returns [B, 128] L2-normalized item vectors.
        """
        id_emb = self.item_id_embedding(item_ids)  # [B, 128]
        combined = torch.cat([id_emb, item_features], dim=-1)  # [B, 128+item_feature_dim]
        return F.normalize(self.item_projection(combined), dim=-1)

    def encode_user(
        self,
        user_ids: torch.Tensor,
        history_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Encode users via attention over history + user_id_embedding → L2-norm.
        
        Args:
            user_ids: [B] user indices
            history_item_ids: [B, 50] padded history item indices
        Returns [B, 128] L2-normalized user vectors.
        """
        batch_size, max_len = history_item_ids.shape

        # ── Get item embeddings for history items (shared vocabulary) ──
        history_emb = self.item_id_embedding(history_item_ids)  # [B, 50, 128]

        # ── Build padding mask ──
        seq_mask = history_item_ids == self.padding_idx  # [B, 50]

        # ── MultiHeadAttention over history ──
        if self.ablation == "mean_pool":
            # Ablation: replace attention with simple mean pooling.
            # Tests whether attention earns its keep at this scale.
            attn_output = history_emb
        else:
            attn_output, _ = self.attention(
                query=history_emb,
                key=history_emb,
                value=history_emb,
                key_padding_mask=seq_mask,
            )
        # Residual connection + LayerNorm for stable gradients.
        history_emb = self.attn_norm(history_emb + attn_output)

        # ── Mean pool over valid (non-padding) positions ──
        valid_mask = (~seq_mask).float().unsqueeze(-1)  # [B, 50, 1]
        pooled = (history_emb * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
        # Zero-history users get a zero vector (valid_mask.sum=0 → pooled=0).

        # ── User ID embedding (skipped in no_user_id ablation) ──
        if self.ablation == "no_user_id":
            user_emb = torch.zeros(batch_size, self.embedding_dim, device=history_emb.device)
        else:
            user_emb = self.user_id_embedding(user_ids)  # [B, 128]

        combined = torch.cat([user_emb, pooled], dim=-1)  # [B, 256]
        return F.normalize(self.user_projection(combined), dim=-1)

    @torch.no_grad()
    def encode_all_items(
        self,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        """Precompute embeddings for ALL warm catalog items.
        
        Caller does not need to pass IDs; this method constructs them internally.
        Used at evaluation for full-catalog scoring via single matrix multiplication.
        Args:
            item_features: [num_items, item_feature_dim]
        Returns [num_items, 128] L2-normalized item vectors.
        """
        all_ids = torch.arange(len(item_features), device=item_features.device)
        return self.encode_item(all_ids, item_features)
