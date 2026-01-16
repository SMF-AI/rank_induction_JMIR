from typing import Optional, Tuple, Literal

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, heads, dropout=attn_drop, batch_first=True
        )
        self.drop1 = nn.Dropout(proj_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop=proj_drop)

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.drop1(a)
        x = x + self.mlp(self.norm2(x))
        return x


class PatchLevelAggregator(nn.Module):
    """
    16x16 토큰을 Linear proj → (pos+CLS) → ViT blocks → LayerNorm → CLS 추출
    """

    def __init__(
        self,
        in_dim: int = 1024,
        embed_dim: int = 192,
        depth: int = 4,
        heads: int = 3,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        learned_pos: bool = True,
        grid_hw: Tuple[int, int] = (16, 16),
    ):
        super().__init__()
        self.grid_h, self.grid_w = grid_hw
        self.proj = nn.Linear(in_dim, embed_dim, bias=False)
        self.pos = nn.Parameter(torch.zeros(1, self.grid_h * self.grid_w, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, heads, mlp_ratio, attn_drop, drop)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (K, R, C, D_in) with R=C=16
        returns: region_cls (K, D), patch_tokens (K, R*C, D)
        """
        if x.ndim != 4:
            raise ValueError(f"PatchLevelAggregator expects (K,R,C,D), got {x.shape}")
        K, R, C, Din = x.shape
        assert (R, C) == (
            self.grid_h,
            self.grid_w,
        ), f"expect {(self.grid_h,self.grid_w)}, got {(R,C)}"

        tokens = x.view(K, R * C, Din)  # (K, 256, D_in)
        tokens = self.proj(tokens)  # (K, 256, D)
        tokens = tokens + self.pos.to(tokens.device)  # pos add

        cls = self.cls.expand(K, -1, -1)  # (K,1,D)
        seq = torch.cat([cls, tokens], dim=1)  # (K, 257, D)
        for blk in self.blocks:
            seq = blk(seq)
        seq = self.norm(seq)

        cls_tok = seq[:, 0]  # (K, D)
        patch_tok = seq[:, 1:]  # (K, 256, D)
        return cls_tok, patch_tok


class GlobalAttentionPool(nn.Module):
    """
    x: (K, D) → attn weights (K,) & pooled (D,)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(dim))
        nn.init.normal_(self.q, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # alignment: (K,)
        scale = math.sqrt(x.size(-1))
        alignment = (x @ self.q) / scale
        weights = torch.softmax(alignment, dim=0)  # (K,)
        pooled = (weights.unsqueeze(1) * x).sum(0)  # (D,)
        return pooled, alignment


class RegionLevelAggregator(nn.Module):
    """
    입력:  region sequence (K, D)  — 각 region의 CLS 임베딩
    출력:  pooled (D,), attn_weights (K,), attn_scores (K,), refined_tokens (K, D)
    구성:  ViT blocks (no CLS, no pos) → GlobalAttentionPool (GAP)
    """

    def __init__(
        self,
        dim: int = 192,
        depth: int = 2,
        heads: int = 3,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(dim, heads, mlp_ratio, attn_drop, drop)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.gap = GlobalAttentionPool(dim)

    def forward(self, region_seq: torch.Tensor):
        """
        region_seq: (K, D)
        """
        x = region_seq.unsqueeze(0)  # (1, K, D) for batch_first MHA
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # (1, K, D)
        tokens = x.squeeze(0)  # (K, D)

        pooled, scores = self.gap(tokens)  # (D,), (K,)
        weights = torch.softmax(scores, dim=0)  # (K,)
        return pooled, weights, scores, tokens


class HIPT(nn.Module):
    """
    입력 feature:
      - (K, R, C, D)  또는  (1, K, R, C, D)  [기본 R=C=16, D=1024]
    출력 (ABMIL/DSMIL 동일 인터페이스):
      - logits: (C_out,)
      - second:
          * return_with == "attention_weight": (K,)
          * return_with == "attention_score" : (K,)
          * return_with == "contribution"    : (K, C_out)
          * else                             : (K,)  (attention_weight)
    """

    def __init__(
        self,
        in_dim: int = 1024,
        grid_hw: Tuple[int, int] = (16, 16),
        embed_dim: int = 192,  # D'
        patch_depth: int = 4,
        patch_heads: int = 3,
        region_depth: int = 2,
        region_heads: int = 3,
        num_classes: int = 2,
        threshold: Optional[float] = None,
        return_with: Literal[
            "contribution", "attention_weight", "attention_score"
        ] = "attention_score",
    ):
        super().__init__()
        self.threshold = threshold
        self.return_with = return_with

        self.patch_agg = PatchLevelAggregator(
            in_dim=in_dim,
            embed_dim=embed_dim,
            depth=patch_depth,
            heads=patch_heads,
            grid_hw=grid_hw,
        )
        self.region_agg = RegionLevelAggregator(
            dim=embed_dim, depth=region_depth, heads=region_heads
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (K, R, C, D)  or  (1, K, R, C, D)
        """
        if x.ndim == 5 and x.shape[0] == 1:
            x = x.squeeze(0)
        if x.ndim != 4:
            raise ValueError(f"HIPT expects (K,R,C,D) or (1,K,R,C,D), got {x.shape}")

        # (1) Patch-level: each region's 16x16 grid → region CLS (K, D')
        region_cls, _ = self.patch_agg(x)  # (K, D'), (K,256,D')

        # (2) Region-level: Transformer (no CLS/pos) → Global Attn Pooling
        pooled, attn_w, attn_scores, refined = self.region_agg(
            region_cls
        )  # (D'), (K,), (K,), (K,D')

        weights = attn_w
        if self.threshold is not None:
            n = weights.numel()
            thr = self.threshold / float(n)
            weights = torch.clamp(weights - thr, min=0.0)  # (K,)

            denom = weights.sum()
            if denom <= 0:
                weights = attn_w
            else:
                weights = weights / denom

        # (4) Slide logits
        #   Classify directly with GAP result (pooled)
        logits = self.classifier(pooled)  # (C_out,)

        if self.return_with == "contribution":
            return logits, weights.unsqueeze(1) * self.classifier(refined)  # (K, C_out)

        if self.return_with == "attention_weight":
            return logits, weights

        if self.return_with == "attention_score":
            return logits, attn_scores

        return logits, weights
