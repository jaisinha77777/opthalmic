"""
OphthalmicTransformer: pure PyTorch Transformer for ophthalmic disease modeling.
No LSTM, no ensembles, no sinusoidal positional encoding.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────
# Feature Tokenizer
# ─────────────────────────────────────────────────────────

class FeatureTokenizer(nn.Module):
    """
    Converts one tabular row into a sequence of d_model-dim tokens (one per feature).

    - Numerical feature i : Linear(1, d_model) applied to scalar value;
      missing values are replaced by a per-feature learned mask parameter.
    - Categorical feature i : Embedding(vocab_size_i + 1, d_model),
      last index (vocab_size_i) is the MASK embedding.
    - Binary feature i : Embedding(3, d_model), index 2 is MASK.
    - Learned feature positional embedding: nn.Embedding(N_features, d_model).

    Output: [B, N_features, d_model]
    """

    def __init__(
        self,
        col_types: List[str],
        cat_vocab_sizes: Dict[str, int],
        col_names: List[str],
        d_model: int = 256,
    ) -> None:
        super().__init__()
        self.col_types = col_types
        self.col_names = col_names
        self.d_model = d_model
        self.n_features = len(col_types)

        self.num_projections = nn.ModuleList()
        self.cat_embeddings = nn.ModuleList()
        self.bin_embeddings = nn.ModuleList()
        self.mask_tokens = nn.ParameterList()

        num_idx = 0
        cat_idx = 0
        bin_idx = 0

        for i, (ctype, cname) in enumerate(zip(col_types, col_names)):
            if ctype == "numerical":
                self.num_projections.append(nn.Linear(1, d_model))
                self.mask_tokens.append(nn.Parameter(torch.zeros(d_model)))
                num_idx += 1
            elif ctype == "categorical":
                vs = cat_vocab_sizes.get(cname, 1)
                # +1 for MASK index
                self.cat_embeddings.append(nn.Embedding(vs + 1, d_model, padding_idx=None))
                self.mask_tokens.append(nn.Parameter(torch.zeros(d_model)))
                cat_idx += 1
            elif ctype == "binary":
                self.bin_embeddings.append(nn.Embedding(3, d_model))
                self.mask_tokens.append(nn.Parameter(torch.zeros(d_model)))
                bin_idx += 1

        self.feature_pos_emb = nn.Embedding(self.n_features, d_model)
        self._num_count = num_idx
        self._cat_count = cat_idx
        self._bin_count = bin_idx

        self._init_weights()

    def _init_weights(self) -> None:
        for proj in self.num_projections:
            nn.init.xavier_uniform_(proj.weight)
        for emb in self.cat_embeddings:
            nn.init.normal_(emb.weight, std=0.02)
        for emb in self.bin_embeddings:
            nn.init.normal_(emb.weight, std=0.02)

    def forward(self, x: torch.Tensor, missingness: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x : [B, N_features] — numerical as float, categorical/binary as int indices
            missingness : [B, N_features] binary mask (1=missing)
        Returns:
            tokens : [B, N_features, d_model]
        """
        B = x.size(0)
        tokens = torch.zeros(B, self.n_features, self.d_model, device=x.device, dtype=x.dtype)

        num_ptr = 0
        cat_ptr = 0
        bin_ptr = 0

        for i, ctype in enumerate(self.col_types):
            col_val = x[:, i]
            if missingness is not None:
                is_missing = missingness[:, i].bool()
            else:
                is_missing = torch.zeros(B, dtype=torch.bool, device=x.device)

            if ctype == "numerical":
                proj = self.num_projections[num_ptr]
                mt = self.mask_tokens[i]
                # Replace missing with 0 before projection; then overwrite with mask token
                safe_val = col_val.clone()
                safe_val[is_missing] = 0.0
                tok = proj(safe_val.unsqueeze(-1))  # [B, d_model]
                # Overwrite missing rows with learned mask token
                tok[is_missing] = mt.unsqueeze(0).expand(is_missing.sum(), -1)
                tokens[:, i] = tok
                num_ptr += 1

            elif ctype == "categorical":
                emb = self.cat_embeddings[cat_ptr]
                idx = col_val.long()
                idx = idx.clamp(0, emb.num_embeddings - 1)
                tok = emb(idx)  # [B, d_model]
                tokens[:, i] = tok
                cat_ptr += 1

            elif ctype == "binary":
                emb = self.bin_embeddings[bin_ptr]
                idx = col_val.long().clamp(0, 2)
                tok = emb(idx)
                tokens[:, i] = tok
                bin_ptr += 1

        pos_indices = torch.arange(self.n_features, device=x.device)
        tokens = tokens + self.feature_pos_emb(pos_indices).unsqueeze(0)
        return tokens


# ─────────────────────────────────────────────────────────
# Attention hook helper
# ─────────────────────────────────────────────────────────

class AttentionCapture(nn.Module):
    """Wraps a TransformerEncoderLayer to expose attention weights."""

    def __init__(self, layer: nn.TransformerEncoderLayer) -> None:
        super().__init__()
        self.layer = layer
        self.last_attn_weights: Optional[torch.Tensor] = None

    def forward(self, src: torch.Tensor, src_mask=None, src_key_padding_mask=None, **kwargs) -> torch.Tensor:
        # Manually call self-attention to extract weights
        x = src
        # Pre-LN
        x_norm = self.layer.norm1(x)
        attn_out, attn_weights = self.layer.self_attn(
            x_norm, x_norm, x_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,  # keep per-head
        )
        self.last_attn_weights = attn_weights.detach()  # [B, n_heads, L, L]
        x = x + self.layer.dropout1(attn_out)
        # FFN sublayer (post-attn, pre-LN)
        x_norm2 = self.layer.norm2(x)
        ff_out = self.layer.linear2(
            self.layer.dropout(self.layer.activation(self.layer.linear1(x_norm2)))
        )
        x = x + self.layer.dropout2(ff_out)
        return x


# ─────────────────────────────────────────────────────────
# Main Transformer
# ─────────────────────────────────────────────────────────

class OphthalmicTransformer(nn.Module):
    """
    Tabular Transformer with:
    - FeatureTokenizer → token sequence
    - Prepended CLS token
    - Pre-LN TransformerEncoderLayers with attention capture
    - Prediction head (from CLS)
    - Uncertainty head (predicted variance)
    """

    def __init__(
        self,
        col_types: List[str],
        cat_vocab_sizes: Dict[str, int],
        col_names: List[str],
        n_features: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 512,
        dropout: float = 0.15,
        n_classes: int = 2,
        task: str = "classification",
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.n_classes = n_classes
        self.task = task

        self.tokenizer = FeatureTokenizer(col_types, cat_vocab_sizes, col_names, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.layers = nn.ModuleList([
            AttentionCapture(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
            )
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # Prediction head
        self.pred_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

        # Uncertainty head
        self.unc_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Softplus(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(self, x: torch.Tensor, missingness: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns latent CLS state and per-layer attention weights.
        x: [B, N_features] for tabular; call forward for full dict.
        """
        B = x.size(0)
        tokens = self.tokenizer(x, missingness)  # [B, N_feat, d_model]
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, tokens], dim=1)  # [B, N_feat+1, d_model]

        attn_weights: List[torch.Tensor] = []
        h = seq
        for layer in self.layers:
            h = layer(h)
            if layer.last_attn_weights is not None:
                attn_weights.append(layer.last_attn_weights)

        h = self.final_norm(h)
        latent = h[:, 0]  # CLS token = S_t ∈ R^256
        return latent, attn_weights

    def forward(
        self,
        x: torch.Tensor,
        missingness: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        latent, attn_weights = self.encode(x, missingness)
        logits = self.pred_head(latent)
        pred_variance = self.unc_head(latent)
        return {
            "logits": logits,
            "latent_state": latent,
            "pred_variance": pred_variance,
            "attention_weights": attn_weights,
        }


# ─────────────────────────────────────────────────────────
# Temporal Transformer (sequential patient data)
# ─────────────────────────────────────────────────────────

class CausalTransformerEncoder(nn.Module):
    """Transformer with a causal (auto-regressive) attention mask."""

    def __init__(self, d_model: int = 256, n_heads: int = 8, n_layers: int = 2, d_ff: int = 512, dropout: float = 0.15) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionCapture(
                nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads,
                    dim_feedforward=d_ff, dropout=dropout,
                    activation="gelu", batch_first=True, norm_first=True,
                )
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device) * float("-inf"), diagonal=1
        )
        h = x
        for layer in self.layers:
            h = layer(h, src_mask=causal_mask)
        return self.norm(h)


class TemporalTransformer(nn.Module):
    """
    Sequential patient data encoder.
    Input: [B, T, N_features]
    Output dict identical to OphthalmicTransformer.forward()
    """

    def __init__(
        self,
        col_types: List[str],
        cat_vocab_sizes: Dict[str, int],
        col_names: List[str],
        n_features: int,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.15,
        n_classes: int = 2,
        task: str = "classification",
        seq_len: int = 6,
    ) -> None:
        super().__init__()
        self.T = seq_len
        self.n_features = n_features
        self.d_model = d_model
        self.task = task
        self.n_classes = n_classes

        self.tokenizer = FeatureTokenizer(col_types, cat_vocab_sizes, col_names, d_model)

        # 2D positional encoding
        self.time_pos = nn.Embedding(seq_len, d_model)
        self.feat_pos = nn.Embedding(n_features, d_model)

        # Cross-feature encoder
        self.cross_encoder = nn.ModuleList([
            AttentionCapture(
                nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads,
                    dim_feedforward=d_ff, dropout=dropout,
                    activation="gelu", batch_first=True, norm_first=True,
                )
            )
            for _ in range(4)
        ])

        # Temporal encoder (causal)
        self.temporal_encoder = CausalTransformerEncoder(d_model, n_heads, 2, d_ff, dropout)

        self.final_norm = nn.LayerNorm(d_model)

        self.pred_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )
        self.unc_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        x: torch.Tensor,
        missingness: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        x: [B, T, N_features]
        """
        B, T, F = x.shape
        t_idx = torch.arange(T, device=x.device)
        f_idx = torch.arange(F, device=x.device)

        time_pe = self.time_pos(t_idx)   # [T, d_model]
        feat_pe = self.feat_pos(f_idx)   # [F, d_model]

        all_tokens = []
        attn_weights: List[torch.Tensor] = []

        for t in range(T):
            x_t = x[:, t, :]
            m_t = missingness[:, t, :] if missingness is not None else None
            tok_t = self.tokenizer(x_t, m_t)  # [B, F, d_model]
            # Add 2D positional encoding
            tok_t = tok_t + time_pe[t].unsqueeze(0).unsqueeze(0) + feat_pe.unsqueeze(0)
            all_tokens.append(tok_t)

        # Stack: [B, T*F, d_model]
        seq = torch.cat(all_tokens, dim=1)

        h = seq
        for layer in self.cross_encoder:
            h = layer(h)
            if layer.last_attn_weights is not None:
                attn_weights.append(layer.last_attn_weights)

        # Reshape to [B, T, F, d_model] → mean over F → [B, T, d_model]
        h = h.view(B, T, F, self.d_model).mean(dim=2)

        # Causal temporal encoder
        h = self.temporal_encoder(h)  # [B, T, d_model]
        latent = h[:, -1]             # last timestep = S_t

        logits = self.pred_head(latent)
        pred_variance = self.unc_head(latent)

        return {
            "logits": logits,
            "latent_state": latent,
            "pred_variance": pred_variance,
            "attention_weights": attn_weights,
        }


# ─────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────

def build_model(
    feature_metadata: Dict[str, Any],
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 6,
    d_ff: int = 1024,   # standard 4× d_model ratio; was 512 (2×) which under-parameterised FFN
    dropout: float = 0.15,
) -> nn.Module:
    """Construct the appropriate model based on feature metadata."""
    col_types = feature_metadata["col_types"]
    cat_vocab_sizes = feature_metadata["cat_vocab_sizes"]
    col_names = feature_metadata["col_names"]
    n_features = feature_metadata["n_features"]
    n_classes = feature_metadata["n_classes"]
    task = feature_metadata["task"]
    has_sequences = feature_metadata.get("has_sequences", False)
    seq_len = feature_metadata.get("seq_len", 1)

    if has_sequences and seq_len > 1:
        return TemporalTransformer(
            col_types=col_types,
            cat_vocab_sizes=cat_vocab_sizes,
            col_names=col_names,
            n_features=n_features,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            n_classes=n_classes,
            task=task,
            seq_len=seq_len,
        )
    else:
        return OphthalmicTransformer(
            col_types=col_types,
            cat_vocab_sizes=cat_vocab_sizes,
            col_names=col_names,
            n_features=n_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            n_classes=n_classes,
            task=task,
        )
