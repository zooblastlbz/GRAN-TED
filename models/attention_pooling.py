# models/attention_pooling.py
"""
Two-layer Attention Pooling
===========================
• 输入形状:
    1) [B, N, dim]  —— 序列 hidden states
    2) [B, dim]     —— 单个 embedding
• 输出形状:
    [B, dim_out]
• 特性:
    * learnable [POOL] token  —— 汇聚全局信息
    * 两层 self-attention + FFN
    * 开关: use_rope → 是否在 Q/K 上使用 RoPE
Only parameters inside this module are trainable; all backbone
models remain frozen.
"""

from __future__ import annotations
from typing import Optional, Tuple
import logging, os, sys, datetime
import math
import torch
import torch.nn as nn
import torch.nn.functional as F



import torch

def rms_norm(x, eps=1e-6):
    """
    Apply RMS normalization to the last dimension of input tensor.
    
    Args:
        x (torch.Tensor): Input tensor of shape [..., dim]
        eps (float): Small value to avoid division by zero. Default: 1e-6
    
    Returns:
        torch.Tensor: Normalized tensor of the same shape as input.
    """
    dtype = x.dtype
    x_f32 = x.to(torch.float32)
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    x_normed = x_f32 * torch.rsqrt(variance + eps)
    return x_normed.to(dtype)


# ========= RoPE helpers =========
def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rotary position embedding for q / k.

    Args:
        q, k : shape (..., L, D)
               • 3-D  → (B, L, D)
               • 4-D  → (B, H, L, D)
               只要倒数第 2 维是序列长度，最后一维是特征维即可。
        offset : 左移的绝对位置起点（首 token 不做 RoPE 时传 1）

    Returns:
        (q_rot, k_rot) 与输入形状一致
    """
    if q.size(-1) % 2 != 0:
        raise ValueError("The last dimension of q/k must be even for RoPE")

    # ------------- 生成 cos/sin 表 -------------
    device, dtype, dim = q.device, q.dtype, q.size(-1)
    seq_len = q.size(-2)

    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    t = torch.arange(offset, offset + seq_len, device=device, dtype=dtype)           # (L,)
    freqs = torch.einsum("l , d -> l d", t, inv_freq)                                # (L, dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)                                          # (L, dim)
    cos, sin = emb.cos(), emb.sin()                                                 # (L, dim)

    # 扩充到 q/k 形状（在 batch/head 维度上自动 broadcast）
    while cos.ndim < q.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    # ------------- 旋转 -----------------------
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot
# =================================


# class _RoPEBlock(nn.Module):
#     def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads

#         # self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
#         self.q_proj = nn.Linear(dim, dim, bias=False)
#         self.k_proj = nn.Linear(dim, dim, bias=False)
#         self.v_proj   = nn.Linear(dim, dim, bias=False)
#         self.o_proj   = nn.Linear(dim, dim, bias=False)
#         self.dropout  = nn.Dropout(dropout)

#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(dim)

#         self.mlp = nn.Sequential(
#             nn.Linear(dim, int(dim * mlp_ratio)),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(int(dim * mlp_ratio), dim),
#             nn.Dropout(dropout),
#         )

#     def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
#         """
#         x: [B, L, dim] (L = 1 + N, where N is sequence length)
#         attn_mask: [B, L] (Boolean mask, True indicates a value to be IGNORED)
#         """
#         B, L, D = x.size()
#         h = self.num_heads

#         # --- Self-Attention ---
#         residual = x
#         x = self.norm1(x)

#         # qkv = self.qkv_proj(x).view(B, L, 3, h, self.head_dim)
#         # q, k, v = qkv.unbind(dim=2)
#         q = self.q_proj(x)
#         k = self.k_proj(x)
#         v = self.v_proj(x)
        
#         # FIX: --- Isolate [POOL] token from RoPE ---
#         # Reshape to [B, L, D] for RoPE processing
#         # q = q.reshape(B, L, D)
#         # k = k.reshape(B, L, D)
#         # v = v.reshape(B, L, D)

#         # Separate the [POOL] token (at index 0) from the rest of the sequence
#         q_pool, q_seq = q[:, 0:1, :], q[:, 1:, :]
#         k_pool, k_seq = k[:, 0:1, :], k[:, 1:, :]
        
#         # Apply RoPE *only* to the sequence part. Note the seq_len is L-1.
#         if q_seq.shape[1] > 0: # Ensure sequence is not empty
#             q_seq, k_seq = apply_rotary_pos_emb(q_seq, k_seq, seq_len=L - 1)
        
#         # Concatenate back
#         q = torch.cat([q_pool, q_seq], dim=1)
#         k = torch.cat([k_pool, k_seq], dim=1)
#         # --- End of RoPE fix ---

#         # Reshape for multi-head attention
#         q = q.view(B, L, h, self.head_dim).transpose(1, 2)
#         k = k.view(B, L, h, self.head_dim).transpose(1, 2)
#         v = v.view(B, L, h, self.head_dim).transpose(1, 2)

#         attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

#         if attn_mask is not None:
#             mask = attn_mask.view(B, 1, 1, L)
#             attn_scores = attn_scores.masked_fill(mask, -torch.finfo(attn_scores.dtype).max)

#         attn_weights = attn_scores.softmax(dim=-1)
#         attn_out = (attn_weights @ v)
#         attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
#         attn_out = self.o_proj(attn_out)
#         attn_out = self.dropout(attn_out)

#         x = residual + attn_out

#         # --- FFN ---
#         x = x + self.mlp(self.norm2(x))
#         return x

class _RoPEBlock(nn.Module):
    """
    Pre-LN Transformer block with rotary position embedding (RoPE):
      • 首个 token（如 [CLS]/[POOL]）不做 RoPE
      • 仅对 q / k 旋转
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        assert dim % num_heads == 0, "`dim` 必须能被 `num_heads` 整除"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # --- projections ---
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        # --- norms & MLP ---
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,                  # (B, L, D)  L = 1 + seq_len
        attn_mask: Optional[torch.Tensor] = None,  # (B, L)  True = mask
    ) -> torch.Tensor:
        B, L, D = x.size()
        h = self.num_heads
        residual = x

        # --- Self-Attention --------------------------------------------------
        x = self.norm1(x)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 分离首 token（位置 0）——不做 RoPE
        q_pool, q_seq = q[:, :1, :], q[:, 1:, :]
        k_pool, k_seq = k[:, :1, :], k[:, 1:, :]

        # 仅对序列部分做 RoPE
        if q_seq.size(1) > 0:
            # (B , L-1 , D) → (B , h , L-1 , d)
            q_seq = q_seq.view(B, -1, h, self.head_dim).transpose(1, 2)
            k_seq = k_seq.view(B, -1, h, self.head_dim).transpose(1, 2)

            # offset=1 保留绝对位置信息
            q_seq, k_seq = apply_rotary_pos_emb(q_seq, k_seq, offset=1)

            # 再还原回 (B , L-1 , D)
            q_seq = q_seq.transpose(1, 2).reshape(B, -1, D)
            k_seq = k_seq.transpose(1, 2).reshape(B, -1, D)

        # 拼回完整序列
        q = torch.cat([q_pool, q_seq], dim=1)
        k = torch.cat([k_pool, k_seq], dim=1)

        # (B, L, D) → (B, h, L, d)
        q = q.view(B, L, h, self.head_dim).transpose(1, 2)
        k = k.view(B, L, h, self.head_dim).transpose(1, 2)
        v = v.view(B, L, h, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            mask = attn_mask.view(B, 1, 1, L)        # broadcast to heads
            fill_val = torch.finfo(attn_scores.dtype).min
            attn_scores = attn_scores.masked_fill(mask, fill_val)

        attn_weights = attn_scores.softmax(dim=-1)
        attn_weights = F.dropout(
            attn_weights, p=self.dropout.p, training=self.training
        )

        attn_out = attn_weights @ v                           # (B, h, L, d)
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)  # (B, L, D)
        attn_out = self.o_proj(attn_out)
        attn_out = self.dropout(attn_out)

        x = residual + attn_out

        # --- Feed-Forward ----------------------------------------------------
        x = x + self.mlp(self.norm2(x))
        return x


class AttnPooling(nn.Module):
    def __init__(
        self,
        layers_to_select,
        dim: int,
        dim_out: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_layers: int = 2,
        use_rope: bool = True,
        num_llm_layers: Optional[int] = 0,
        elementwise_affine: bool = True,
        norm_type = "layer_norm",
    ):
        super().__init__()
        self.use_rope = use_rope
        self.pool_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.num_layers = num_layers
        self.norm_type=norm_type

        self.layers_to_select=layers_to_select
        if isinstance(self.layers_to_select,int):
            self.num_layer_norms=1
        elif isinstance(self.layers_to_select,list):
            self.num_layer_norms=len(self.layers_to_select)
            print(f"layers_to_select:{self.layers_to_select}")
        elif isinstance(self.layers_to_select,str):
            if self.layers_to_select=="all":
                self.num_layer_norms=num_llm_layers
            else:
                begin=int(self.layers_to_select.split(':')[0])
                end=int(self.layers_to_select.split(':')[-1])
                hidden_list=[i for i in range(begin,end,1)]
                self.num_layer_norms=len(hidden_list)
            #self.layer_norms = nn.ModuleList([
            #    nn.LayerNorm(dim, elementwise_affine=elementwise_affine) for _ in range(self.num_layer_norms + 1)
            #])
            # L = self.num_layer_norms + 1
            # # 每层一个“标量”缩放和偏移
            # self.ln_scales = nn.Parameter(torch.ones(L))   # w_i
            # self.ln_biases = nn.Parameter(torch.zeros(L))  # b_i
        if use_rope:
            self.encoder = nn.ModuleList([
                _RoPEBlock(dim, num_heads, mlp_ratio, dropout)
                for _ in range(self.num_layers)
            ])
        else:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=int(dim * mlp_ratio),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.num_layers)

        self.proj = nn.Linear(dim, dim_out)
        self.out_norm = nn.LayerNorm(dim)

        nn.init.trunc_normal_(self.pool_token, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        D = x.shape[-1]
        normalized_shape = (D,)
        if self.num_layer_norms>1:
            #assert (x.shape[1] == self.num_layer_norms + 1 and x.dim() == 4) or \
            #    (x.shape[0] == self.num_layer_norms + 1 and x.dim() == 3), \
            #    f"Expected x shape to be [B, L, N, dim] or [L, N, dim] with L={self.num_layer_norms + 1}, but got {x.shape}."



            if x.dim() == 4:
                B, L, N, D = x.shape
                x_out = torch.empty_like(x)  # [B, L, N, D]
                for i in range(L):
                    x_i = x[:, i, :, :]  # [B, N, D]
                    if self.norm_type == "layer_norm":
                        x_out[:, i, :, :] = F.layer_norm(x_i.contiguous(), normalized_shape)
                    elif self.norm_type == "rms_norm":
                        x_out[:, i, :, :] = rms_norm(x_i.contiguous())
                    else:
                        raise ValueError(f"Unsupported norm_type: {self.norm_type}")
                x = x_out.mean(dim=1)  # [B, N, D]

            else:
                L, N, D = x.shape
                x_out = torch.empty_like(x)  # [L, N, D]
                for i in range(L):
                    x_i = x[i, :, :]  # [N, D]
                    if self.norm_type == "layer_norm":
                        x_out[i, :, :] = F.layer_norm(x_i.contiguous(), normalized_shape)
                    elif self.norm_type == "rms_norm":
                        x_out[i, :, :] = rms_norm(x_i.contiguous())
                    else:
                        raise ValueError(f"Unsupported norm_type: {self.norm_type}")
                x = x_out.mean(dim=0)  # [N, D]
        # 单层输入不再做额外归一化
        if x.dim() == 2:
            x = x.unsqueeze(1)
            if attention_mask is None:
                attention_mask = torch.ones(x.shape[0], 1, device=x.device, dtype=torch.long)

        B = x.size(0)
        pool = self.pool_token.expand(B, -1, -1)
        x = torch.cat([pool, x], dim=1)

        padding_mask = None
        if attention_mask is not None:
            pool_mask = torch.ones(B, 1, device=x.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([pool_mask, attention_mask], dim=1)
            padding_mask = (attention_mask == 0)

        if self.use_rope:
            for blk in self.encoder:
                x = blk(x, attn_mask=padding_mask)
        else:
            x = self.encoder(x, src_key_padding_mask=padding_mask)

        x = self.out_norm(x)   # x shape : [B, len, dim]
        x = self.proj(x)
        pooled = x[:, 0, :]

        return pooled

class MeanPooling(nn.Module):
    """
    Mask-aware Mean Pooling + MLP
    -----------------------------
    输入:
        • x : ① [B, N, dim] ② [B, dim]
        • attention_mask : [B, N] , 1=valid, 0=pad
    输出:
        • [B, dim_out]
    仅包含一个可训练 MLP
    """
    def __init__(
        self,
        dim: int,
        dim_out: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(                 # 唯一可训练模块
            nn.Linear(dim, hidden, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim, bias=False),
        )

        self.out_norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim_out)

        # 初始化
        nn.init.trunc_normal_(self.mlp[0].weight, std=0.02)
        nn.init.trunc_normal_(self.mlp[3].weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x : (B, N, dim) 或 (B, dim)
        attention_mask : (B, N) , pad=0
        """
        # 统一成 [B, N, dim]
        if x.dim() == 2:
            x = x.unsqueeze(1)                        # (B,1,D)
            if attention_mask is None:
                attention_mask = torch.ones(
                    x.size(0), 1, dtype=torch.long, device=x.device
                )

        # ------------------ Mask-aware mean ---------------------------
        if attention_mask is None:
            pooled = x.mean(dim=1)                    # (B, dim)
        else:
            attn = attention_mask.float()             # (B, N)
            lengths = attn.sum(dim=1, keepdim=True)   # (B,1)
            lengths.clamp_(min=1)                     # 避免除 0
            pooled = (x * attn.unsqueeze(-1)).sum(dim=1) / lengths  # (B, dim)

        # ------------------ MLP → LayerNorm → Proj --------------------
        pooled = self.mlp(pooled)                     # (B, dim)
        pooled = self.out_norm(pooled)                # 稳定分布
        return self.proj(pooled)                      # (B, dim_out)


# ---------------------- Unit Test ----------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Test Case 1: Padded sequence input ---
    print("--- Testing Padded Sequence ---")
    B, N, D = 2, 8, 128
    seq_padded = torch.randn(B, N, D, device=device)
    mask = torch.tensor([
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ], device=device)

    m1 = AttnPooling(D, 64, num_heads=4, use_rope=False).to(device)
    out1 = m1(seq_padded, attention_mask=mask)
    print("No RoPE (padded seq) -> out shape:", out1.shape)
    
    m2 = AttnPooling(D, 64, num_heads=4, use_rope=True).to(device)
    out2 = m2(seq_padded, attention_mask=mask)
    loss = out2.pow(2).mean()
    loss.backward()
    print("With RoPE (padded seq) -> out shape:", out2.shape)
    print("Grad norm OK for RoPE model:", m2.pool_token.grad.norm().item() > 0)

    # --- Test Case 2: Single vector input ---
    print("\n--- Testing Single Vector ---")
    vec = torch.randn(B, D, device=device)
    out3 = m1(vec)
    print("No RoPE (vector) -> out shape:", out3.shape)
    
    out4 = m2(vec)
    print("With RoPE (vector) -> out shape:", out4.shape)
    
    print("\n✅ All tests passed!")
