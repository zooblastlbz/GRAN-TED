# models/embedding_model.py
from __future__ import annotations
from typing import List, Union, Sequence, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


class EmbeddingModel(nn.Module):
    """
    推理封装
    --------
    • 仅做前向，不更新参数
    • 支持分批 encode(List[str])，返回 (B, dim_out) Tensor
    """

    def __init__(
        self,
        wrapper: nn.Module,
        attn_pool: nn.Module,
        *,
        normalize: bool = True,          # 是否对输出做 L2-normalize
        device: Optional[str] = None,    # 若 None 则沿用 wrapper.device
    ):
        super().__init__()
        self.wrapper = wrapper.eval()
        self.attn_pool = attn_pool.eval()
        self.normalize = normalize

        # 推理全冻结
        for p in self.attn_pool.parameters():
            p.requires_grad_(False)

        # 推理设备：优先显式传入，否则跟 wrapper 保持一致
        self.device = torch.device(device) if device else next(wrapper.parameters()).device
        self.attn_pool.to(self.device)   # wrapper 已在自身 device

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def encode(
        self,
        texts: Union[str, Sequence[str]],
        batch_size: int = 32,
    ) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]

        all_embeds = []

        for idx in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[idx : idx + batch_size]

            # --- LLM wrapper 前向，取 hidden_states + mask ---
            hidden, mask = self.wrapper(texts=batch_texts)        # (B,N,D) , (B,N)
            
            dtype_target = next(self.attn_pool.parameters()).dtype   # fp16 or fp32

            if hidden.dtype != dtype_target:
                hidden = hidden.to(dtype_target)

            # --- Attention Pooling → (B,dim_out) ---
            embed = self.attn_pool(hidden, attention_mask=mask)   # (B,dim_out)

            if self.normalize:
                embed = F.normalize(embed, dim=-1)

            all_embeds.append(embed.cpu())   # 收集到 CPU，减显存压力

        # 拼回完整顺序
        return torch.cat(all_embeds, dim=0)

    # 兼容直接 __call__
    def forward(self, texts: Union[str, Sequence[str]], batch_size: int = 32):
        return self.encode(texts, batch_size)
