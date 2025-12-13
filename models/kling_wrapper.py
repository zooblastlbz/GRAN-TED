from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Optional, Tuple, Any

import sys
sys.path.append("/ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/proj_text_enc/attn_pool_contrastive/models")  # noqa
from qwen2_llmorph import Qwen2TextEncoder

class KlingBaseWrapper(nn.Module):

    def __init__(
        self,
        *,
        max_sequence_length: int = 1024,
        device: str = "cuda",
        bf16: bool = True,                      
        clean_caption: bool = False,            
        do_classifier_free_guidance: bool = False,
        return_pooled: bool = False,            # False→(B,T,D)；True→(B,1,D)
    ):
        super().__init__()
        self.device = torch.device(device)
        self.max_seq_len = max_sequence_length
        self.clean_caption = clean_caption
        self.do_cfg = do_classifier_free_guidance
        self.return_pooled = return_pooled

        # dtype
        if bf16 and torch.cuda.is_available():
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float16

        text_encoder = Qwen2TextEncoder()

        self.text_encoder = text_encoder.to(self.dtype).to(self.device).eval()
        self.text_encoder.requires_grad_(False)

    @torch.inference_mode()
    def forward(
        self,
        *,
        texts: Union[str, List[str]],
        input_ids=None,
        attention_mask=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if input_ids is not None:
            raise ValueError("KlingBaseWrapper 仅支持原始文本 `texts` 输入（不接收 input_ids）。")

        if isinstance(texts, str):
            texts = [texts]

        # kling 的 encode_prompt：返回 (prompt_embeds, prompt_attention_mask, _, _)
        prompt_embeds, prompt_attention_mask, _, _ = self.text_encoder.encode_prompt(
            texts,
            do_classifier_free_guidance=self.do_cfg,
            device=self.text_encoder.text_encoder.device
                if hasattr(self.text_encoder, "text_encoder") else self.device,
            clean_caption=self.clean_caption,
            max_sequence_length=self.max_seq_len,
        )
        # 期望形状：embeds (B,T,D), mask (B,T)
        embeds = prompt_embeds.to(self.device, dtype=self.dtype)
        mask = prompt_attention_mask.to(self.device).long()

        if not self.return_pooled:
            # 返回原生序列向量
            return embeds, mask

        # 句子级单向量：mask-aware mean pooling
        # mask: (B,T)→(B,T,1) 以便广播
        mask_f = mask.unsqueeze(-1).to(embeds.dtype)  # (B,T,1)
        denom = torch.clamp(mask_f.sum(dim=1, keepdim=True), min=1.0)  # (B,1,1)
        pooled = (embeds * mask_f).sum(dim=1, keepdim=True) / denom    # (B,1,D)

        pooled_mask = torch.ones(pooled.size(0), 1, dtype=torch.long, device=self.device)
        return pooled, pooled_mask



if __name__ == "__main__":
    wrapper = KlingBaseWrapper(
        max_sequence_length=2048,
        device="cuda:0",
        bf16=True,
        clean_caption=False,
        do_classifier_free_guidance=False,
        return_pooled=False,  # True→句向量 (B,1,D)；False→序列 (B,T,D)
    )

    texts = [
        "A cozy living room scene features an anthropomorphic cat, dressed in long sleeves and an apron...",
        "A cozy living room scene features an anthropomorphic cat."
    ]
    embeds, mask = wrapper(texts=texts)
    print("embeds:", embeds.shape)  # (B,T,D) 或 (B,1,D)
    print("mask  :", mask.shape)    # (B,T)   或 (B,1)
