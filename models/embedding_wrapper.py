from __future__ import annotations
from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# ---------- helper ----------
def last_token_pool(last_hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
    """
    从 Qwen3-Embedding 输出里取“最后一个非 pad token”向量。
    支持 left-pad / right-pad 两种方式。
    返回 shape: (B, D)
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:                       # 全部样本都是 left-pad
        return last_hidden_states[:, -1]   # 直接取最后位置
    else:                                  # right-pad
        seq_len = attention_mask.sum(dim=1) - 1          # 每行最后 token idx
        batch_idx = torch.arange(last_hidden_states.size(0),
                                 device=last_hidden_states.device)
        return last_hidden_states[batch_idx, seq_len]


# ============================================================
class _BaseQwen3Embed(nn.Module):
    """共用: tokenizer / model / tokenize"""
    def __init__(
        self,
        model_name: str="/ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/models/Qwen/Qwen3-Embedding-8B",
        device: str = "cuda",
        bf16: bool = True,
        max_len: int = 8192,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.max_len = max_len
        dtype = torch.bfloat16 if bf16 else torch.float16

        # left padding 是 Qwen3-Embedding 默认要求
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left", trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).eval().to(self.device)
        self.model.requires_grad_(False)

    # --------  util --------
    def _tokenize_batch(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            padding=True, truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return enc.input_ids.to(self.device), enc.attention_mask.to(self.device)

    # -------- 抽象接口 --------
    def forward(self, *args, **kwargs):
        raise NotImplementedError


# ============================================================ #
# 1)  返回全序列 hidden states                                  #
# ============================================================ #
class Qwen3EmbedSequenceWrapper(_BaseQwen3Embed):
    """
    返回 Qwen3-Embedding 的 last_hidden_state  —— shape [B, N, D]
    """
    def __init__(self,
                 model_name: str = "/ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/models/Qwen/Qwen3-Embedding-8B",
                 device: str = "cuda",
                 bf16: bool = True,
                 max_len: int = 8192,
                 layers_to_select: int = -1,
                 select_all_layers_or_not: bool = False,):
        super().__init__(model_name, device, bf16, max_len)
        self.layers_to_select = layers_to_select
        self.select_all_layers_or_not = select_all_layers_or_not

    @torch.inference_mode()
    def forward(
        self,
        *,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        texts: Optional[Union[str, List[str]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # -------- prepare inputs --------
        if input_ids is None:
            if texts is None:
                raise ValueError("Provide either `input_ids` or `texts`")
            if isinstance(texts, str):
                texts = [texts]
            input_ids, attention_mask = self._tokenize_batch(texts)
        else:
            input_ids = input_ids.to(self.device)
            attention_mask = (attention_mask.to(self.device)
                              if attention_mask is not None
                              else torch.ones_like(input_ids))

        # -------- forward --------
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        if self.select_all_layers_or_not:
            # 选择所有层的 hidden states，每层  apply mean normalization to each layer’s embeddings before averaging
            hidden = torch.stack(out.hidden_states, dim=1)        # (B, L, N, D)
            # layer_mean = hidden.mean(dim=2, keepdim=True)         # (B, L, 1, D)
            # hidden = hidden - layer_mean                          # 去均值

            # layer_max = hidden.max(dim=2, keepdim=True)[0]        # (B, L, 1, D)
            # layer_min = hidden.min(dim=2, keepdim=True)[0]        # (B, L, 1, D)
            # layer_range = (layer_max - layer_min).clamp(min=1e-6) # 避免除 0

            # hidden = hidden / layer_range                         # 缩放到 (-1,1) 左右
            # hidden = hidden.mean(dim=1)                           # (B, N, D)
        else:
            hidden = out.hidden_states[self.layers_to_select]     # 选择指定层的 hidden states [B, N, dim]

        return hidden, attention_mask


# ============================================================ #
# 2)  返回句级 embedding  (last_token_pool)                      #
# ============================================================ #
class Qwen3EmbedEmbeddingWrapper(_BaseQwen3Embed):
    """
    取 last_token_pool 并扩维成 [B, 1, D]，attention_mask=[B,1]
    """
    def __init__(self,
                 model_name: str = "/ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/models/Qwen/Qwen3-Embedding-8B",
                 device: str = "cuda",
                 bf16: bool = True,
                 max_len: int = 8192,
                 layers_to_select: int = -1,
                 select_all_layers_or_not: bool = False,
                 normalize: bool = False):
        super().__init__(model_name, device, bf16, max_len)
        self.normalize = normalize     # 是否做 L2-normalize
        self.layers_to_select = layers_to_select
        self.select_all_layers_or_not = select_all_layers_or_not

    @torch.inference_mode()
    def forward(
        self,
        *,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        texts: Optional[Union[str, List[str]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # -------- prepare inputs --------
        if input_ids is None:
            if texts is None:
                raise ValueError("Provide either `input_ids` or `texts`")
            if isinstance(texts, str):
                texts = [texts]
            input_ids, attention_mask = self._tokenize_batch(texts)
        else:
            input_ids = input_ids.to(self.device)
            attention_mask = (attention_mask.to(self.device)
                              if attention_mask is not None
                              else torch.ones_like(input_ids))

        # -------- forward --------
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # ---- choose hidden seq ----
        if self.select_all_layers_or_not:
            hidden = torch.stack(out.hidden_states, dim=1)      # (B,L,N,D)
            # layer_mean = hidden.mean(dim=2, keepdim=True)
            # hidden = hidden - layer_mean
            # rng = (hidden.max(dim=2, keepdim=True)[0] -
            #        hidden.min(dim=2, keepdim=True)[0]).clamp_(min=1e-6)
            # seq = (hidden / rng).mean(dim=1)                        # (B,N,D)
            seq = hidden
        elif self.layers_to_select not in (-1, None):
            seq = out.hidden_states[self.layers_to_select]      # (B,N,D)
        else:
            seq = out.last_hidden_state                         # (B,N,D)


        # last_token_pool -> (B, D)
        emb = last_token_pool(seq, attention_mask)
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=-1)

        emb = emb.unsqueeze(1)                     # (B, 1, D)
        mask = torch.ones(emb.size(0), 1, device=self.device, dtype=torch.long)
        return emb, mask


class JINAv4Wrapper(nn.Module):
    """
    Jina Embeddings v4 文本编码 wrapper（模型冻结）
    ------------------------------------------------
    • 仅支持文本输入：`texts` / list[str]
    • 两种输出模式：
        1) `return_multivector=True`  → 句子 → 多向量 (N_i,D)
           – 自动 pad 到同长 → [B, N_max, D] + mask [B, N_max]
        2) `return_multivector=False` → 句子级单向量
           – 输出 [B, 1, D] + mask [B,1]
    """
    def __init__(
        self,
        model_name: str = "/ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/models/jinaai/jina-embeddings-v4",
        *,
        task: str = "text-matching",           # retrieval / text-matching / code ...
        prompt_name: Optional[str] = None, # "query" / "passage" / None
        return_multivector: bool = True,
        device: str = "cuda",
        bf16: bool = True,
        normalize: bool = True,            # L2-norm on each vector
    ):
        super().__init__()
        self.device = torch.device(device)
        self.task   = task
        self.prompt = prompt_name
        self.return_multivector = return_multivector
        self.normalize = normalize

        dtype = torch.bfloat16 if bf16 else torch.float16

        # Jina v4 模型在 AutoModel 上自动注册 encode_text / encode_image
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).eval().to(self.device)
        self.model.requires_grad_(False)

    # ------------------------------------------------------------
    # forward 仅支持文本路径
    # ------------------------------------------------------------
    @torch.inference_mode()
    def forward(
        self,
        *,
        texts: Union[str, List[str]],
        # 兼容其它 wrapper 的参数占位
        input_ids=None,
        attention_mask=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if input_ids is not None:
            raise ValueError("JINAv4Wrapper 仅支持原始文本 `texts` 输入")

        if isinstance(texts, str):
            texts = [texts]

        # ---- 编码 ---------------------------------------------------------
        embeds = self.model.encode_text(
            texts=texts,
            task=self.task,
            prompt_name=self.prompt,
            return_multivector=self.return_multivector,
        )  # list[Tensor] 或 Tensor

        # ---- 处理单向量模式 ----------------------------------------------
        if not self.return_multivector:
            if isinstance(embeds, list):              # 新版 transformers 仍返回 list
                embeds = torch.stack(embeds, 0)       # (B,D)
            if self.normalize:
                embeds = F.normalize(embeds, p=2, dim=-1)
            embeds = embeds.to(self.device).unsqueeze(1)      # (B,1,D)
            mask   = torch.ones(embeds.size(0), 1,
                                dtype=torch.long, device=self.device)
            return embeds, mask

        # ---- 多向量模式 ---------------------------------------------------
        # embeds: list[Tensor (N_i,D)]
        batch_size = len(embeds)
        # L2-norm 每个向量
        if self.normalize:
            embeds = [F.normalize(e, p=2, dim=-1) for e in embeds]

        max_len = max(e.size(0) for e in embeds)
        dim     = embeds[0].size(1)
        dtype   = embeds[0].dtype

        pad_tensor = torch.zeros(batch_size, max_len, dim,
                                 dtype=dtype, device=self.device)
        pad_mask   = torch.zeros(batch_size, max_len,
                                 dtype=torch.long, device=self.device)

        for i, e in enumerate(embeds):
            n = e.size(0)
            pad_tensor[i, :n] = e.to(self.device)
            pad_mask[i, :n]   = 1

        return pad_tensor, pad_mask
