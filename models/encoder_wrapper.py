from __future__ import annotations
from typing import List, Tuple, Union, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from transformers import T5Tokenizer, T5EncoderModel

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Optional, Union, List, Tuple

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Optional, Union, List, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Optional, Union, List, Tuple


class CLIPTextWrapper(nn.Module):
    def __init__(
        self,
        model_name: str = "/ytech_m2v5_hdd/workspace/kling_mm/libozhou/Model/AltCLIP",
        device: str = "cuda",
        layers_to_select: int = -1,
        select_all_layers_or_not: bool = False,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.layers_to_select = layers_to_select
        self.select_all_layers_or_not = select_all_layers_or_not

        # ✅ 使用 AutoTokenizer 和 AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        full_model = AutoModel.from_pretrained(
            model_name,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).eval().to(self.device)

        # 提取文本编码器（AltCLIP 是 .text_model）
        self.text_encoder = full_model.text_model
        self.text_encoder.requires_grad_(False)

        # 配置信息
        self.hidden_size = self.text_encoder.config.hidden_size
        self.num_layers = self.text_encoder.config.num_hidden_layers
        self.max_context_length = self.text_encoder.config.max_position_embeddings

    @torch.inference_mode()
    def forward(
        self,
        *,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        texts: Optional[Union[str, List[str]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if input_ids is None:
            if texts is None:
                raise ValueError("Provide either `texts` or `input_ids`.")
            if isinstance(texts, str):
                texts = [texts]
            enc = self.tokenizer(
                texts,
                padding="longest",
                truncation=True,
                max_length=self.max_context_length,
                return_tensors="pt",
            )
            input_ids = enc.input_ids.to(self.device)
            attention_mask = enc.attention_mask.to(self.device)
        else:
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device) if attention_mask is not None else torch.ones_like(input_ids)

        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        if self.select_all_layers_or_not:
            all_hidden = torch.stack(outputs.hidden_states, dim=1)
            return all_hidden, attention_mask
        else:
            hidden = outputs.hidden_states[self.layers_to_select]
            return hidden, attention_mask

class UMT5Wrapper(nn.Module):
    """
    冻结版 **UMT5** Encoder
    -----------------------
    • 两种输入路径
      1) input_ids + attention_mask (Tensor batch)
      2) texts / list[str]         (原始句子)

    • 输出
      - 若 select_all_layers_or_not = False:
          (hidden_states[B, N, D], attention_mask[B, N])
          其中 hidden_states 为指定层 (layers_to_select, 默认 -1=最后一层)
      - 若 select_all_layers_or_not = True:
          (all_hidden[B, L, N, D], attention_mask[B, N])
          其中 all_hidden 为编码器“所有层 + embedding”的堆叠 (L = num_layers + 1)

    • 选层策略
      – layers_to_select: int，取单层 (默认 −1 = last)
      – select_all_layers_or_not=True: 返回所有层堆叠 (不做平均，便于外部自定义融合)
    """

    def __init__(
        self,
        model_name: str = "/ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/models/google/umt5-xxl",
        device: str = "cuda",
        bf16: bool = True,
        max_len: int = 2048,
        layers_to_select: int = -1,
        select_all_layers_or_not: bool = False,
    ):
        super().__init__()
        self.device  = torch.device(device)
        self.max_len = max_len
        self.layers_to_select         = layers_to_select
        self.select_all_layers_or_not = select_all_layers_or_not

        # CPU 上不强行用 bf16，避免不支持导致的数值/速度问题
        dtype = torch.bfloat16 if bf16 else torch.float16

        # --- Tokenizer & Model ------------------------------------------------
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        # 兜底：某些 ckpt 可能缺省 pad_token，统一设置为 eos
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = T5EncoderModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,  # 与你给的初始化保持一致
        ).eval().to(self.device)
        self.model.encoder.dropout = torch.nn.Dropout(0.0)

        # 只用 Encoder，全部冻结
        self.model.requires_grad_(False)

        # 可选：记录维度信息
        self.hidden_size = getattr(self.model.config, "d_model", None)

    # -------------------------------------------------------------------------
    # util : 批量文本 → (input_ids , attention_mask)
    # -------------------------------------------------------------------------
    def _tokenize_batch(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids      = enc.input_ids.to(self.device)
        attention_mask = enc.attention_mask.to(self.device)
        return input_ids, attention_mask

    # -------------------------------------------------------------------------
    # forward : 统一入口
    # -------------------------------------------------------------------------
    @torch.inference_mode()
    def forward(
        self,
        *,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        texts: Optional[Union[str, List[str]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # ---- 1) 文本路径 -----------------------------------------------------
        if input_ids is None:
            if texts is None:
                raise ValueError("Provide either `texts` or `input_ids`.")
            if isinstance(texts, str):
                texts = [texts]
            input_ids, attention_mask = self._tokenize_batch(texts)

        # ---- 2) 张量路径 -----------------------------------------------------
        else:
            input_ids = input_ids.to(self.device)
            if attention_mask is None:
                # 若未提供，默认所有位置可见
                attention_mask = torch.ones_like(input_ids, device=self.device)
            else:
                attention_mask = attention_mask.to(self.device)

        # ---- 3) Encoder 前向 (T5EncoderModel 已是 enc-only) ------------------
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # outputs.hidden_states: Tuple[(B, N, D)], len = num_layers + 1 (含词嵌入输出)
        # outputs.last_hidden_state: (B, N, D)

        if self.select_all_layers_or_not:
            # 形状 (B, L, N, D)
            all_hidden = torch.stack(outputs.hidden_states, dim=1)
            return all_hidden, attention_mask
        else:
            # 仅取指定单层 (默认 -1 = 最后一层)
            hidden = outputs.hidden_states[self.layers_to_select]
            return hidden, attention_mask





class T5GemmaWrapper(nn.Module):
    """
    冻结版 **T5-Gemma** Encoder
    ---------------------------
    • 两种输入路径  
      1. `input_ids` + `attention_mask` (Tensor batch)  
      2. `texts` / list[str]             (原始句子)

    • 输出  
      `(hidden_states  [B,N,dim] , attention_mask [B,N])`

    • 选层策略  
      – `layers_to_select`  : 只取单层 (默认 −1 = last)  
      – `select_all_layers_or_not=True` : 对所有层归一化后平均
    """

    def __init__(
        self,
        model_name: str = "/ytech_m2v5_hdd/workspace/kling_mm/Models/t5gemma-2b-2b-ul2",
        device: str = "cuda",
        bf16: bool = True,
        max_len: int = 2048,
        layers_to_select: int = -1,
        select_all_layers_or_not: bool = False,
    ):
        super().__init__()
        self.device  = torch.device(device)
        self.max_len = max_len
        self.layers_to_select        = layers_to_select
        self.select_all_layers_or_not = select_all_layers_or_not

        dtype = torch.bfloat16 if bf16 else torch.float16

        # --- Tokenizer & Model ------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        ).eval().to(self.device)

        # 只用 **Encoder** 参数，全部冻结
        self.model.requires_grad_(False)

    # -------------------------------------------------------------------------
    # util : 批量文本 → (input_ids , attention_mask)
    # -------------------------------------------------------------------------
    def _tokenize_batch(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            padding="longest",          # 保持 batch 对齐
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return enc.input_ids.to(self.device), enc.attention_mask.to(self.device)

    # -------------------------------------------------------------------------
    # forward : 统一入口
    # -------------------------------------------------------------------------
    @torch.inference_mode()
    def forward(
        self,
        *,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        texts: Optional[Union[str, List[str]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # ---- 1. 文本路径 -----------------------------------------------------
        if input_ids is None:
            if texts is None:
                raise ValueError("Provide either `texts` or `input_ids`.")
            if isinstance(texts, str):
                texts = [texts]
            input_ids, attention_mask = self._tokenize_batch(texts)

        # ---- 2. 张量路径 -----------------------------------------------------
        else:
            input_ids = input_ids.to(self.device)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, device=self.device)
            else:
                attention_mask = attention_mask.to(self.device)

        # ---- 3. Encoder 前向 -------------------------------------------------
        # 仅跑 encoder，节省显存 & 计算；需要 hidden_states
        encoder_outputs = self.model.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        if self.select_all_layers_or_not:
            # encoder_outputs.hidden_states : Tuple[(B,N,D)] , len = num_layers+1
            hidden = torch.stack(encoder_outputs.hidden_states, dim=1)  # (B,L,N,D)
        else:
            # 直接取指定层 (默认 last : -1)
            hidden = encoder_outputs.hidden_states[self.layers_to_select]  # (B,N,D)

        return hidden, attention_mask


# ----------------------------- quick test ------------------------------------
# ----------------------------- quick test ------------------------------------
if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = UMT5Wrapper(
        model_name="/ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/models/google/umt5-xxl",
        device=dev,
        bf16=True,
        max_len=1024,
        select_all_layers_or_not=True,  # 单层
        layers_to_select=-1,             # 取最后一层
    )

    # 1) 文本路径
    h, m = wrapper(texts=[
        "Test UMT5 encoder with short sentence.",
        "你好，世界！"
    ])
    print("[texts] hidden:", h.shape, "mask:", m.shape)

    # 2) 张量路径（模拟外部已 tokenized）
    toks = wrapper.tokenizer(
        ["another sample sentence"], padding=True, truncation=True,
        max_length=128, return_tensors="pt"
    )
    h2, m2 = wrapper(input_ids=toks.input_ids, attention_mask=toks.attention_mask)
    print("[tensor] hidden:", h2.shape, "mask:", m2.shape)

    print("✅ UMT5Wrapper ready.")
