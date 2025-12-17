from __future__ import annotations
from typing import List, Union, Tuple, Optional
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2_5_VLForConditionalGeneration, AutoModel,Qwen3VLForConditionalGeneration
# from .TextEncoder.llava.model.language_model.modeling_llava_video_siglip import LlavaForConditionalGeneration as KwaiLlavaForConditionalGeneration

class Qwen25Wrapper(nn.Module):
    """
    冻结版 Qwen-2.5 编码器
    ----------------------
    · 输入 1: input_ids & attention_mask (Tensor batch)
    · 输入 2: texts / list[str]          (普通句子)
    返回: hidden_state [B,N,dim]  &  attention_mask [B,N]
    """
    def __init__(
        self,
        model_name: str = "path/to/Qwen2.5",
        device: str = "cuda",
        bf16: bool = True,
        max_len: int = 2048,
        layers_to_select: int = -1,
        select_all_layers_or_not: bool = False,
    ):
        super().__init__()
        self.device  = torch.device(device)
        self.max_len = max_len
        dtype = torch.bfloat16 if bf16 else torch.float16
        self.layers_to_select = layers_to_select
        self.select_all_layers_or_not = select_all_layers_or_not

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).eval().to(self.device)
        self.model.requires_grad_(False)
    
    # def get_num_llm_layers(self) -> int:


    # -------- util: texts -> Tensor --------
    def _tokenize_batch(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            padding="longest",          # 明确指定
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return enc.input_ids.to(self.device), enc.attention_mask.to(self.device)

    # -------- public forward --------
    @torch.inference_mode()
    def forward(
        self,
        *,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        texts: Optional[Union[str, List[str]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # ------ 文本模式 ------
        if input_ids is None:
            if texts is None:
                raise ValueError("Provide either `input_ids` or `texts`")
            if isinstance(texts, str):
                texts = [texts]
            input_ids, attention_mask = self._tokenize_batch(texts)

        # ------ 张量模式 ------
        else:
            input_ids = input_ids.to(self.device)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, device=self.device)
            else:
                attention_mask = attention_mask.to(self.device)

        # 前向
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False
        )
        if self.select_all_layers_or_not:
            # 选择所有层的 hidden states，每层  apply mean normalization to each layer’s embeddings before averaging
            hidden = torch.stack(out.hidden_states[1:], dim=1)        # (B, L, N, D)
            # layer_mean = hidden.mean(dim=2, keepdim=True)         # (B, L, 1, D)
            # hidden = hidden - layer_mean                          # 去均值

            # layer_max = hidden.max(dim=2, keepdim=True)[0]        # (B, L, 1, D)
            # layer_min = hidden.min(dim=2, keepdim=True)[0]        # (B, L, 1, D)
            # layer_range = (layer_max - layer_min).clamp(min=1e-6) # 避免除 0

            # hidden = hidden / layer_range                         # 缩放到 (-1,1) 左右
            # hidden = hidden.mean(dim=1)                           # (B, N, D)
        else:
            hidden = out.hidden_states[self.layers_to_select]  # 选择指定层的 hidden states [B, N, dim]

        return hidden, attention_mask


class Qwen25VLWrapper(nn.Module):
    """
    Qwen-2.5-VL 纯文本编码器（模型冻结）
    ---------------------------------
    forward 支持两条路径:
      • 张量模式: input_ids (+ attention_mask)
      • 文本模式: texts / list[str]
    返回:
      hidden_states  [B, N, dim]
      attention_mask [B, N]
    """

    def __init__(
        self,
        model_name: str = "path/to/Qwen2.5-VL",
        device: str = "cuda",
        bf16: bool = True,
        max_len: int = 2048,
        layers_to_select: int = -1,
        select_all_layers_or_not: bool = False,
    ):
        super().__init__()
        self.device  = torch.device(device)
        self.max_len = max_len

        dtype = torch.bfloat16 if bf16 else torch.float16
        self.layers_to_select = layers_to_select
        self.select_all_layers_or_not = select_all_layers_or_not
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).eval().to(self.device)
        self.model.requires_grad_(False)

    # --------- 文本批量 → tensor ---------
    def _tokenize_batch(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return enc.input_ids.to(self.device), enc.attention_mask.to(self.device)

    # -------------- forward --------------
    @torch.inference_mode()
    def forward(
        self,
        *,
        # ~~ tensor path ~~
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        # ~~ text path ~~
        texts: Optional[Union[str, List[str]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # ===== 文本路径 =====
        if input_ids is None:
            if texts is None:
                raise ValueError("Either `texts` or `input_ids` must be provided.")
            if isinstance(texts, str):
                texts = [texts]
            input_ids, attention_mask = self._tokenize_batch(texts)

        # ===== 张量路径 =====
        else:
            input_ids = input_ids.to(self.device)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, device=self.device)
            else:
                attention_mask = attention_mask.to(self.device)

        # 前向推理，取最后 hidden
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,          # 关闭 KV-Cache，节省显存
        )
        if self.select_all_layers_or_not:

            hidden = torch.stack(out.hidden_states[1:], dim=1)        # (B, L, N, D)

        else:
            hidden = out.hidden_states[self.layers_to_select]  # 选择指定层的 hidden states [B, N, dim]

        return hidden, attention_mask


class Qwen3Wrapper(nn.Module):
    """
    冻结版 Qwen-3 编码器
    --------------------
    • 输入 1) `input_ids` + `attention_mask`         (batch tensor)
    • 输入 2) `texts` / list[str]                   (原始句子)
    • 返回  : (`hidden_states` [B,N,dim] , `attention_mask` [B,N])
    • 支持:
        – 选定单层 (layers_to_select)
        – 或对所有层做归一化后平均 (select_all_layers_or_not=True)
    """
    def __init__(
        self,
        model_name: str = "path/to/Qwen3",
        device: str = "cuda",
        bf16: bool = True,
        max_len: int = 4096,
        layers_to_select: int = -1,
        select_all_layers_or_not: bool = False,
    ):
        super().__init__()
        self.device  = torch.device(device)
        self.max_len = max_len
        self.layers_to_select = layers_to_select
        self.select_all_layers_or_not = select_all_layers_or_not

        dtype = torch.bfloat16 if bf16 else torch.float16

        # --- tokenizer & model ---
        # Qwen3 的 tokenizer 不强制使用 chat_template；这里只做纯文本编码
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).eval().to(self.device)
        # 冻结参数
        self.model.requires_grad_(False)

    # ------------------------------------------------------------
    # util: 批量文本 → (input_ids, attention_mask)
    # ------------------------------------------------------------
    def _tokenize_batch(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return enc.input_ids.to(self.device), enc.attention_mask.to(self.device)

    # ------------------------------------------------------------
    # forward : 统一入口
    # ------------------------------------------------------------
    @torch.inference_mode()
    def forward(
        self,
        *,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        texts: Optional[Union[str, List[str]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # ---- 文本路径 -------------------------------------------------------
        if input_ids is None:
            if texts is None:
                raise ValueError("Provide either `input_ids` or `texts`.")
            if isinstance(texts, str):
                texts = [texts]
            input_ids, attention_mask = self._tokenize_batch(texts)

        # ---- 张量路径 -------------------------------------------------------
        else:
            input_ids = input_ids.to(self.device)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, device=self.device)
            else:
                attention_mask = attention_mask.to(self.device)

        # ---- 前向推理 -------------------------------------------------------
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,   # 必须打开
            use_cache=False,
        )

        # ---- 取 hidden states ---------------------------------------------
        if self.select_all_layers_or_not:
            # → shape (B, L, N, D)
            hidden = torch.stack(out.hidden_states[1:], dim=1)  # (B, L, N, D)

            # # 每层均值归一化 → [-1,1] 区间
            # layer_mean  = hidden.mean(dim=2, keepdim=True)
            # hidden = hidden - layer_mean
            # layer_max   = hidden.max(dim=2, keepdim=True)[0]
            # layer_min   = hidden.min(dim=2, keepdim=True)[0]
            # layer_range = (layer_max - layer_min).clamp(min=1e-6)
            # hidden = hidden / layer_range

            # # 跨层平均 → (B, N, D)
            # hidden = hidden.mean(dim=1)

        else:
            # 直接取指定层 (默认 -1 = 最后一层)
            hidden = out.hidden_states[self.layers_to_select]   # (B,N,D)

        return hidden, attention_mask

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Union, List, Tuple

class Llama3Wrapper(nn.Module):
    """
    冻结版 Llama-3.1 编码器
    --------------------
    • 输入 1) `input_ids` + `attention_mask`         (batch tensor)
    • 输入 2) `texts` / list[str]                   (原始句子)
    • 返回  : (`hidden_states` [B,N,dim] , `attention_mask` [B,N])
    • 支持:
        – 选定单层 (layers_to_select)
        – 或对所有层做归一化后平均 (select_all_layers_or_not=True)
    """
    def __init__(
        self,
        device,
        model_name: str = "path/to/Llama3.1",
        bf16: bool = True,
        max_len: int = 4096,
        layers_to_select: int = -1,
        select_all_layers_or_not: bool = False,
    ):
        super().__init__()
        self.device  = torch.device(device)
        self.max_len = max_len
        self.layers_to_select = layers_to_select
        self.select_all_layers_or_not = select_all_layers_or_not

        dtype = torch.bfloat16 if bf16 else torch.float16

        # --- tokenizer & model ---
        # Llama3.1 的 tokenizer 不强制使用 chat_template；这里只做纯文本编码
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        # 设置 pad_token（Llama 没有预设 pad_token）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).eval().to(self.device)
        # 冻结参数
        self.model.requires_grad_(False)

    # ------------------------------------------------------------
    # util: 批量文本 → (input_ids, attention_mask)
    # ------------------------------------------------------------
    def _tokenize_batch(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return enc.input_ids.to(self.device), enc.attention_mask.to(self.device)

    # ------------------------------------------------------------
    # forward : 统一入口
    # ------------------------------------------------------------
    @torch.inference_mode()
    def forward(
        self,
        *,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        texts: Optional[Union[str, List[str]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # ---- 文本路径 -------------------------------------------------------
        if input_ids is None:
            if texts is None:
                raise ValueError("Provide either `input_ids` or `texts`.")
            if isinstance(texts, str):
                texts = [texts]
            input_ids, attention_mask = self._tokenize_batch(texts)

        # ---- 张量路径 -------------------------------------------------------
        else:
            input_ids = input_ids.to(self.device)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, device=self.device)
            else:
                attention_mask = attention_mask.to(self.device)

        # ---- 前向推理 -------------------------------------------------------
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,   # 必须打开
            use_cache=False,
        )

        # ---- 取 hidden states ---------------------------------------------
        if self.select_all_layers_or_not:
            # → shape (B, L, N, D)
            hidden = torch.stack(out.hidden_states[1:], dim=1)  # (B, L, N, D)

            # # 每层均值归一化 → [-1,1] 区间
            # layer_mean  = hidden.mean(dim=2, keepdim=True)
            # hidden = hidden - layer_mean
            # layer_max   = hidden.max(dim=2, keepdim=True)[0]
            # layer_min   = hidden.min(dim=2, keepdim=True)[0]
            # layer_range = (layer_max - layer_min).clamp(min=1e-6)
            # hidden = hidden / layer_range

            # # 跨层平均 → (B, N, D)
            # hidden = hidden.mean(dim=1)

        else:
            # 直接取指定层 (默认 -1 = 最后一层)
            hidden = out.hidden_states[self.layers_to_select]   # (B,N,D)

        return hidden, attention_mask
        
class Qwen3VLWrapper(nn.Module):
    """
    Qwen3-VL 纯文本编码器（模型冻结）
    ---------------------------------
    forward 支持两条路径:
      • 张量模式: input_ids (+ attention_mask)
      • 文本模式: texts / list[str]
    返回:
      hidden_states  [B, N, dim]
      attention_mask [B, N]
    """

    def __init__(
        self,
        # ⬇️ **修改**: 请替换为你自己的 Qwen3-VL 模型路径
        layers_to_select,
        model_name: str = "path/to/Qwen3-VL",
        device: str = "cuda",
        bf16: bool = True,
        max_len: int = 2048,
        select_all_layers_or_not: bool = False,
    ):
        super().__init__()
        self.device  = torch.device(device)
        self.max_len = max_len

        dtype = torch.bfloat16 if bf16 else torch.float16
        self.layers_to_select = layers_to_select
        self.select_all_layers_or_not = select_all_layers_or_not
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )

            
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

        self.model=model.language_model.eval().to(self.device)
        self.model.requires_grad_(False)

    # --------- 文本批量 → tensor ---------
    def _tokenize_batch(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """ (此函数与 Qwen2.5-VL 版本相同) """
        enc = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return enc.input_ids.to(self.device), enc.attention_mask.to(self.device)

    # -------------- forward --------------
    @torch.inference_mode()
    def forward(
        self,
        *,
        # ~~ tensor path ~~
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        # ~~ text path ~~
        texts: Optional[Union[str, List[str]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ (此函数的核心逻辑与 Qwen2.5-VL 版本相同) """

        # ===== 文本路径 =====
        if input_ids is None:
            if texts is None:
                raise ValueError("Either `texts` or `input_ids` must be provided.")
            if isinstance(texts, str):
                texts = [texts]
            input_ids, attention_mask = self._tokenize_batch(texts)

        # ===== 张量路径 =====
        else:
            input_ids = input_ids.to(self.device)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, device=self.device)
            else:
                attention_mask = attention_mask.to(self.device)

        # 前向推理，取最后 hidden
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,          # 关闭 KV-Cache，节省显存
        )
        
        if self.select_all_layers_or_not:
        
            hidden = torch.stack(out.hidden_states[1:], dim=1)        # (B, L, N, D)

        else:
            # (逻辑保持不变)
          
            hidden = out.hidden_states[self.layers_to_select] 


        return hidden, attention_mask

class InternVL3Wrapper(nn.Module):
    """
    InternVL‑3 **纯文本**编码器（冻结）
    ---------------------------------
    • 输入两种模式  
      1. `input_ids` + `attention_mask`         (Tensor batch)  
      2. `texts` / list[str]                   (原始句子)

    • 输出  
      `(hidden_states  [B,N,dim] , attention_mask [B,N])`

    • 支持  
      – 选定单层 (`layers_to_select`)  
      – 或对所有层归一化后平均 (`select_all_layers_or_not=True`)
    """

    def __init__(
        self,
        model_name: str = "path/to/InternVL-3",
        device: str = "cuda",
        bf16: bool = True,
        max_len: int = 2048,
        layers_to_select: int = -1,
        select_all_layers_or_not: bool = False,
    ):
        """
        Args
        ----
        model_name : HuggingFace Repo / 本地路径  
        max_len    : tokenizer 截断长度  
        hf_kwargs  : 其余传给 `AutoModel.from_pretrained` 的关键字  
                     （如 `device_map`, `load_in_8bit`, `torch_dtype`…）
        """
        super().__init__()
        self.device  = torch.device(device)
        self.max_len = max_len
        self.layers_to_select         = layers_to_select
        self.select_all_layers_or_not = select_all_layers_or_not

        # ----- dtype -----
        dtype = torch.bfloat16 if bf16 else torch.float16

        # ----- tokenizer & model -----
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).eval().to(self.device)

        # 冻结所有参数
        self.model.requires_grad_(False)

    # ---------------------------------------------------------------------
    # util : 批量文本 → (input_ids , attention_mask)
    # ---------------------------------------------------------------------
    def _tokenize_batch(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return enc.input_ids.to(self.device), enc.attention_mask.to(self.device)

    # ---------------------------------------------------------------------
    # forward
    # ---------------------------------------------------------------------
    @torch.inference_mode()
    def forward(
        self,
        *,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        texts: Optional[Union[str, List[str]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        hidden : Tensor  (B, N, D)  
        mask   : Tensor  (B, N)
        """
        # ===== 文本路径 =====
        if input_ids is None:
            if texts is None:
                raise ValueError("Provide either `texts` or `input_ids`.")
            if isinstance(texts, str):
                texts = [texts]
            input_ids, attention_mask = self._tokenize_batch(texts)

        # ===== 张量路径 =====
        else:
            input_ids = input_ids.to(self.device)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, device=self.device)
            else:
                attention_mask = attention_mask.to(self.device)
        # print(input_ids)
        # print(input_embeds)
        # ===== 仅跑语言模型 =====
        # InternVL‑3: language_model == Llama‑like backbone
        out = self.model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        if self.select_all_layers_or_not:
            # (B, L, N, D)
            hidden = torch.stack(out.hidden_states[1:], dim=1)  # (B, L, N, D)
        else:
            hidden = out.hidden_states[self.layers_to_select]  # (B,N,D)
        # print(hidden)

        return hidden, attention_mask


class MiniCPMWrapper(nn.Module):
    """
    ---------------------------------
    • 输入两种模式  
      1. `input_ids` + `attention_mask`         (Tensor batch)  
      2. `texts` / list[str]                   (原始句子)

    • 输出  
      `(hidden_states  [B,N,dim] , attention_mask [B,N])`

    • 支持  
      – 选定单层 (`layers_to_select`)  
      – 或对所有层归一化后平均 (`select_all_layers_or_not=True`)
    """

    def __init__(
        self,
        model_name: str = "path/to/MiniCPM",
        device: str = "cuda",
        bf16: bool = True,
        max_len: int = 2048,
        layers_to_select: int = -1,
        select_all_layers_or_not: bool = False,
    ):
        """
        Args
        ----
        model_name : HuggingFace Repo / 本地路径  
        max_len    : tokenizer 截断长度  
        hf_kwargs  : 其余传给 `AutoModel.from_pretrained` 的关键字  
                     （如 `device_map`, `load_in_8bit`, `torch_dtype`…）
        """
        super().__init__()
        self.device  = torch.device(device)
        self.max_len = max_len
        self.layers_to_select         = layers_to_select
        self.select_all_layers_or_not = select_all_layers_or_not

        # ----- dtype -----
        dtype = torch.bfloat16 if bf16 else torch.float16

        # ----- tokenizer & model -----
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation='sdpa',
        ).eval().to(self.device)

        # 冻结所有参数
        self.model.requires_grad_(False)

    # ---------------------------------------------------------------------
    # util : 批量文本 → (input_ids , attention_mask)
    # ---------------------------------------------------------------------
    def _tokenize_batch(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return enc.input_ids.to(self.device), enc.attention_mask.to(self.device)

    # ---------------------------------------------------------------------
    # forward
    # ---------------------------------------------------------------------
    @torch.inference_mode()
    def forward(
        self,
        *,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        texts: Optional[Union[str, List[str]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        hidden : Tensor  (B, N, D)  
        mask   : Tensor  (B, N)
        """
        # ===== 文本路径 =====
        if input_ids is None:
            if texts is None:
                raise ValueError("Provide either `texts` or `input_ids`.")
            if isinstance(texts, str):
                texts = [texts]
            input_ids, attention_mask = self._tokenize_batch(texts)

        # ===== 张量路径 =====
        else:
            input_ids = input_ids.to(self.device)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, device=self.device)
            else:
                attention_mask = attention_mask.to(self.device)
        # print(input_ids)
        # print(input_embeds)
        # ===== 仅跑语言模型 =====
        # InternVL‑3: language_model == Llama‑like backbone
        out = self.model.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        if self.select_all_layers_or_not:
            # (B, L, N, D)
            hidden = torch.stack(out.hidden_states[1:], dim=1)  # (B, L, N, D)
        else:
            hidden = out.hidden_states[self.layers_to_select]  # (B,N,D)
        # print(hidden)

        return hidden, attention_mask


class OvisWrapper(nn.Module):
    """
    ---------------------------------
    • 输入两种模式  
      1. `input_ids` + `attention_mask`         (Tensor batch)  
      2. `texts` / list[str]                   (原始句子)

    • 输出  
      `(hidden_states  [B,N,dim] , attention_mask [B,N])`

    • 支持  
      – 选定单层 (`layers_to_select`)  
      – 或对所有层归一化后平均 (`select_all_layers_or_not=True`)
    """

    def __init__(
        self,
        model_name: str = "path/to/Ovis",
        device: str = "cuda",
        bf16: bool = True,
        max_len: int = 2048,
        layers_to_select: int = -1,
        select_all_layers_or_not: bool = False,
    ):
        """
        Args
        ----
        model_name : HuggingFace Repo / 本地路径  
        max_len    : tokenizer 截断长度  
        hf_kwargs  : 其余传给 `AutoModel.from_pretrained` 的关键字  
                     （如 `device_map`, `load_in_8bit`, `torch_dtype`…）
        """
        super().__init__()
        self.device  = torch.device(device)
        self.max_len = max_len
        self.layers_to_select         = layers_to_select
        self.select_all_layers_or_not = select_all_layers_or_not

        # ----- dtype -----
        dtype = torch.bfloat16 if bf16 else torch.float16

        # ----- tokenizer & model -----
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).eval().to(self.device)

        # 冻结所有参数
        self.model.requires_grad_(False)

    # ---------------------------------------------------------------------
    # util : 批量文本 → (input_ids , attention_mask)
    # ---------------------------------------------------------------------
    def _tokenize_batch(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return enc.input_ids.to(self.device), enc.attention_mask.to(self.device)

    # ---------------------------------------------------------------------
    # forward
    # ---------------------------------------------------------------------
    @torch.inference_mode()
    def forward(
        self,
        *,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        texts: Optional[Union[str, List[str]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        hidden : Tensor  (B, N, D)  
        mask   : Tensor  (B, N)
        """
        # ===== 文本路径 =====
        if input_ids is None:
            if texts is None:
                raise ValueError("Provide either `texts` or `input_ids`.")
            if isinstance(texts, str):
                texts = [texts]
            input_ids, attention_mask = self._tokenize_batch(texts)

        # ===== 张量路径 =====
        else:
            input_ids = input_ids.to(self.device)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, device=self.device)
            else:
                attention_mask = attention_mask.to(self.device)
        # print(input_ids)
        # print(input_embeds)
        # ===== 仅跑语言模型 =====
        # InternVL‑3: language_model == Llama‑like backbone
        out = self.model.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        if self.select_all_layers_or_not:
            # (B, L, N, D)
            hidden = torch.stack(out.hidden_states[1:], dim=1)  # (B, L, N, D)
        else:
            hidden = out.hidden_states[self.layers_to_select]  # (B,N,D)
        # print(hidden)

        return hidden, attention_mask




# --------------------------- Demo ---------------------------
if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = MiniCPMWrapper()

    txts = ["A man in traditional attire sits and then stands up in a serene outdoor setting with bamboo and lanterns."]
    h, m = wrapper(texts=txts)

    print(h.shape, m.shape)

    print("✅ MiniCPMWrapper ready.")
    