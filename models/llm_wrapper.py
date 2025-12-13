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
        model_name: str = "/ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/models/Qwen/Qwen2.5-7B",
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
        model_name: str = "/ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/models/Qwen/Qwen2.5-VL-7B-Instruct",
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
        model_name: str = "/ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/models/Qwen/Qwen3-8B-Base",
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
        model_name: str = "/ytech_m2v5_hdd/workspace/kling_mm/Models/Llama3-8b-Instruct/",
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
        model_name: str = "/ytech_m2v5_hdd/workspace/kling_mm/Models/Qwen3-VL-8B-Instruct/",
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
        try:
            
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
            ).eval().to(self.device)
            print(f"Loaded as AutoModelForCausalLM")
        except:
            # 如果失败，使用通用AutoModel
           
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
            ).eval().to(self.device)
            print(f"Loaded as AutoModel")
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
            # (逻辑保持不变)
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
            # (逻辑保持不变)
            if isinstance(self.layers_to_select,int):
                hidden = out.hidden_states[self.layers_to_select]  # 选择指定层的 hidden states [B, N, dim]
            elif isinstance(self.layers_to_select,list):
                hidden = torch.stack([out.hidden_states[i] for i in self.layers_to_select],dim=1)
            elif isinstance(self.layers_to_select,str):
                begin=int(self.layers_to_select.split(':')[0])
                end=int(self.layers_to_select.split(':')[-1])
                hidden_list=[i for i in range(begin,end,1)]
                hidden = torch.stack([out.hidden_states[i] for i in hidden_list],dim=1)

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
        model_name: str = "/ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/models/OpenGVLab/InternVL3-8B-hf",
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
        model_name: str = "/ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/models/openbmb/MiniCPM-V-4_5",
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
        model_name: str = "/ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/models/AIDC-AI/Ovis2.5-9B",
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


class KwaiLlavaWrapper(nn.Module):
    """
    冻结版 **Kwai‑LLaVA** 纯文本编码器
    --------------------------------
    · 输入 1) `input_ids` + `attention_mask`      (Tensor batch)
    · 输入 2) `texts` / list[str]                (原始句子)
    · 返回   (`hidden_states` [B,N,dim] , `attention_mask` [B,N])

    支持两种取 hidden 方式：
        • `layers_to_select` 指定单层 (默认 -1 即最后一层)
        • `select_all_layers_or_not=True` 时，返回所有层 (B,L,N,D)
          ——若要进一步聚合，可在外部做 mean/max 等操作
    """

    def __init__(
        self,
        model_name: str = "/mmu_mllm_hdd/yangsihan05/text_enc/text_enc-KwaiYii-7B-Qwen2_2025_6_9_most_data_2025_06_09_22_46/checkpoint-7750/hf_new",  # ← 修改为本地 ckpt 或 HF 地址
        device: str = "cuda",
        bf16: bool = True,
        max_len: int = 4096,
        layers_to_select: int = -1,
        select_all_layers_or_not: bool = False,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.max_len = max_len
        self.layers_to_select = layers_to_select
        self.select_all_layers_or_not = select_all_layers_or_not

        dtype = torch.bfloat16 if bf16 else torch.float16

        # --- tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        # --- model ---
        # KwaiLlavaForConditionalGeneration 包含：
        #   • vision_tower
        #   • multi_modal_projector
        #   • language_model (KwaiYiiForCausalLM)
        # 纯文本场景我们只用 language_model
        full_model = KwaiLlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        full_model.eval().to(self.device)
        full_model.requires_grad_(False)

        # 仅保留 language 模块，节省显存
        self.llm = full_model

    # --------------------------------------------------
    # util : texts -> (input_ids , attention_mask)
    # --------------------------------------------------
    def _tokenize_batch(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return enc.input_ids.to(self.device), enc.attention_mask.to(self.device)

    # --------------------------------------------------
    # forward
    # --------------------------------------------------
    @torch.inference_mode()
    def forward(
        self,
        *,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        texts: Optional[Union[str, List[str]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        与其它 wrapper 相同的统一接口。
        """
        # ---------- 文本路径 ----------
        if input_ids is None:
            if texts is None:
                raise ValueError("Provide either `input_ids` or `texts`.")
            if isinstance(texts, str):
                texts = [texts]
            input_ids, attention_mask = self._tokenize_batch(texts)

        # ---------- 张量路径 ----------
        else:
            input_ids = input_ids.to(self.device)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, device=self.device)
            else:
                attention_mask = attention_mask.to(self.device)

        # ---------- 前向 ----------
        outputs = self.llm.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # ---------- hidden states ----------
        if self.select_all_layers_or_not:
            # 返回所有层：(B, L, N, D)
            hidden = torch.stack(outputs.hidden_states[1:], dim=1)
        else:
            hidden = outputs.hidden_states[self.layers_to_select]  # (B,N,D)

        return hidden, attention_mask

# --------------------------- Demo ---------------------------
if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = MiniCPMWrapper()

    txts = ["FPS-25. A man in traditional attire sits and then stands up in a serene outdoor setting with bamboo and lanterns. A man with dark hair pulled back in a topknot and a white cloth draped over it is shown. He appears to be of East Asian descent, likely in his 20s or 30s, with dark eyes and a clean-shaven face. He is wearing a light beige, loose-fitting robe with long sleeves and a white undergarment. His build is average. He looks towards the camera with a serious and contemplative expression. The background features a serene outdoor setting with bamboo and lanterns, suggesting a traditional or historical context. The scene is set at night, with the lanterns providing soft, warm light that contrasts with the dark, lush greenery. In a dimly lit outdoor setting with candles and a table in the background, a man dressed in traditional attire is seated. He begins to turn his head and body to the right, gradually rising from his seated position. The man continues to stand up, fully straightening his posture while looking forward with a serious and contemplative expression, perhaps concerned or apprehensive. realistic, appearing to be a scene from a TV series or movie, characterized by a focus on natural lighting and a shallow depth of field, with a muted color palette that contributes to a somewhat somber or contemplative mood. The scene is characterized by high saturation, moderate contrast, moderate brightness and normal color temperature. The camera tilts upward, with a medium depth of field, and the lens is roughly at the same height as the subject. The camera captures the man's profile in a medium close-up shot, with the man positioned slightly to the right of the center of the frame.", "FPS-25. The video plays at normal speed. A man in traditional attire sits at a wooden table in a serene outdoor setting surrounded by bamboo and dim lighting. a middle-aged Asian man with short black hair, wearing a white robe and a light brown vest. He has a white cloth wrapped around his head. He is sitting at a low wooden table. The background is a dimly lit outdoor setting, possibly a courtyard or garden. There are bamboo trees and other greenery visible in the background. A small wooden structure with a lantern is also present. The lighting is provided by candles on the table. The scene appears to be set at night. In a dimly lit outdoor setting with a table holding a basket, a candle, and some food items, a man dressed in traditional attire with a white headband and a beige robe is seated. He begins to rise from his seated position, gradually straightening his posture. As he stands up, he turns his head to look behind him, his expression serious and alert, with a hint of concern or apprehension. realistic, appearing to be a scene from a TV series or movie, characterized by a focus on character portrayal. It features dramatic lighting, with a mix of natural and artificial light sources, creating a moody atmosphere. The color palette is muted, with earthy tones dominating the scene, suggesting a historical or period setting. The scene has high saturation, moderate contrast, moderate brightness and neutral-toned colors. The camera follows the man panning to the upper left at a normal speed, maintaining a medium close-up shot. The camera is positioned at a lower angle, with the man slightly to the right of the center of the frame. The video uses a shallow depth of field, making the distant background noticeably blurred. The shot transitions from a medium close-up of the man's back to a profile view."]
    h, m = wrapper(texts=txts)

    print(h.shape, m.shape)

    print("✅ MiniCPMWrapper ready.")
    