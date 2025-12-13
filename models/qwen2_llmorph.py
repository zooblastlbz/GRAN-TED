from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Type, Union, Dict
import sys
sys.path.append("/ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/proj_text_enc/attn_pool_contrastive/models")  # noqa

import re

from embeddings import TextProjection

from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel
from bs4 import BeautifulSoup
import ftfy
import urllib.parse as ul

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.utils import deprecate
from einops import rearrange, repeat
# from flash_attn import flash_attn_func, flash_attn_varlen_func
# from flash_attn.bert_padding import index_first_axis, pad_input

from diffusers.models.attention_processor import Attention

from diffusers.models.attention import _chunked_feed_forward
from diffusers.models.attention_processor import Attention
from diffusers.utils.torch_utils import maybe_allow_in_graph

from transformers import Qwen2Model, Qwen2ForCausalLM, Qwen2PreTrainedModel, Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer,
    Qwen2RMSNorm,
    Qwen2Attention,
    # Qwen2FlashAttention2,
    # Qwen2SdpaAttention,
    Qwen2MLP,
)

from normalization import RMSNorm


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim, inner_dim, mult=4.0, dropout=0.0, bias=False):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        self.linear1 = nn.Linear(dim, inner_dim, bias=bias)
        self.linear2 = nn.Linear(dim, inner_dim, bias=bias)
        self.linear3 = nn.Linear(inner_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    @torch.compile
    def silu_multiply(self, a, b):
        return F.silu(a) * b

    def forward(self, hidden_states):
        hidden_states_1 = self.linear1(hidden_states)
        hidden_states_2 = self.linear2(hidden_states)
        hidden_states = self.silu_multiply(hidden_states_1, hidden_states_2)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.linear3(hidden_states)
        return hidden_states


class Qwen2TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "/ytech_m2v5_hdd/workspace/kling_mm/guanyushuo/codes/mllm_data_process/kling_text_encoder/draft/model/Qwen2-7B",
            trust_remote_code=True,
        )
        config_kwargs = {
            "cache_dir": None,
            "revision": None,
            "token": None,
            "trust_remote_code": True,
        }
        config = AutoConfig.from_pretrained("/ytech_m2v5_hdd/workspace/kling_mm/guanyushuo/codes/mllm_data_process/kling_text_encoder/draft/model/Qwen2-7B", **config_kwargs)

        model = Qwen2BiForMNTP.from_pretrained(
            "/ytech_m2v5_hdd/workspace/kling_mm/guanyushuo/codes/mllm_data_process/kling_text_encoder/draft/model/Qwen2-7B",
            from_tf=False,
            config=config,
            cache_dir=None,
            revision=None,
            token=None,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            # low_cpu_mem_usage=model_args.low_cpu_mem_usage,
            # device_map=device_map,
            low_cpu_mem_usage=True,
        )
        text_encoder = model
        text_encoder.model = PeftModel.from_pretrained(
            text_encoder.model,
            "/ytech_m2v5_hdd/workspace/kling_mm/guanyushuo/codes/mllm_data_process/kling_text_encoder/draft/model/Qwen2-7B-mntp-r64",
        )
        text_encoder.model = text_encoder.model.merge_and_unload()
        self.text_encoder = text_encoder
        self.adapter = Adapter()

        state_dict = torch.load("/ytech_m2v5_hdd/workspace/kling_mm/guanyushuo/codes/mllm_data_process/kling_text_encoder/draft/model/adapter_v2.ckpt", weights_only=True)
        self.adapter = load_model(self.adapter, state_dict)

    def _text_preprocessing(self, text, clean_caption=False):
        if not isinstance(text, (tuple, list)):
            text = [text]

        def process(text: str):
            if clean_caption:
                text = self._clean_caption(text)
            else:
                text = text.lower().strip()
            return text

        return [process(t) for t in text]

    def _clean_caption(self, caption):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub("<person>", "person", caption)
        # urls:
        caption = re.sub(
            r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        caption = re.sub(
            r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features="html.parser").text

        # @<nickname>
        caption = re.sub(r"@[\w\d]+\b", "", caption)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
        caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
        caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
        caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
        caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
        caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
        caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
        #######################################################

        # все виды тире / all types of dash --> "-"
        caption = re.sub(
            r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",  # noqa
            "-",
            caption,
        )

        # кавычки к одному стандарту
        caption = re.sub(r"[`´«»“”¨]", '"', caption)
        caption = re.sub(r"[‘’]", "'", caption)

        # &quot;
        caption = re.sub(r"&quot;?", "", caption)
        # &amp
        caption = re.sub(r"&amp", "", caption)

        # ip adresses:
        caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

        # article ids:
        caption = re.sub(r"\d:\d\d\s+$", "", caption)

        # \n
        caption = re.sub(r"\\n", " ", caption)

        # "#123"
        caption = re.sub(r"#\d{1,3}\b", "", caption)
        # "#12345.."
        caption = re.sub(r"#\d{5,}\b", "", caption)
        # "123456.."
        caption = re.sub(r"\b\d{6,}\b", "", caption)
        # filenames:
        caption = re.sub(r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption)

        #
        caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""

        caption = re.sub(self.bad_punct_regex, r" ", caption)  # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r"(?:\-|\_)")
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, " ", caption)

        caption = ftfy.fix_text(caption)
        caption = html.unescape(html.unescape(caption))

        caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
        caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
        caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

        caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
        caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
        caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
        caption = re.sub(r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", caption)
        caption = re.sub(r"\bpage\s+\d+\b", "", caption)

        caption = re.sub(r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", caption)  # j2d1a2a...

        caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)

        caption = re.sub(r"\b\s+\:\s+", r": ", caption)
        caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
        caption = re.sub(r"\s+", " ", caption)

        caption.strip()

        caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
        caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
        caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
        caption = re.sub(r"^\.\S+$", "", caption)

        return caption.strip()

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        do_classifier_free_guidance: bool = True,
        negative_prompt: str = "",
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        clean_caption: bool = False,
        max_sequence_length: int = 512,
        **kwargs,
    ):
        if "mask_feature" in kwargs:
            deprecation_message = "The use of `mask_feature` is deprecated. It is no longer used in any computation and that doesn't affect the end results. It will be removed in a future version."
            deprecate("mask_feature", "1.0.0", deprecation_message, standard_warn=False)

        if device is None:
            #device = self._execution_device
            device = self.device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # See Section 3.1. of the paper.
        max_length = max_sequence_length

        if prompt_embeds is None:
            prompt = self._text_preprocessing(prompt, clean_caption=clean_caption)
            ## add for qwen text encoder
            prompt = ["_" if p == "" else p for p in prompt]
            text_inputs = self.tokenizer(prompt, return_tensors="pt", padding="longest", max_length=max_length, truncation=True)
            text_input_ids = text_inputs

            prompt_attention_mask = text_inputs["attention_mask"]
            prompt_attention_mask = prompt_attention_mask.to(device)
            text_input_ids['input_ids'] = text_input_ids['input_ids'].long()
            
            #print(f"-------input_ids={text_input_ids['input_ids'].shape}")
            if text_input_ids['input_ids'].numel() == 0:
                print(f"------{prompt}")            
            
            prompt_embeds = self.text_encoder(**text_input_ids.to(device), output_hidden_states=True)["hidden_states"]
            prompt_embeds = torch.concatenate((prompt_embeds[-2], prompt_embeds[-5], prompt_embeds[-8], prompt_embeds[-11]), dim=1)

        if self.text_encoder is not None:
            dtype = self.text_encoder.dtype
        elif self.transformer is not None:
            dtype = self.transformer.dtype
        else:
            dtype = None

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        prompt_attention_mask = prompt_attention_mask.view(bs_embed, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens = [negative_prompt] * batch_size
            # uncond_tokens = ['.'] * batch_size
            uncond_tokens = self._text_preprocessing(uncond_tokens, clean_caption=clean_caption)
            max_length = prompt_embeds.shape[1] // 4

            uncond_input = self.tokenizer(uncond_tokens, return_tensors="pt", padding="max_length", max_length=max_length, truncation=True)

            negative_prompt_attention_mask = uncond_input["attention_mask"]
            negative_prompt_attention_mask = negative_prompt_attention_mask.to(device)

            negative_prompt_embeds = self.text_encoder(**uncond_input.to(device), output_hidden_states=True)["hidden_states"]
            negative_prompt_embeds = torch.concatenate(
                (negative_prompt_embeds[-2], negative_prompt_embeds[-5], negative_prompt_embeds[-8], negative_prompt_embeds[-11]), dim=1
            )

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            negative_prompt_attention_mask = negative_prompt_attention_mask.view(bs_embed, -1)
            negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(num_images_per_prompt, 1)
        else:
            negative_prompt_embeds = None
            negative_prompt_attention_mask = None

        prompt_attention_mask = prompt_attention_mask.to(self.text_encoder.dtype)
        prompt_embeds = self.adapter(prompt_embeds, prompt_attention_mask)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = self.adapter(negative_prompt_embeds, negative_prompt_attention_mask)

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask


def load_model(model, state_dict):
    # 遍历模型的每个参数和名称
    for name, param in model.named_parameters():
        if name in state_dict:
            # 直接更新参数值
            try:
                param.data.copy_(state_dict[name])
            except RuntimeError as e:
                print(f"Error loading {name}: {e}")
            state_dict.pop(name)
        else:
            print(f"Missing in state_dict: {name}")

    # 检查模型中不需要的参数
    if len(state_dict) > 0:
        for name in state_dict:
            print(f"Unexpected in state_dict: {name}")
    return model


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_caption_projection = nn.ModuleList([TextProjection(3584, 3584) for _ in range(4)])
        self.caption_aggregation = nn.ModuleList(
            [
                CaptionAggregation(
                    3584,
                    28,
                    128,
                    dropout=0.0,
                    cross_attention_dim=3584,
                    num_embeds_ada_norm=None,
                    attention_bias=True,
                    only_cross_attention=False,
                    double_self_attention=False,
                    upcast_attention=False,
                    norm_elementwise_affine=False,
                    norm_eps=1e-6,
                    attention_type="default",
                    use_temp_attn=False,
                    image_temp_attn=False,
                    ffn_scale=1.0,
                )
                for _ in range(2)
            ]
        )

        for name, module in self.caption_aggregation.named_modules():
            if isinstance(module, Attention) and (name.endswith("attn1") or name.endswith("attn2")):
                processor = MaskedAttnProcessor2_0(
                    use_flash_attn=False,
                    rope=None,
                    qk_norm=True,
                    embed_dim=128,
                )
                module.set_processor(processor)

    def forward(self, encoder_hidden_states, encoder_attention_mask):
        aux_encoder_attention_mask, aux_cross_attn_mask_kwargs = prepare_mask(encoder_attention_mask.repeat(1, 3), 1, None, mask_type="cross")
        encoder_attention_mask, cross_attn_mask_kwargs = prepare_mask(encoder_attention_mask, 1, None, mask_type="cross")

        encoder_hidden_states = encoder_hidden_states.chunk(4, dim=1)
        encoder_hidden_states = [self.pre_caption_projection[i](encoder_hidden_states[i]) for i in range(len(encoder_hidden_states))]
        aux_encoder_hidden_states = torch.concatenate(encoder_hidden_states[1:], dim=1)
        encoder_hidden_states = encoder_hidden_states[0]
        for block in self.caption_aggregation:
            encoder_hidden_states = block(
                encoder_hidden_states,
                spatial_attention_mask=encoder_attention_mask,
                temporal_attention_mask=None,
                spatial_temporal_attention_mask=None,
                encoder_hidden_states=aux_encoder_hidden_states,
                encoder_attention_mask=aux_encoder_attention_mask,
                timestep=None,
                patch_resolution=None,
                spatial_attn_mask_kwargs=cross_attn_mask_kwargs,
                temporal_attn_mask_kwargs=None,
                spatial_temporal_attn_mask_kwargs=None,
                cross_attn_mask_kwargs=aux_cross_attn_mask_kwargs,
                cross_attention_kwargs=None,
                class_labels=None,
                num_frames=None,
            )
        return encoder_hidden_states


class TokenMerge(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(1, 2, 2)):
        super().__init__()
        self.t = patch_size[0]
        self.h = patch_size[1]
        self.w = patch_size[2]
        self.proj = nn.Linear(in_features * self.t * self.h * self.w, out_features, bias=False)

    def forward(self, x):
        x = rearrange(x, "... (t nt) (h nh) (w nw) e -> ... t h w (nt nh nw e)", nt=self.t, nh=self.h, nw=self.w)
        return self.proj(x)


class TokenSplitWithoutSkip(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(1, 2, 2)):
        super().__init__()
        self.t = patch_size[0]
        self.h = patch_size[1]
        self.w = patch_size[2]
        self.proj = nn.Linear(in_features, out_features * self.t * self.h * self.w, bias=False)
        # self.proj.weight.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, "... t h w (nt nh nw e) -> ... (t nt) (h nh) (w nw) e", nt=self.t, nh=self.h, nw=self.w)


@torch.compile
def multiply_addition(a, b):
    return a * (b + 1)


@maybe_allow_in_graph
class CaptionAggregation(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ada_norm_continous_conditioning_embedding_dim: Optional[int] = None,
        ada_norm_bias: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        use_temp_attn: bool = False,
        image_temp_attn: bool = False,
        ffn_scale: float = 1.0,
    ):
        super().__init__()

        # Define 4 blocks. Each block has its own normalization layer.

        # 1. Self-Attn
        self.norm1 = RMSNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        # 3. Cross-Attn
        self.norm2 = RMSNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim if not double_self_attention else None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        # 4. Feed-forward
        self.norm3 = RMSNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.ff = SwiGLUFeedForward(
            dim,
            dropout=dropout,
            inner_dim=ff_inner_dim,
            mult=4.0 * ffn_scale,
            bias=ff_bias,
        )

        # 5. Scale
        self.scale_table = nn.Embedding.from_pretrained(torch.randn(3, dim) / dim**0.5)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

        self.register_buffer("tensor_0", torch.tensor([[0]]))
        self.register_buffer("tensor_1", torch.tensor([[1]]))
        self.register_buffer("tensor_2", torch.tensor([[2]]))

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        spatial_attention_mask: Optional[torch.FloatTensor] = None,
        temporal_attention_mask: Optional[torch.FloatTensor] = None,
        spatial_temporal_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        patch_resolution: Optional[Tuple[int, int, int]] = None,
        spatial_attn_mask_kwargs: Dict[str, Any] = None,
        temporal_attn_mask_kwargs: Dict[str, Any] = None,
        spatial_temporal_attn_mask_kwargs: Dict[str, Any] = None,
        cross_attn_mask_kwargs: Dict[str, Any] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        num_frames: int = 1,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.FloatTensor:
        # copied from diffusers/models/attention.py BasicTransformerBlock.forward
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}

        # Notice that normalization is always applied before the real computation in the following blocks.
        # 1. Self-Attention

        scale_msa = self.scale_table(self.tensor_0)
        scale_mca = self.scale_table(self.tensor_1)
        scale_mlp = self.scale_table(self.tensor_2)

        norm_hidden_states = multiply_addition(self.norm1(hidden_states), scale_msa)
        attn_output = self.attn1(
            norm_hidden_states,
            attention_mask=spatial_attention_mask,
            patch_resolution=None,
            selfattn_mask_kwargs=spatial_attn_mask_kwargs,
        )
        hidden_states = hidden_states + attn_output

        # 3. Cross-Attention
        norm_hidden_states = multiply_addition(self.norm2(hidden_states), scale_mca)
        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            patch_resolution=None,
            selfattn_mask_kwargs=spatial_attn_mask_kwargs,
            crossattn_mask_kwargs=cross_attn_mask_kwargs,
        )
        hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = multiply_addition(self.norm3(hidden_states), scale_mlp)

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states

        return hidden_states


@torch.no_grad()
def prepare_mask(mask, num_frames, patch_size, token_merge_size=None, mask_type=None):
    if mask is None:
        return None, None

    if mask_type != "cross":
        mask = F.avg_pool3d(mask, kernel_size=patch_size, stride=patch_size)
        assert torch.all((mask == 0) | (mask == 1)), "mask is not binary"

    if token_merge_size is not None:
        mask = F.avg_pool3d(mask, kernel_size=token_merge_size, stride=token_merge_size)
        assert torch.all((mask == 0) | (mask == 1)), "mask is not binary"

    if mask_type == "cross":
        mask = repeat(mask, "b l -> (b f) 1 l", f=num_frames).contiguous()  # num_frames
    elif mask_type == "spatial":
        mask = rearrange(mask, "b c f h w -> (b f) c (h w)")
    elif mask_type == "temporal":
        mask = rearrange(mask, "b c f h w -> (b h w) c f")
    elif mask_type == "3d":
        mask = rearrange(mask, "b c f h w -> b c (f h w)")
    else:
        raise ValueError(f"mask_type={mask_type}, not in (cross, spatial, temporal, 3d)")

    assert mask.ndim == 3 and mask.shape[1] == 1, f"mask.shape = {mask.shape}"
    unpadding_args = unpadding_mask_args(mask[:, 0])
    mask = (1 - mask) * -10000.0
    return mask, unpadding_args


def unpadding_mask_args(attention_mask):
    assert attention_mask.ndim == 2
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    unpadding_args_dict = {}
    unpadding_args_dict["seqlens_in_batch"] = seqlens_in_batch
    unpadding_args_dict["indices"] = indices
    unpadding_args_dict["cu_seqlens"] = cu_seqlens
    unpadding_args_dict["max_seqlen_in_batch"] = max_seqlen_in_batch
    return unpadding_args_dict


class MaskedAttnProcessor2_0(nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
        self,
        use_flash_attn=False,
        rope=None,
        qk_norm: bool = False,
        embed_dim=72,
        eps: float = 1e-6,
        token_merge_size=None,
        hidden_dim: int = 1152,
        latent_dim: int = 1152,
    ):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        super().__init__()

        if token_merge_size is not None:
            self.token_merge = TokenMerge(in_features=hidden_dim, out_features=latent_dim, patch_size=token_merge_size)
            self.token_split = TokenSplitWithoutSkip(in_features=latent_dim, out_features=hidden_dim, patch_size=token_merge_size)
        else:
            self.token_merge = None
            self.token_split = None

        self.rope = rope

        if qk_norm:
            self.q_norm = RMSNorm(embed_dim, eps=eps)
            self.k_norm = RMSNorm(embed_dim, eps=eps)
        else:
            self.q_norm = None
            self.k_norm = None

        self.use_flash_attn = use_flash_attn
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        if torch.cuda.is_available() and torch.version.hip:
            self.flash_attn_max_head_dim = 128
        elif torch.cuda.is_available() and torch.version.cuda:
            self.flash_attn_max_head_dim = 256
        else:
            self.flash_attn_max_head_dim = None

    def _attn_varlen(self, query, key, value, crossattn_mask_kwargs=None, selfattn_mask_kwargs=None):
        assert crossattn_mask_kwargs != None or selfattn_mask_kwargs != None, "crossattn_mask_kwargs 和 selfattn_mask_kwargs不可同时为None"

        batch, seqlen = query.shape[:2]

        # for q
        if selfattn_mask_kwargs is None:
            max_seqlen_in_batch_q = query.shape[1]
            cu_seqlens_q = torch.arange(0, query.shape[0] * query.shape[1] + 1, query.shape[1], dtype=torch.int32, device="cuda")
            indices_q = torch.arange(0, query.shape[0] * query.shape[1], device="cuda")
            query = rearrange(query, "b s ... -> (b s) ...")
        else:
            max_seqlen_in_batch_q = selfattn_mask_kwargs["max_seqlen_in_batch"]
            cu_seqlens_q = selfattn_mask_kwargs["cu_seqlens"]
            indices_q = selfattn_mask_kwargs["indices"]
            query = index_first_axis(rearrange(query, "b s ... -> (b s) ..."), indices_q)

        # for k & v
        if crossattn_mask_kwargs != None:
            cu_seqlens_kv = crossattn_mask_kwargs["cu_seqlens"]
            max_seqlen_in_batch_kv = crossattn_mask_kwargs["max_seqlen_in_batch"]
            indices_kv = crossattn_mask_kwargs["indices"]
        else:
            cu_seqlens_kv = selfattn_mask_kwargs["cu_seqlens"]
            max_seqlen_in_batch_kv = selfattn_mask_kwargs["max_seqlen_in_batch"]
            indices_kv = selfattn_mask_kwargs["indices"]

        # TODO: index_first_axis is not efficient.
        key = index_first_axis(rearrange(key, "b s ... -> (b s) ..."), indices_kv)
        value = index_first_axis(rearrange(value, "b s ... -> (b s) ..."), indices_kv)
        attn_output_unpad = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_kv,
            dropout_p=0.0,
            softmax_scale=None,
            causal=False,
        )

        hidden_states = pad_input(attn_output_unpad, indices_q, batch, seqlen)
        return hidden_states

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        patch_resolution: Optional[Tuple[int, int, int]] = None,
        crossattn_mask_kwargs: Optional[dict] = None,
        selfattn_mask_kwargs: Optional[dict] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        if self.token_merge is not None:
            hidden_states = rearrange(hidden_states, "b (t h w) d -> b t h w d", t=patch_resolution[0], h=patch_resolution[1], w=patch_resolution[2])
            hidden_states = self.token_merge(hidden_states)
            merge_b, merge_t, merge_h, merge_w, merge_d = hidden_states.shape
            patch_resolution = (merge_t, merge_h, merge_w)
            hidden_states = rearrange(hidden_states, "b t h w d -> b (t h w) d")

        batch_size, sequence_length, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if self.rope is not None:
            query = self.rope(query, patch_resolution)
            key = self.rope(key, patch_resolution)

        if self.q_norm is not None:
            query = self.q_norm(query)
        if self.k_norm is not None:
            key = self.k_norm(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        if self.use_flash_attn and query.dtype is not torch.float32 and query.shape[-1] <= self.flash_attn_max_head_dim:
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            if selfattn_mask_kwargs is None and crossattn_mask_kwargs is None:
                hidden_states = flash_attn_func(query, key, value, dropout_p=0.0, softmax_scale=None, causal=False)
            else:
                hidden_states = self._attn_varlen(query, key, value, crossattn_mask_kwargs=crossattn_mask_kwargs, selfattn_mask_kwargs=selfattn_mask_kwargs)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            if attention_mask is not None:
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

            hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if self.token_split is not None:
            hidden_states = rearrange(hidden_states, "b (t h w) d -> b t h w d", t=merge_t, h=merge_h, w=merge_w)
            hidden_states = self.token_split(hidden_states)
            hidden_states = rearrange(hidden_states, "b t h w d -> b (t h w) d")

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class ModifiedQwen2Attention(Qwen2Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedQwen2FlashAttention2(Qwen2Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedQwen2SdpaAttention(Qwen2Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


QWEN2_ATTENTION_CLASSES = {
    "eager": ModifiedQwen2Attention,
    "flash_attention_2": ModifiedQwen2FlashAttention2,
    "sdpa": ModifiedQwen2SdpaAttention,
}


class ModifiedQwen2DecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size

        self.self_attn = QWEN2_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen2BiModel(Qwen2Model):
    _no_split_modules = ["ModifiedQwen2DecoderLayer"]

    def __init__(self, config: Qwen2Config):
        # Qwen2PreTrainedModel.__init__(self, config)
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([ModifiedQwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()


class Qwen2BiForMNTP(Qwen2ForCausalLM):
    def __init__(self, config):
        Qwen2PreTrainedModel.__init__(self, config)
        self.model = Qwen2BiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    # getter for PEFT model
    def get_model_for_peft(self):
        return self.model

    # setter for PEFT model
    def set_model_for_peft(self, model: PeftModel):
        self.model = model

    # save the PEFT model
    def save_peft_model(self, path):
        self.model.save_pretrained(path)


if __name__ == "__main__":
    text_encoder = Qwen2TextEncoder().to(torch.bfloat16).to("cuda:0")
    prompt_embeds, prompt_attention_mask, _, _ = text_encoder.encode_prompt(
        [
            "A cozy living room scene features an anthropomorphic cat, dressed in long sleeves and an apron, sitting comfortably on a plush sofa, intently focused on knitting a sweater with his hands. The cat's paws move deftly as he works on the intricate pattern, and the numerous rows of completed knitting on the floor suggest that he has been busy for some time. Next to him, an anthropomorphic kitten, also dressed in clothes, sits on the sofa, gazing longingly at the partially completed sweater. The kitten's small hands rest on the fabric, as if trying to get a feel for the soft yarn. The room is dimly lit, with a warm glow emanating from a nearby lamp, casting a comforting ambiance over the scene. The walls are adorned with family photos and knick-knacks, giving the impression of a warm and welcoming home. The air is thick with the scent of freshly baked cookies, wafting from the nearby kitchen, adding to the sense of comfort and relaxation.",
            "A cozy living room scene features an anthropomorphic cat."
        ],
        do_classifier_free_guidance=False,
        device=text_encoder.text_encoder.device,
        clean_caption=False,
        max_sequence_length=2048,
    )
    print(prompt_embeds.shape)
    print(prompt_attention_mask.shape)
