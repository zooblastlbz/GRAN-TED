import math

import torch
import torch.nn as nn
from diffusers.models.activations import FP32SiLU
from diffusers.models.embeddings import TimestepEmbedding, Timesteps


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class BaseRoPE:
    @torch.compile
    def apply_rope(self, x, freqs_cos, freqs_sin, shift_freqs_cos=None, shift_freqs_sin=None, num_wins=None):
        batch, num_heads, num_patches, embed_dim = x.shape
        inputs = x
        x = x.reshape(batch, num_heads, num_patches, -1, 2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        x = x.reshape(batch, num_heads, num_patches, embed_dim)

        freqs_cos = freqs_cos.reshape(-1, embed_dim)
        freqs_sin = freqs_sin.reshape(-1, embed_dim)
        
        if shift_freqs_cos is not None and shift_freqs_sin is not None and num_wins is not None:
            # print('shift_rope')
            # 处理cos部分
            shift_freqs_cos = shift_freqs_cos.reshape(-1, embed_dim)
            normal_freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(0)  # [1, 1, num_patches, embed_dim]
            shift_freqs_cos = shift_freqs_cos.unsqueeze(0).unsqueeze(0)  # [1, 1, num_patches, embed_dim]
            
            # 在第二个维度重复num_heads次
            normal_freqs_cos = normal_freqs_cos.expand(-1, num_heads, -1, -1)  # [1, num_heads, num_patches, embed_dim]
            shift_freqs_cos = shift_freqs_cos.expand(-1, num_heads, -1, -1)  # [1, num_heads, num_patches, embed_dim]
            
            # 创建完整的freqs_cos序列
            freqs_cos_list = [shift_freqs_cos] + [normal_freqs_cos] * (num_wins - 1)
            freqs_cos = torch.cat(freqs_cos_list, dim=0)  # [windows, num_heads, num_patches, embed_dim]
            
            # 处理sin部分
            shift_freqs_sin = shift_freqs_sin.reshape(-1, embed_dim)
            normal_freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(0)  # [1, 1, num_patches, embed_dim]
            shift_freqs_sin = shift_freqs_sin.unsqueeze(0).unsqueeze(0)  # [1, 1, num_patches, embed_dim]
            
            # 在第二个维度重复num_heads次
            normal_freqs_sin = normal_freqs_sin.expand(-1, num_heads, -1, -1)  # [1, num_heads, num_patches, embed_dim]
            shift_freqs_sin = shift_freqs_sin.expand(-1, num_heads, -1, -1)  # [1, num_heads, num_patches, embed_dim]
            
            # 创建完整的freqs_sin序列
            freqs_sin_list = [shift_freqs_sin] + [normal_freqs_sin] * (num_wins - 1)
            freqs_sin = torch.cat(freqs_sin_list, dim=0)  # [windows, num_heads, num_patches, embed_dim]
            
            # 将freqs_cos和freqs_sin在第一个维度重复batch // windows次
            freqs_cos = freqs_cos.repeat(batch // num_wins, 1, 1, 1)  # [batch, num_heads, num_patches, embed_dim]
            freqs_sin = freqs_sin.repeat(batch // num_wins, 1, 1, 1)  # [batch, num_heads, num_patches, embed_dim]

        return inputs * freqs_cos + x * freqs_sin

    def _forward(self, x, patch_resolution, num_wins=None):
        if self.freqs_cos.device != x.device or self.freqs_cos.dtype != x.dtype:
            self.freqs_cos = self.freqs_cos.to(x.device, x.dtype)
        if self.freqs_sin.device != x.device or self.freqs_sin.dtype != x.dtype:
            self.freqs_sin = self.freqs_sin.to(x.device, x.dtype)
        freqs_cos, freqs_sin = self.freqs_cos, self.freqs_sin

        for dim_idx, length in enumerate(patch_resolution):
            freqs_cos = torch.narrow(freqs_cos, dim_idx, 0, length)
            freqs_sin = torch.narrow(freqs_sin, dim_idx, 0, length)

        if num_wins is not None:
            length = freqs_sin.shape[0]
            shift_freqs_cos = freqs_cos[: length // 2]
            shift_freqs_sin = freqs_sin[: length // 2]
            shift_freqs_cos = torch.cat([shift_freqs_cos] * 2)
            shift_freqs_sin = torch.cat([shift_freqs_sin] * 2)
        else:
            shift_freqs_cos = None
            shift_freqs_sin = None

        return self.apply_rope(x, freqs_cos, freqs_sin, shift_freqs_cos, shift_freqs_sin, num_wins)


class RoPE2D(BaseRoPE):
    def __init__(self, embed_dim, max_patch_resolution=(160, 160), interpolation_scale=1, theta=10000.0, repeat_interleave=True):
        super().__init__()

        if embed_dim % 4 != 0:
            raise ValueError("embed_dim must be divisible by 4.")
        self.axis_embed_dim = embed_dim // 2
        self.theta = theta

        self.max_patch_resolution = max_patch_resolution
        self.interpolation_scale = interpolation_scale
        self.repeat_interleave = repeat_interleave

        freqs_cos, freqs_sin = self.compute_position_embedding()
        # self.register_buffer("freqs_cos", freqs_cos)
        # self.register_buffer("freqs_sin", freqs_sin)
        self.freqs_cos = freqs_cos
        self.freqs_sin = freqs_sin

    def compute_position_embedding(self):
        frequency = 1.0 / (self.theta ** (torch.arange(0, self.axis_embed_dim, 2).float() / self.axis_embed_dim))

        h, w = self.max_patch_resolution
        position_h = torch.arange(h)[:, None].float() / self.interpolation_scale @ frequency[None, :]
        position_w = torch.arange(w)[:, None].float() / self.interpolation_scale @ frequency[None, :]

        if self.repeat_interleave:
            position_h = position_h.repeat_interleave(2, dim=1)
            position_w = position_w.repeat_interleave(2, dim=1)
        else:
            position_h = position_h.repeat(1, 2)
            position_w = position_w.repeat(1, 2)

        height = position_h[:, None, :].expand(h, w, self.axis_embed_dim)
        width = position_w[None, :, :].expand(h, w, self.axis_embed_dim)
        position = torch.cat((height, width), dim=-1)

        freqs_cos = position.cos()
        freqs_sin = position.sin()
        return freqs_cos, freqs_sin

    def forward(self, x, patch_resolution):
        if isinstance(patch_resolution, list) and all(isinstance(resolution, tuple) for resolution in patch_resolution):
            num_frames = patch_resolution[0][0]
            assert all(resolution[0] == num_frames for resolution in patch_resolution)
            output = torch.zeros_like(x)
            for i, resolution in enumerate(patch_resolution):
                valid_sequence_length = math.prod(resolution[1:])
                sub_x = x[i * num_frames : i * num_frames + num_frames, :, :valid_sequence_length]
                sub_output = self._forward(sub_x, resolution[1:])
                output[i * num_frames : i * num_frames + num_frames, :, :valid_sequence_length] = sub_output
            return output
        elif isinstance(patch_resolution, tuple):
            return self._forward(x, patch_resolution[1:])
        else:
            raise TypeError("patch_resolution must be a list of tuple or a tuple.")


class RoPE3D(BaseRoPE):
    def __init__(self, embed_dim, max_patch_resolution=(40, 160, 160), interpolation_scale=1, theta_time=10000.0, theta_space=10000.0, embed_dim_time=None, embed_dim_space=None, repeat_interleave=True):
        super().__init__()

        self.embed_dim = embed_dim
        if embed_dim_time is not None and embed_dim_space is not None:
            if (embed_dim_time + embed_dim_space * 2) != embed_dim:
                raise ValueError("embed_dim_time + embed_dim_space * 2 must be equal to embed_dim.")
            self.embed_dim_time = embed_dim_time
            self.embed_dim_space = embed_dim_space
        else:
            self.embed_dim_time = embed_dim // 3
            self.embed_dim_space = embed_dim // 3

        self.theta_time = theta_time
        self.theta_space = theta_space

        self.max_patch_resolution = max_patch_resolution
        self.interpolation_scale = interpolation_scale
        self.repeat_interleave = repeat_interleave

        freqs_cos, freqs_sin = self.compute_position_embedding()
        # self.register_buffer("freqs_cos", freqs_cos)
        # self.register_buffer("freqs_sin", freqs_sin)
        self.freqs_cos = freqs_cos
        self.freqs_sin = freqs_sin

    def compute_position_embedding(self):
        frequency_time = 1.0 / (self.theta_time ** (torch.arange(0, self.embed_dim_time, 2).float() / self.embed_dim_time))
        frequency_space = 1.0 / (self.theta_space ** (torch.arange(0, self.embed_dim_space, 2).float() / self.embed_dim_space))

        t, h, w = self.max_patch_resolution
        position_t = torch.arange(t)[:, None].float() @ frequency_time[None, :]
        position_h = torch.arange(h)[:, None].float() / self.interpolation_scale @ frequency_space[None, :]
        position_w = torch.arange(w)[:, None].float() / self.interpolation_scale @ frequency_space[None, :]

        if self.repeat_interleave:
            position_t = position_t.repeat_interleave(2, dim=1)
            position_h = position_h.repeat_interleave(2, dim=1)
            position_w = position_w.repeat_interleave(2, dim=1)
        else:
            position_t = position_t.repeat(1, 2)
            position_h = position_h.repeat(1, 2)
            position_w = position_w.repeat(1, 2)

        frame = position_t[:, None, None, :].expand(t, h, w, self.embed_dim_time)
        height = position_h[None, :, None, :].expand(t, h, w, self.embed_dim_space)
        width = position_w[None, None, :, :].expand(t, h, w, self.embed_dim_space)
        position = torch.cat((frame, height, width), dim=-1)

        freqs_cos = position.cos()
        freqs_sin = position.sin()

        if self.embed_dim > self.embed_dim_time + self.embed_dim_space * 2:
            res_embed_dim = self.embed_dim - (self.embed_dim_time + self.embed_dim_space * 2)
            cos_shape = freqs_cos.shape[:-1] + (res_embed_dim,)
            sin_shape = freqs_sin.shape[:-1] + (res_embed_dim,)
            freqs_cos = torch.cat((freqs_cos, torch.ones(cos_shape)), dim=-1)
            freqs_sin = torch.cat((freqs_sin, torch.zeros(sin_shape)), dim=-1)

        return freqs_cos, freqs_sin

    def forward(self, x, patch_resolution, num_wins=None):
        if isinstance(patch_resolution, list) and all(isinstance(resolution, tuple) for resolution in patch_resolution):
            output = torch.zeros_like(x)
            for i, resolution in enumerate(patch_resolution):
                valid_sequence_length = math.prod(resolution)
                sub_x = x[i : i + 1, :, :valid_sequence_length]
                sub_output = self._forward(sub_x, resolution, num_wins)
                output[i : i + 1, :, :valid_sequence_length] = sub_output
            return output
        elif isinstance(patch_resolution, tuple):
            return self._forward(x, patch_resolution, num_wins)
        else:
            raise TypeError("patch_resolution must be a list of tuple or a tuple.")


class CombinedTimestepConditionEmbeddings(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        use_text_condition: bool = False,
        use_frames_resolution_condition: bool = False,
        use_unpadded_resolution_condition: bool = False,
        use_resolution_condition: bool = False,
        use_frames_condition: bool = False,
        use_noise_aug_condition: bool = False,
        split_conditions: bool = False,
        sample_proj_bias: bool = True,
    ):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim, sample_proj_bias=sample_proj_bias)

        self.use_additional_conditions = (
            use_frames_resolution_condition or use_unpadded_resolution_condition or use_resolution_condition or use_frames_condition
        )
        self.use_text_condition = use_text_condition
        self.use_frames_resolution_condition = use_frames_resolution_condition
        self.use_unpadded_resolution_condition = use_unpadded_resolution_condition
        self.use_resolution_condition = use_resolution_condition
        self.use_frames_condition = use_frames_condition
        self.use_noise_aug_condition = use_noise_aug_condition
        self.split_conditions = split_conditions

        if self.use_additional_conditions:
            self.additional_condition_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        if use_text_condition:
            self.text_embedder = TimestepEmbedding(in_channels=2048, time_embed_dim=embedding_dim, sample_proj_bias=sample_proj_bias)
        if use_frames_resolution_condition:
            self.frames_resolution_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim // 3, sample_proj_bias=sample_proj_bias)
        if use_unpadded_resolution_condition:
            self.unpadded_resolution_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim // 2, sample_proj_bias=sample_proj_bias)
        if use_resolution_condition:
            self.resolution_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim // 2, sample_proj_bias=sample_proj_bias)
        if use_frames_condition:
            self.frames_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim, sample_proj_bias=sample_proj_bias)
        ## add for vsr
        if use_noise_aug_condition:
            self.additional_noise_aug_proj = zero_module(nn.Linear(in_features=embedding_dim, 
                                                    out_features=embedding_dim, bias=False))

        # zero init
        for name, module in self.named_children():
            if isinstance(module, TimestepEmbedding) and not name.endswith("timestep_embedder"):
                module.linear_2.weight.data.zero_()
                if hasattr(module.linear_2, "bias") and module.linear_2.bias is not None:
                    module.linear_2.bias.data.zero_()

    def forward(
        self, timestep, batch_size, hidden_dtype, frames=None, resolution=None, unpadded_resolution=None, 
        frames_resolution=None, prompt_embeds_pooled=None, noise_aug_val=None
    ):
        timesteps_proj = self.time_proj(timestep.view(-1))
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)

        conditioning = timesteps_emb
        spatial_conditioning = 0
        temporal_conditioning = 0

        if self.use_text_condition and prompt_embeds_pooled is not None:
            text_emb = self.text_embedder(prompt_embeds_pooled)
            conditioning = conditioning + text_emb
        if self.use_frames_resolution_condition and frames_resolution is not None:
            frames_resolution_emb = self.additional_condition_proj(frames_resolution.flatten()).to(hidden_dtype)
            frames_resolution_emb = self.frames_resolution_embedder(frames_resolution_emb).reshape(batch_size, -1)
            spatial_conditioning = spatial_conditioning + frames_resolution_emb
        if self.use_unpadded_resolution_condition and unpadded_resolution is not None:
            unpadded_resolution_emb = self.additional_condition_proj(unpadded_resolution.flatten()).to(hidden_dtype)
            unpadded_resolution_emb = self.unpadded_resolution_embedder(unpadded_resolution_emb).reshape(batch_size, -1)
            spatial_conditioning = spatial_conditioning + unpadded_resolution_emb
        if self.use_resolution_condition and resolution is not None:
            resolution_emb = self.additional_condition_proj(resolution.flatten()).to(hidden_dtype)
            resolution_emb = self.resolution_embedder(resolution_emb).reshape(batch_size, -1)
            spatial_conditioning = spatial_conditioning + resolution_emb
        if self.use_frames_condition and frames is not None:
            frames_emb = self.additional_condition_proj(frames.flatten()).to(hidden_dtype)
            frames_emb = self.frames_embedder(frames_emb).reshape(batch_size, -1)
            temporal_conditioning = temporal_conditioning + frames_emb

        ## add for vsr
        if self.use_noise_aug_condition and noise_aug_val is not None:
            noise_timesteps_proj = self.time_proj(noise_aug_val.view(-1))
            noise_timesteps_emb = self.timestep_embedder(noise_timesteps_proj.to(dtype=hidden_dtype)) # (N, D)
            noise_timesteps_emb = self.additional_noise_aug_proj(noise_timesteps_emb)
            '''
            print(f"-----noise_timesteps_emb: {noise_aug_val}")
            torch.save(noise_timesteps_emb, "/home/xiamenghan/DiffusionGen/m2v-diffusers-vsr/output/cache_dir/aug_emb.pt")
            '''
            if conditioning.shape[0] != noise_timesteps_emb.shape[0]:
                repeat_times = int(conditioning.shape[0] // noise_timesteps_emb.shape[0])
                noise_timesteps_emb = noise_timesteps_emb.repeat(repeat_times,1)
            conditioning = conditioning + noise_timesteps_emb

        if self.split_conditions:
            return conditioning, spatial_conditioning, temporal_conditioning
        else:
            return conditioning + spatial_conditioning + temporal_conditioning


class TextProjection(nn.Module):
    def __init__(self, in_features, hidden_size, out_features=None, act_fn="gelu_tanh", bias=True):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_size, bias=bias)
        if act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = nn.SiLU()
        elif act_fn == "silu_fp32":
            self.act_1 = FP32SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=out_features, bias=bias)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states
