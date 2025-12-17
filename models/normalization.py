import numbers
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from embeddings import CombinedTimestepConditionEmbeddings



@torch.compile
def rmsnorm(hidden_states, weight, eps, dtype):
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    hidden_states = hidden_states.to(dtype)
    hidden_states = hidden_states * weight
    return hidden_states


@torch.compile
def rmsnorm_without_weight(hidden_states, eps, dtype):
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    hidden_states = hidden_states.to(dtype)
    return hidden_states


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, elementwise_affine: bool = True):
        super().__init__()

        if isinstance(dim, numbers.Integral):
            dim = (dim,)
        self.dim = torch.Size(dim)

        self.eps = eps

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

    def forward(self, hidden_states):
        if self.weight is not None:
            hidden_states = rmsnorm(hidden_states, self.weight, self.eps, self.weight.dtype)
        else:
            hidden_states = rmsnorm_without_weight(hidden_states, self.eps, hidden_states.dtype)

        return hidden_states
