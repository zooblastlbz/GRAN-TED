from __future__ import annotations

from typing import List, Optional, Protocol, Tuple, Union, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class Encoder(Protocol):
    """
    统一的编码器接口，包装 LLM / 多模态 / embedding 模型。

    forward 需返回:
        hidden_states: torch.Tensor
        attention_mask: torch.Tensor
    """

    def forward(
        self,
        *,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        texts: Optional[Union[str, List[str]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...


class TrainableEncoder(nn.Module):
    """
    提供一个可选的基类，便于 wrapper 继承并声明 forward 签名。
    """

    def forward(
        self,
        *,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        texts: Optional[Union[str, List[str]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
