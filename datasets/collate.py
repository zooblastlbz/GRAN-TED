from __future__ import annotations
from typing import List, Dict, Callable
import torch


def make_contrastive_collate(
    tokenizer,
    max_len: int = 512,
    device: str = "cuda"
) -> Callable:
    """
    返回可传给 DataLoader 的 collate_fn
    输出 dict：input_ids1, attention_mask1, input_ids2, attention_mask2
    """
    dev = torch.device(device)

    def _collate(batch: List):                # batch = [(t1, t2), ...]
        text1_list, text2_list = zip(*batch)  # unzip

        enc1 = tokenizer(
            list(text1_list),
            padding="longest",
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        enc2 = tokenizer(
            list(text2_list),
            padding="longest",
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        return {
            "input_ids1":      enc1.input_ids.to(dev),
            "attention_mask1": enc1.attention_mask.to(dev),
            "input_ids2":      enc2.input_ids.to(dev),
            "attention_mask2": enc2.attention_mask.to(dev),
        }

    return _collate

# datasets/collate_jina.py


def make_jina_contrastive_collate(
    max_len: int | None = None,
) -> Callable:
    """
    生成可直接传给 DataLoader 的 `collate_fn`（Contrastive Learning）

    • batch 输入 : List[Tuple[str,str]] ，即 (text1,text2)
    • 输出 dict :
        {
            "texts1": List[str],   # 与 batch_size 等长
            "texts2": List[str],
        }
    • 不做任何 token→ID 工作；由下游 wrapper / encoder 负责
    • 可选 `max_len` ：若提供就用 Python slice 截字符串长度
      （只对极端长文本做保险；不考虑 token 数）
    """
    def _collate(batch: List[Tuple[str, str]]):
        texts1, texts2 = zip(*batch)        # 拆分

        if max_len is not None:
            # 简单对字符做切片 —— 若想用 token-based 可自行替换
            texts1 = [t[:max_len] for t in texts1]
            texts2 = [t[:max_len] for t in texts2]

        return {
            "texts1": list(texts1),         # DataLoader 会返回 dict
            "texts2": list(texts2),
        }

    return _collate
