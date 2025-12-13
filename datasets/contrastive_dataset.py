import json
from pathlib import Path
from typing import Tuple, List, Union
from torch.utils.data import Dataset

import re
import random

# 定义六段式前缀匹配字典
required_prefixes = {
    1: "1. ",
    2: "\n\n2. ",
    3: "\n\n3. ",
    4: "\n\n4. ",
    5: "\n\n5. ",
    6: "\n\n6. "
}

def split_into_sections(caption: str) -> dict:
    section_indices = []
    for i in range(1, 7):
        prefix = required_prefixes[i]
        idx = caption.find(prefix)
        if idx != -1:
            section_indices.append((i, idx))
    section_indices.sort(key=lambda x: x[1])
    
    sections = {}
    for i in range(len(section_indices)):
        cur_num, start_idx = section_indices[i]
        end_idx = section_indices[i + 1][1] if i + 1 < len(section_indices) else len(caption)
        sections[cur_num] = caption[start_idx:end_idx].strip()
    
    return sections

def extract_random_sections(caption1: str, caption2: str):
    sections2 = split_into_sections(caption2)
    n = len(sections2)
    if n == 0:
        return "", ""  # caption2 不含任何段落
    
    x = random.randint(1, n)
    selected_indices = random.sample(list(sections2.keys()), x)
    
    # 从 caption1 和 caption2 中抽取相同编号段落
    sections1 = split_into_sections(caption1)
    new_caption1 = "\n\n".join([sections1[i] for i in selected_indices if i in sections1])
    new_caption2 = "\n\n".join([sections2[i] for i in selected_indices])

    return new_caption1, new_caption2, selected_indices

class ContrastiveTextPairDatasetWithAug(Dataset):
    """
    读取每行形如 {"caption1": "...", "caption2": "..."}
    的 JSONL 文件，返回 (caption1, caption2) 字符串 tuple
    """
    def __init__(self, jsonl_file: Union[str, Path]):
        self.path = Path(jsonl_file)
        assert self.path.exists(), f"{self.path} not found"
        self.samples: List[Tuple[str, str]] = self._load()

    def _load(self) -> List[Tuple[str, str]]:
        samples = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                new_cap1, new_cap2, idxs = extract_random_sections(obj["caption1"], obj["caption2"])
                samples.append( (new_cap1, new_cap2) )
        return samples

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx): return self.samples[idx]   # (str, str)


class ContrastiveTextPairDataset(Dataset):
    def __init__(self, jsonl_file: Union[str, Path]):
        self.path = Path(jsonl_file)
        assert self.path.exists(), f"{self.path} not found"
        self.samples: List[Tuple[str, str]] = self._load()

    def _load(self) -> List[Tuple[str, str]]:
        samples = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                samples.append( (obj["processed_caption1"], obj["processed_caption2"]) )
        return samples

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx): return self.samples[idx]   # (str, str)
