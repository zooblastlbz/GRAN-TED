from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class DataConfig:
    path: str
    batch_size: int
    num_workers: int = 0
    max_len: int = 512


@dataclass
class ModelConfig:
    backbone: str
    layers_to_select: Union[int, List[int], str] = -1
    select_all_layers_or_not: bool = False
    adapter: str = "attn"
    proj_dim: int = 128
    num_layers: int = 2
    use_rope: bool = True
    temperature: float = 0.1
    model_name: Optional[str] = None      # for t5_gemma etc.
    task: Optional[str] = None            # for jina_v4
    max_len: int = 512                    # tokenizer truncation fallback
    norm_type: str = "layer_norm"


@dataclass
class OptimizerConfig:
    lr: float
    weight_decay: float = 1e-4


@dataclass
class TrainerConfig:
    output: str
    epochs: int
    accumulation_steps: int = 1
    warmup_epochs: float = 0.01
    amp: bool = True
    save_every_steps: int = 400
    num_max_saved: int = 5
    lr_schedule: str = "constant"  # constant | cosine


@dataclass
class TrainConfig:
    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    trainer: TrainerConfig
    seed: int = 42

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TrainConfig":
        """
        Convert a dict (e.g., yaml.safe_load) into structured config.
        Supports both the new nested layout and the legacy flat keys.
        """
        if "data" in raw and isinstance(raw["data"], (str, Path)):
            raw["data"] = {"path": raw["data"]}

        # ---- legacy flat keys compatibility ----
        if "output" in raw and "trainer" not in raw:
            raw["trainer"] = {
                "output": raw.get("output"),
                "epochs": raw.get("epochs"),
                "accumulation_steps": raw.get("accum_steps", raw.get("accumulation_steps", 1)),
                "warmup_epochs": raw.get("warmup_epochs", 0.01),
                "amp": raw.get("amp", True),
                "save_every_steps": raw.get("save_every_steps", 400),
                "num_max_saved": raw.get("num_max_saved", 5),
                "lr_schedule": raw.get("lr_schedule", "constant"),
            }
        if "backbone" in raw and "model" not in raw:
            raw["model"] = {
                "backbone": raw.get("backbone"),
                "layers_to_select": raw.get("layers_to_select", -1),
                "select_all_layers_or_not": raw.get("select_all_layers_or_not", False),
                "adapter": raw.get("adapter", "attn"),
                "proj_dim": raw.get("proj_dim", 128),
                "num_layers": raw.get("num_layers", 2),
                "use_rope": raw.get("use_rope", True),
                "temperature": raw.get("temperature", 0.1),
                "model_name": raw.get("model_name"),
                "task": raw.get("task"),
                "max_len": raw.get("max_len", 512),
                "norm_type": raw.get("norm_type", "layer_norm"),
            }
        # drop deprecated keys if present
        if "model" in raw and isinstance(raw["model"], dict):
            for k in ("use_softmax_weights", "load_softmax_weights", "trainable_layer_weights", "use_norm", "use_post_norm"):
                raw["model"].pop(k, None)
        if "data" in raw and isinstance(raw["data"], dict):
            raw["data"].setdefault("batch_size", raw.get("batch_size"))
            raw["data"].setdefault("num_workers", raw.get("num_workers", 0))
            raw["data"].setdefault("max_len", raw.get("max_len", 512))

        if "optimizer" not in raw:
            raw["optimizer"] = {
                "lr": raw.get("lr"),
                "weight_decay": raw.get("weight_decay", 1e-4),
            }

        _require_keys(raw, ["data", "model", "optimizer", "trainer"])

        data_cfg = DataConfig(**raw["data"])
        model_cfg = ModelConfig(**raw["model"])
        optim_cfg = OptimizerConfig(**raw["optimizer"])
        trainer_cfg = TrainerConfig(**raw["trainer"])
        seed = raw.get("seed", 42)
        return cls(
            data=data_cfg,
            model=model_cfg,
            optimizer=optim_cfg,
            trainer=trainer_cfg,
            seed=seed,
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TrainConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError("Config file must contain a mapping at top level.")
        return cls.from_dict(raw)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _require_keys(cfg: Dict[str, Any], keys: List[str]) -> None:
    missing = [k for k in keys if k not in cfg or cfg[k] is None]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
