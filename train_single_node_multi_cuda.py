# train_ddp.py
# ============
import argparse, os, sys, torch, torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import random
import numpy as np
from pathlib import Path

# 期待通过环境变量提供 PYTHONPATH=src:${REPO_ROOT}
from datasets.contrastive_dataset import ContrastiveTextPairDataset
from datasets.collate import make_contrastive_collate, make_jina_contrastive_collate
from models.encoder_wrapper import T5GemmaWrapper, UMT5Wrapper, CLIPTextWrapper
from models.llm_wrapper import (
    Qwen25Wrapper,
    Qwen25VLWrapper,
    Qwen3Wrapper,
    InternVL3Wrapper,
    KwaiLlavaWrapper,
    MiniCPMWrapper,
    OvisWrapper,
    Qwen3VLWrapper,
    Llama3Wrapper,
)
from models.embedding_wrapper import Qwen3EmbedEmbeddingWrapper, Qwen3EmbedSequenceWrapper, JINAv4Wrapper
from models.kling_wrapper import KlingBaseWrapper
from models.attention_pooling import AttnPooling, MeanPooling
from models.contrastive_model import ContrastiveModel
from trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
import logging, sys

from config import TrainConfig  # type: ignore

# ---------- CLI ----------
p = argparse.ArgumentParser()
p.add_argument("--config", required=True, help="YAML 配置文件")
p.add_argument("--local_rank", type=int, default=-1,
               help="Deepspeed / torchrun 会自动传入，用来区分每块卡的进程")
args = p.parse_args()

# ---------- 分布式初始化 ----------
distributed = args.local_rank != -1
if distributed:
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
else:
    world_size = 1


# ---------- 读取配置 ----------
cfg_obj = TrainConfig.from_yaml(args.config)
cfg = cfg_obj.to_dict()
data_cfg = cfg["data"]
model_cfg = cfg["model"]
trainer_cfg = cfg["trainer"]
optimizer_cfg = cfg["optimizer"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 对所有 GPU 生效
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed = cfg_obj.seed
set_seed(seed + (args.local_rank if distributed else 0))


# ---------- 日志 ----------
def setup_logger(save_dir: str, filename: str = "train.log", enable: bool = True):
    if not enable:
        return
    os.makedirs(save_dir, exist_ok=True)
    logfile = os.path.join(save_dir, filename)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(logfile, mode="a", encoding="utf-8"),
        ],
    )
is_master = (not distributed) or (dist.get_rank() == 0)
setup_logger(trainer_cfg["output"], enable=is_master)
if is_master:
    logging.info(f"DDP world size = {world_size}")

# ---------- 设备 ----------
device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
if is_master:
    logging.info(f"[INFO] Use device: {device}")



# ---------- Backbone Wrapper ----------
backbone = model_cfg["backbone"]
if backbone == "qwen25":
    wrapper = Qwen25Wrapper(
        device=device,
        layers_to_select=model_cfg["layers_to_select"],
        select_all_layers_or_not=model_cfg["select_all_layers_or_not"],
    )
elif backbone == "qwen25vl" or backbone == "xiaomi":
    wrapper = Qwen25VLWrapper(
        model_name=model_cfg.get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct"),
        device=device,
        layers_to_select=model_cfg["layers_to_select"],
        select_all_layers_or_not=model_cfg["select_all_layers_or_not"],
    )
elif backbone == "qwen3embed_seq":
    wrapper = Qwen3EmbedSequenceWrapper(
        device=device,
        layers_to_select=model_cfg["layers_to_select"],
        select_all_layers_or_not=model_cfg["select_all_layers_or_not"],
    )
elif backbone == "qwen3embed_embed":
    wrapper = Qwen3EmbedEmbeddingWrapper(
        device=device,
        layers_to_select=model_cfg["layers_to_select"],
        select_all_layers_or_not=model_cfg["select_all_layers_or_not"],
    )
elif backbone == "qwen3":
    wrapper = Qwen3Wrapper(
        device=device,
        model_name=model_cfg.get("model_name", "Qwen/Qwen2-7B-Instruct"),
        layers_to_select=model_cfg["layers_to_select"],
        select_all_layers_or_not=model_cfg["select_all_layers_or_not"],
    )
elif backbone == "qwen3vl":
    wrapper = Qwen3VLWrapper(
        device=device,
        model_name=model_cfg.get("model_name", "Qwen/Qwen2-VL-7B-Instruct"),
        layers_to_select=model_cfg["layers_to_select"],
        select_all_layers_or_not=model_cfg["select_all_layers_or_not"],
    )
elif backbone == "minicpm":
    wrapper = MiniCPMWrapper(
        device=device,
        layers_to_select=model_cfg["layers_to_select"],
        select_all_layers_or_not=model_cfg["select_all_layers_or_not"],
    )
elif backbone == "ovis2_5":
    wrapper = OvisWrapper(
        device=device,
        layers_to_select=model_cfg["layers_to_select"],
        select_all_layers_or_not=model_cfg["select_all_layers_or_not"],
    )
elif backbone == "jina_v4":
    task = "retrieval" if model_cfg.get("task") == "retrieval" else "text-matching"
    wrapper = JINAv4Wrapper(device=device, task=task, prompt_name=None)
elif backbone == "t5_gemma":
    wrapper = T5GemmaWrapper(
        model_name=model_cfg.get("model_name", "google/t5-v1_1-large"),
        device=device,
        layers_to_select=model_cfg["layers_to_select"],
        select_all_layers_or_not=model_cfg["select_all_layers_or_not"],
    )
elif backbone == "umt5":
    wrapper = UMT5Wrapper(
        model_name=model_cfg.get("model_name", "google/umt5-xxl"),
        device=device,
        layers_to_select=model_cfg["layers_to_select"],
        select_all_layers_or_not=model_cfg["select_all_layers_or_not"],
    )
elif backbone == "internvl3" or backbone == "internvl3_5":
    wrapper = InternVL3Wrapper(
        model_name=model_cfg.get("model_name", "OpenGVLab/InternVL3-8B"),
        device=device,
        layers_to_select=model_cfg["layers_to_select"],
        select_all_layers_or_not=model_cfg["select_all_layers_or_not"],
    )
elif backbone == "kwai_llava":
    wrapper = KwaiLlavaWrapper(
        device=device,
        layers_to_select=model_cfg["layers_to_select"],
        select_all_layers_or_not=model_cfg["select_all_layers_or_not"],
    )
elif backbone == "clip_text":
    wrapper = CLIPTextWrapper(
        device=device,
        layers_to_select=model_cfg["layers_to_select"],
        select_all_layers_or_not=model_cfg["select_all_layers_or_not"],
    )
elif backbone == "kling":
    wrapper = KlingBaseWrapper(device=device)
elif backbone == "llama3":
    wrapper = Llama3Wrapper(
        model_name=model_cfg.get("model_name", "meta-llama/Meta-Llama-3-8B-Instruct"),
        device=device,
        layers_to_select=model_cfg["layers_to_select"],
        select_all_layers_or_not=model_cfg["select_all_layers_or_not"],
    )
else:
    raise ValueError(f"Unsupported backbone: {backbone}")
wrapper.eval()
wrapper.requires_grad_(False)

# ---------- Data ----------
dataset = ContrastiveTextPairDataset(data_cfg["path"])
sampler = DistributedSampler(dataset, shuffle=True) if distributed else None

if backbone == "jina_v4" or backbone == "internvl3" or backbone == "kling":
    collate = make_jina_contrastive_collate(
        max_len=model_cfg.get("max_len", data_cfg.get("max_len", 1024)),
    )
else:
    collate = make_contrastive_collate(
        wrapper.tokenizer,
        max_len=model_cfg.get("max_len", data_cfg.get("max_len", 1024)),
        device=device,
    )
loader = DataLoader(
    dataset,
    batch_size=data_cfg["batch_size"],
    shuffle=(sampler is None),
    sampler=sampler,
    num_workers=data_cfg.get("num_workers", 0),
    collate_fn=collate,
    drop_last=True
)

# ---------- Model ----------
if "dim_in" in model_cfg.keys():
    dim_in = model_cfg["dim_in"]
elif backbone == "t5_gemma":
    dim_in  = wrapper.model.config.encoder.hidden_size
elif backbone == "umt5":
    dim_in  = wrapper.model.config.d_model
elif backbone == "jina_v4":
    dim_in  = 128
elif backbone == "kwai_llava":
    dim_in  = 4096
elif backbone == "internvl3" or backbone == "internvl3_5":
    try:
        dim_in  = wrapper.model.config.text_config.hidden_size
    except:
        dim_in  = wrapper.model.config.llm_config.hidden_size
elif backbone == "qwen3vl":
    dim_in = 4096
elif backbone == "qwen25vl":
    dim_in = 3584
elif backbone == "ovis2_5":
    dim_in  = wrapper.model.config.llm_config.hidden_size
elif backbone == "kling":
    dim_in  = 3584
elif backbone == "clip_text":
    dim_in =1024
elif backbone == "llama3":
    dim_in = 4096
else:
    dim_in  = wrapper.model.config.hidden_size

if "num_llm_layers" in model_cfg.keys():
    num_llm_layers = model_cfg["num_llm_layers"]
elif backbone == "t5_gemma":
    num_llm_layers = wrapper.model.config.encoder.num_hidden_layers
elif backbone == "umt5":
    num_llm_layers = wrapper.model.config.num_layers
elif backbone == "internvl3" or backbone == "internvl3_5":
    try:
        num_llm_layers = wrapper.model.config.text_config.num_hidden_layers
    except:
        num_llm_layers = wrapper.model.config.llm_config.num_hidden_layers
elif backbone == "qwen3vl":
    num_llm_layers = 36
elif backbone == "qwen25vl":
    num_llm_layers = 28
elif backbone == "kwai_llava":
    num_llm_layers = 36
elif backbone == "ovis2_5":
    num_llm_layers  = wrapper.model.config.llm_config.num_hidden_layers
elif backbone == "kling":
    num_llm_layers = 32
elif backbone == "clip_text":
    num_llm_layers =24
elif backbone == "llama3":
    num_llm_layers = 32
else:
    num_llm_layers = wrapper.model.config.num_hidden_layers

if model_cfg["adapter"] == "attn":
    
    attn    = AttnPooling(dim=dim_in,
                        layers_to_select=model_cfg.get("layers_to_select",-1),
                        norm_type=model_cfg.get("norm_type","layer_norm"),
                        dim_out=model_cfg["proj_dim"],
                        select_all_layers_or_not=model_cfg.get("select_all_layers_or_not",False),
                        use_rope=model_cfg.get("use_rope", True),
                        num_llm_layers=num_llm_layers).to(device)
    attn = attn.to(dtype=torch.bfloat16)   # AttnPooling 用半精度即可
    print(f"Attn Dtype: {next(attn.parameters()).dtype}")

else:
    attn    = MeanPooling(dim=dim_in,
                        dim_out=model_cfg["proj_dim"]).to(device)

model   = ContrastiveModel(wrapper, attn,
                           temperature=model_cfg.get("temperature", 0.1),
                           gather_distributed=False).to(device)

optimizer_cfg = cfg.get("optimizer", {})
optimizer = torch.optim.AdamW(
    model.trainable_parameters(),
    lr=float(optimizer_cfg.get("lr", cfg.get("lr", 1e-4))),
    weight_decay=float(optimizer_cfg.get("weight_decay", cfg.get("weight_decay", 1e-4))),
)

# ---------- DDP 包装 ----------
if distributed:
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
    )

# ---------- TensorBoard ----------
writer = SummaryWriter(os.path.join(trainer_cfg["output"], "tb")) if is_master else None

# ---------- Trainer ----------
def logger(d):
    if is_master:
        print(f'[Epoch {d["epoch"]}] loss={d["loss"]:.4f} lr={d["lr"]:.3e}')

trainer = Trainer(
    model, optimizer, loader,
    epochs=trainer_cfg["epochs"],
    accumulation_steps=trainer_cfg.get("accumulation_steps", cfg.get("accum_steps", 1)),
    warmup_epochs=trainer_cfg.get("warmup_epochs", 0.01),
    amp=trainer_cfg.get("amp", True),
    log_fn=logger,
    ckpt_dir=trainer_cfg["output"] if is_master else None,   # 只主进程存 ckpt
    save_every_steps=trainer_cfg.get("save_every_steps", 400),
    num_max_saved=trainer_cfg.get("num_max_saved", 5),
    writer=writer,
    num_layers=model_cfg.get("num_layers", 2),
    lr_schedule= trainer_cfg.get("lr_schedule", "constant"),
)
trainer.train()

# ---------- 清理 ----------
if distributed:
    dist.destroy_process_group()
