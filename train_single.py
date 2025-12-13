# train.py  —— 仅保留单卡路径
import argparse, yaml, os, torch
from torch.utils.data import DataLoader

import random
import numpy as np

from datasets.contrastive_dataset import ContrastiveTextPairDataset
from datasets.collate import make_contrastive_collate, make_jina_contrastive_collate
from models.encoder_wrapper import T5GemmaWrapper
from models.llm_wrapper import Qwen25Wrapper, Qwen25VLWrapper, Qwen3Wrapper
from models.embedding_wrapper import Qwen3EmbedEmbeddingWrapper, Qwen3EmbedSequenceWrapper, JINAv4Wrapper
from models.attention_pooling import AttnPooling, MeanPooling
from models.contrastive_model import ContrastiveModel
from trainer import Trainer

from torch.utils.tensorboard import SummaryWriter
import logging, os, sys, datetime

def setup_logger(save_dir: str, filename: str = "train.log"):
    os.makedirs(save_dir, exist_ok=True)
    logfile = os.path.join(save_dir, filename)

    logging.basicConfig(
        level=logging.INFO,                       # 全局最低等级
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),    # 终端
            logging.FileHandler(logfile, mode="a", encoding="utf-8"),  # 文件
        ],
    )
    logging.info(f"Logger set up. Log file: {logfile}")


# ---------- CLI ----------
p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
args = p.parse_args()
# 调用
# ---------- config ----------
with open(args.config) as f:
    cfg = yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 对所有 GPU 生效
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(cfg.get("seed", 42))  # 如果 cfg 中没设置，就默认 42


setup_logger(cfg["output"])
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Use device: {device}")

# ---------- Data ----------
dataset = ContrastiveTextPairDataset(cfg["data"],)

# ---------- Backbone Wrapper ----------
backbone = cfg["backbone"]
if backbone == "qwen25":
    wrapper = Qwen25Wrapper(device=device,
                            layers_to_select=cfg["layers_to_select"],
                            select_all_layers_or_not=cfg["select_all_layers_or_not"])
elif backbone == "qwen25vl":
    wrapper = Qwen25VLWrapper(device=device,
                              layers_to_select=cfg["layers_to_select"],
                              select_all_layers_or_not=cfg["select_all_layers_or_not"])
elif backbone == "qwen3embed_seq":
    wrapper = Qwen3EmbedSequenceWrapper(device=device,
                                        layers_to_select=cfg["layers_to_select"],
                                        select_all_layers_or_not=cfg["select_all_layers_or_not"])
elif backbone == "qwen3embed_embed":
    wrapper = Qwen3EmbedEmbeddingWrapper(device=device,
                                         layers_to_select=cfg["layers_to_select"],
                                         select_all_layers_or_not=cfg["select_all_layers_or_not"])
elif backbone == "qwen3":
    wrapper = Qwen3Wrapper(device=device,
                           layers_to_select=cfg["layers_to_select"],
                           select_all_layers_or_not=cfg["select_all_layers_or_not"])
elif backbone == "jina_v4":
    if cfg["task"] == "retrieval":
        task = "retrieval"  # 或 "text-matching"
    else:
        task = "text-matching"
    wrapper = JINAv4Wrapper(device=device,
                            task=task,  # 或 "retrieval"
                            prompt_name=None)
elif backbone == "t5_gemma":
    wrapper = T5GemmaWrapper(model_name=cfg.get("model_name", "/ytech_m2v5_hdd/workspace/kling_mm/Models/t5gemma-2b-2b-ul2"),
                             device=device,
                             layers_to_select=cfg["layers_to_select"],
                             select_all_layers_or_not=cfg["select_all_layers_or_not"])
else:
    raise ValueError(f"Unsupported backbone: {backbone}")

wrapper.eval()  # 确保 wrapper 冻结
wrapper.requires_grad_(False)
if backbone == "jina_v4":
    collate = make_jina_contrastive_collate(
        max_len=cfg.get("max_len", 512),
    )
else:
    collate = make_contrastive_collate(
        wrapper.tokenizer,
        max_len=cfg.get("max_len", 512),
        device=device,
    )

loader = DataLoader(
    dataset,
    batch_size=cfg["batch_size"],
    shuffle=True,
    num_workers=cfg.get("num_workers", 0),
    collate_fn=collate,
)

# ---------- Model ----------
if backbone == "jina_v4":
    dim_in = 128
elif backbone == "t5_gemma":
    dim_in = wrapper.model.config.encoder.hidden_size
else:
    dim_in  = wrapper.model.config.hidden_size

if backbone == "t5_gemma":
    num_llm_layers = wrapper.model.config.encoder.num_hidden_layers
else:
    num_llm_layers = wrapper.model.config.num_hidden_layers

if cfg["adapter"] == "attn":
    attn    = AttnPooling(dim=dim_in,
                        dim_out=cfg["proj_dim"],
                        num_layers=cfg.get("num_layers", 2),
                        use_rope=cfg.get("use_rope", True),
                        select_all_layers_or_not=cfg["select_all_layers_or_not"],
                        num_llm_layers=num_llm_layers).to(device)
else:
    attn    = MeanPooling(dim=dim_in,
                        dim_out=cfg["proj_dim"]).to(device)

model   = ContrastiveModel(wrapper, attn,
                           temperature=cfg.get("temperature", 0.1),
                           gather_distributed=False).to(device)

# ---------- Optimizer ----------
optimizer = torch.optim.AdamW(
    model.attn_pool.parameters(),
    lr=float(cfg["lr"]),
    weight_decay=float(cfg.get("weight_decay", 1e-4)),
)

# ---------- Trainer ----------
tb_dir = os.path.join(cfg["output"], "tb")
writer = SummaryWriter(tb_dir)
def logger(d): print(f'[Epoch {d["epoch"]}] loss={d["loss"]:.4f} lr={d["lr"]:.3e}')


trainer = Trainer(
    model, optimizer, loader,
    epochs=cfg["epochs"],
    accumulation_steps=cfg.get("accum_steps", 1),
    warmup_epochs=cfg.get("warmup_epochs", 1),
    amp=cfg.get("amp", True),
    log_fn=logger,
    ckpt_dir=cfg["output"],
    save_every_steps=cfg.get("save_every_steps", 400),
    num_max_saved=cfg.get("num_max_saved", 5),
    writer=writer,
)
trainer.train()
