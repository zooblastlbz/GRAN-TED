from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import logging, os, sys, datetime


import torch
import torch.nn.functional as F
import torch.distributed as dist
import logging
from typing import Tuple, Dict, Optional


# losses.py
from distributed_utils import gather_features

def info_nce_local_rows_global_cols(
    z1, z2, temperature=0.07, gather_with_grad=False, use_float32_logits=True
):
    """
    - 先 L2-normalize 再传进来
    - 每张卡只对本地行 (2B_local) 计算 CE
    - 列使用跨卡 gather 后的全局特征（开放负样本池）
    """
    B = z1.shape[0]
    # 全局列
    z1_all, z2_all = gather_features(z1, z2, with_grad=gather_with_grad)
    reps_cols = torch.cat([z1_all, z2_all], 0)                # [2B_total, D]
    reps_rows = torch.cat([z1,     z2    ], 0)                # [2B_local, D]

    if use_float32_logits:
        logits = (reps_rows.float() @ reps_cols.float().T) / float(temperature)
        logits = logits.to(reps_rows.dtype)
    else:
        logits = (reps_rows @ reps_cols.T) / temperature

    # mask 自身（同一向量 vs 自己）
    import torch.distributed as dist
    rk = dist.get_rank() if dist.is_initialized() else 0
    ws = dist.get_world_size() if dist.is_initialized() else 1
    offset = rk * B
    global_B = B * ws

    idx = torch.arange(B, device=logits.device)
    # 行 0..B-1 对应本地 z1；它们的“自身 z1 列”是 offset+idx，置 -inf
    logits[idx, offset + idx] = float("-inf")
    # 行 B..2B-1 对应本地 z2；它们的“自身 z2 列”是 global_B + offset+idx
    logits[idx + B, global_B + offset + idx] = float("-inf")

    # 数值稳定
    logits = logits - logits.max(dim=-1, keepdim=True).values
    logits = logits.clamp(-15, 15)

    # 正样本列：z1_i ↔ z2_(global offset + i)，z2_i ↔ z1_(global offset + i)
    pos_cols_for_z1 = global_B + (offset + idx)          # [B]
    pos_cols_for_z2 = (offset + idx)                     # [B]
    labels = torch.cat([pos_cols_for_z1, pos_cols_for_z2], 0).long()

    loss = F.cross_entropy(logits, labels)
    return loss



# ---------- InfoNCE / NT-Xent ----------
def info_nce(
    z1: torch.Tensor,            # [B, D]  已 L2-norm
    z2: torch.Tensor,            # [B, D]
    temperature: float = 1,
    gather_distributed: bool = True,
) -> torch.Tensor:
    """
    NT-Xent / InfoNCE
    -----------------
      • z1_i ↔ z2_i 为正样本，其余为负样本
      • softmax 形式的对比损失
      • 行-最大值中心化 + clamp 提升数值稳定
    """
    # ---- 可选：跨卡 gather 扩大负样本池 ----
    if gather_distributed and dist.is_initialized():
        ws = dist.get_world_size()
        cat1 = [torch.zeros_like(z1) for _ in range(ws)]
        cat2 = [torch.zeros_like(z2) for _ in range(ws)]
        dist.all_gather(cat1, z1.contiguous())
        dist.all_gather(cat2, z2.contiguous())
        z1, z2 = torch.cat(cat1, 0), torch.cat(cat2, 0)

    print(f"[{datetime.datetime.now()}] z1 shape: {z1.shape}, z2 shape: {z2.shape}")
    # ---- 拼接，两视图合并到 2B ----
    reps = torch.cat([z1, z2], dim=0)          # [2B, D]

    # ---- 相似度矩阵 (余弦 / τ) ----
    logits = reps @ reps.T / temperature       # [2B,2B]

    # ---- mask 对角线：自身 vs 自身 ----
    B = z1.size(0)
    diag = torch.eye(2*B, dtype=torch.bool, device=logits.device)
    logits.masked_fill_(diag, float("-inf"))

    # ---- 数值稳定：减行最大 & 限幅 ----
    logits -= logits.max(dim=-1, keepdim=True).values
    logits = logits.clamp(-15, 15)

    # ---- labels：正样本列索引 ----
    labels = torch.arange(B, device=logits.device)
    labels = torch.cat([labels + B, labels])   # [2B]

    # ---- InfoNCE 损失 ----
    loss = F.cross_entropy(logits, labels)
    return loss

# ---------------------------------------

class ContrastiveModel(nn.Module):
    """
    组合:
        · wrapper  (冻结)
        · attn_pool(可训练)
    一次性计算 loss 与嵌入
    """
    def __init__(
        self,
        wrapper: nn.Module,           # LLM / VL 等已冻结
        attn_pool: nn.Module,         # 需要训练
        temperature: float = 1,
        gather_distributed: bool = True,
    ):
        super().__init__()
        self.wrapper = wrapper
        wrapper.eval()  # 确保 wrapper 冻结
        wrapper.requires_grad_(False)
        self.attn_pool = attn_pool
        self.tau = temperature
        self.gather = gather_distributed

    # 方便 Trainer 过滤可学习参数
    def trainable_parameters(self):
        return self.attn_pool.parameters()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        batch keys 由 collate 生成:
            input_ids1, attention_mask1, input_ids2, attention_mask2
        Returns:
            loss, z1, z2
        """
        # ---- 第一句 ----
        h1, m1 = self.wrapper(
            input_ids=batch.get("input_ids1", None),
            attention_mask=batch.get("attention_mask1", None),
            texts=batch.get("texts1", None)  # 兼容文本输入
        )
        # ---- 第二句 ----
        h2, m2 = self.wrapper(
            input_ids=batch.get("input_ids2", None),
            attention_mask=batch.get("attention_mask2", None),
            texts=batch.get("texts2", None)  # 兼容文本输入
        )

        if torch.isnan(h1).any():
            print("h1  contains NaN values!")
        if torch.isnan(h2).any():
            print("h2  contains NaN values!")

        h1 = h1.to(next(self.attn_pool.parameters()).dtype)
        h2 = h2.to(next(self.attn_pool.parameters()).dtype)

        # 分别检查 h1 和 h2 有没有 nan
        if torch.isnan(h1).any():
            print("h1 after align contains NaN values!")
        if torch.isnan(h2).any():
            print("h2 after align contains NaN values!")

        # ---- Attention Pooling ----
        z1 = self.attn_pool(h1, attention_mask=m1)
        z2 = self.attn_pool(h2, attention_mask=m2)

        
        # ---- 打印 z1 和 z2 ----
        if torch.is_tensor(z1) and torch.is_tensor(z2):
            logging.info(f"[INFO] z1: mean={z1.mean().item():.4f}, std={z1.std().item():.4f}")
            logging.info(f"[INFO] z2: mean={z2.mean().item():.4f}, std={z2.std().item():.4f}")
            logging.info(f"[INFO] z1 norm: {z1.norm(p=2, dim=-1).mean().item():.4f}, " f"z2 norm: {z2.norm(p=2, dim=-1).mean().item():.4f}")
            # logging.info(f"[INFO] z1 shape: {z1.shape}, z2 shape: {z2.shape}")
            # logging.info(f"[Z1] {z1[0].detach().cpu().numpy()}")
            # logging.info(f"[Z2] {z2[0].detach().cpu().numpy()}")

        # ---- L2 normalize ----
        z1 = F.normalize(z1, dim=-1, eps=1e-8)
        z2 = F.normalize(z2, dim=-1, eps=1e-8)

        # ---- InfoNCE loss ----
        loss = info_nce_local_rows_global_cols(z1, z2)
        # loss = contrastive_loss(z1, z2, margin=1.0)

        if torch.is_tensor(z1) and torch.is_tensor(z2):
        # ---- 打印正样本的余弦相似度 ----
            pos_sim = (z1 * z2).sum(dim=-1)               # [B]
            pos_min = pos_sim.min().item()
            pos_max = pos_sim.max().item()
            logging.info(f"[INFO] 正样本余弦相似度: {pos_sim.mean().item():.4f}, "
                f"min={pos_min:.4f}, max={pos_max:.4f}")

        # ---- 打印负样本的余弦相似度 ----
            # 构造 [B, B] 相似度矩阵，去掉对角线
            sim_mat = torch.matmul(z1, z2.T)              # 每行 z1_i vs z2_j
            B = z1.size(0)
            neg_mask = ~torch.eye(B, dtype=torch.bool, device=sim_mat.device)
            neg_sim_values = sim_mat[neg_mask]            # 取出所有 i≠j 的项

            neg_mean = neg_sim_values.mean().item()
            neg_min  = neg_sim_values.min().item()
            neg_max  = neg_sim_values.max().item()

            logging.info(f"[INFO] 负样本余弦相似度: mean={neg_mean:.4f}, "
                f"min={neg_min:.4f}, max={neg_max:.4f}")



        return loss, z1, z2
