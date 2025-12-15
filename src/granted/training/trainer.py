# trainer.py
from __future__ import annotations
import math, os, time
from typing import Dict, Any, Optional, Callable

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import logging


class Trainer:
    """
    通用对比学习 Trainer（DDP 友好版）
    只更新 model.attn_pool
    """

    # ---------- 初始化 ----------
    def __init__(
        self,
        model:            torch.nn.Module,
        optimizer:        torch.optim.Optimizer,
        loader:           DataLoader,
        *,
        epochs:           int,
        accumulation_steps: int = 1,
        max_grad_norm:    Optional[float] = 0.5,
        warmup_epochs:    float = 0.01,
        amp:              bool = True,
        log_fn:           Optional[Callable[[Dict[str, Any]], None]] = None,
        ckpt_dir:         Optional[str] = "./checkpoints",
        save_every_steps: int = 400,
        num_max_saved:    int = 5,
        writer:           Optional[SummaryWriter] = None,
        num_layers:       int = 2,
        lr_schedule:      str = "constant",
    ):
        self.model, self.optimizer = model, optimizer
        self.loader      = loader
        self.epochs      = epochs
        self.accum_steps = accumulation_steps
        self.grad_clip   = max_grad_norm
        self.warmup_steps = int(warmup_epochs * len(loader))
        self.total_steps  = int(epochs * len(loader))
        self.amp         = amp
        self.log_fn      = log_fn or (lambda *_: None)
        self.raw_model = self.model.module if hasattr(self.model, "module") else self.model
        self.num_layers = num_layers
        self.lr_schedule = lr_schedule

        # -------- 分布式 ----------
        self.distributed = dist.is_initialized()
        self.rank        = dist.get_rank() if self.distributed else 0
        self.world_size  = dist.get_world_size() if self.distributed else 1
        self.is_master   = self.rank == 0

        # -------- 日志 / TensorBoard / ckpt ----------
        if self.is_master and ckpt_dir is not None:
            os.makedirs(ckpt_dir, exist_ok=True)
        self.ckpt_dir = ckpt_dir if self.is_master else None
        self.save_every_steps = save_every_steps
        self.num_max_saved    = num_max_saved
        self.saved_meta: list[tuple[int, int]] = []           # (epoch, step)
        self.writer = writer if self.is_master else None

        # -------- 学习率调度 ----------
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.step_num = 0

        # -------- AMP ----------
        self.scaler = GradScaler(enabled=amp)

        if self.is_master:
            logging.info(f"[Trainer] world_size={self.world_size}, "
                         f"total_steps={self.total_steps}, "
                         f"warmup_steps={self.warmup_steps}")

    # ---------- 内部：LR 调度 ----------
    def _lr_schedule(self) -> float:
        if self.lr_schedule == "constant":
            return self.base_lr
        elif self.lr_schedule == "cosine":
            if self.step_num < self.warmup_steps:
                return self.base_lr * self.step_num / max(1, self.warmup_steps)
            t = self.step_num - self.warmup_steps
            T = max(1, self.total_steps - self.warmup_steps)
            return self.base_lr * 0.5 * (1 + math.cos(math.pi * t / T))
        else:
            raise ValueError(f"Unknown lr_schedule: {self.lr_schedule}")


    def _set_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    # ---------- 训练主循环 ----------
    def train(self):
        self.model.train()

        # rank>0 用哑 tqdm；主进程正常 tqdm
        def make_iter(dl):
            return tqdm(dl, total=len(dl), dynamic_ncols=True) if self.is_master else dl
        

        for epoch in range(1, self.epochs + 1):
            if self.distributed and isinstance(self.loader.sampler, torch.utils.data.DistributedSampler):
                self.loader.sampler.set_epoch(epoch)

            epoch_loss = 0.0
            start_time = time.time()

            for step, batch in enumerate(make_iter(self.loader), 1):
                lr = self._lr_schedule()
                self._set_lr(lr)

                with autocast(enabled=self.amp):
                    loss, _, _ = self.model(batch)          # model 内部已做 gather
                    loss = loss / self.accum_steps

                self.scaler.scale(loss).backward()
                epoch_loss += loss.item()

                if step % self.accum_steps == 0:
                    # 梯度 norm（仅统计，不裁剪）
                    self.scaler.unscale_(self.optimizer)
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.raw_model.attn_pool.parameters(), max_norm=float("inf")
                    ).item()

                    # 可选裁剪
                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.raw_model.attn_pool.parameters(), self.grad_clip
                        )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    # ---- log grad norm ----
                    if self.is_master:
                        logging.info(f"[GRAD] step={self.step_num:7d} "
                                     f"gnorm={total_grad_norm:.4f}")
                        if self.writer:
                            self.writer.add_scalar("train/grad_norm",
                                                   total_grad_norm, self.step_num)
                    # log loss
                    if self.is_master:
                        logging.info(f"[LOSS] step={self.step_num:7d} "
                                     f"loss={loss.item():.4f} lr={lr:.3e}")
                        if self.writer:
                            self.writer.add_scalar("train/loss_step", loss.item(), self.step_num)

                # ---- tqdm 可视化 ----
                if self.is_master and isinstance(batch, dict):
                    # tqdm 已经在主进程
                    tqdm.write("")  # 刷新缓冲，避免混行
                self.step_num += 1

                # ---- 每 N 步保存 ckpt (主进程) ----
                if self.is_master and \
                   self.ckpt_dir and self.step_num % self.save_every_steps == 0:
                    self._save_ckpt(epoch, loss.item())

                # ---- TensorBoard 按 step 记录 ----
                if self.is_master and self.writer and step % self.accum_steps == 0:
                    self.writer.add_scalar("train/loss_step", loss.item(), self.step_num)
                    self.writer.add_scalar("train/lr", lr, self.step_num)

            # ---------------- 每轮结束 ----------------
            # loss 汇总（平均到全局）
            avg_loss = epoch_loss / len(self.loader)
            if self.distributed:
                loss_tensor = torch.tensor(avg_loss, device="cuda")
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                avg_loss = (loss_tensor / self.world_size).item()

            if self.is_master:
                # 自定义 log_fn
                self.log_fn({"epoch": epoch, "loss": avg_loss, "lr": lr})
                if self.writer:
                    self.writer.add_scalar("train/loss_epoch", avg_loss, epoch)
                # 最后一轮或每轮都存
                self._save_ckpt_final(epoch, avg_loss)
                logging.info(
                    f"[Epoch {epoch}/{self.epochs}] "
                    f"loss={avg_loss:.4f} lr={lr:.3e} "
                    f"time={time.time()-start_time:.1f}s"
                )

            if self.distributed:
                dist.barrier()        # 同步后再进入下一轮

    # ---------- 保存 ckpt（仅主进程调用） ----------
    def _save_ckpt(self, epoch: int, loss: float):
        if not self.ckpt_dir:
            return
        ckpt_name = f"step_{self.step_num}.pt"
        path = os.path.join(self.ckpt_dir, ckpt_name)
        torch.save({
            "epoch":     epoch,
            "step_num":  self.step_num,
            "loss":      loss,
            "model":     self.raw_model.attn_pool.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)
        self.saved_meta.append((epoch, self.step_num))

        # 保留最近 num_max_saved 个 ckpt
        if len(self.saved_meta) > self.num_max_saved:
            old_epoch, old_step = self.saved_meta.pop(0)
            old_name = f"step_{old_step}.pt"
            try:
                os.remove(os.path.join(self.ckpt_dir, old_name))
                logging.info(f"[CKPT] removed {old_name}")
            except FileNotFoundError:
                pass

    def _save_ckpt_final(self, epoch: int, loss: float):
        if not self.ckpt_dir:
            return
        ckpt_name = f"epoch_{epoch}.pt"
        path = os.path.join(self.ckpt_dir, ckpt_name)
        torch.save({
            "epoch":     epoch,
            "step_num":  self.step_num,
            "loss":      loss,
            "model":     self.raw_model.attn_pool.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)
        self.saved_meta.append((epoch, self.step_num))

        # 保留最近 num_max_saved 个 ckpt
        if len(self.saved_meta) > self.num_max_saved:
            old_epoch, old_step = self.saved_meta.pop(0)
            old_name = f"step_{old_step}.pt"
            try:
                os.remove(os.path.join(self.ckpt_dir, old_name))
                logging.info(f"[CKPT] removed {old_name}")
            except FileNotFoundError:
                pass

    # ---------- 析构 ----------
    def __del__(self):
        if getattr(self, "writer", None):
            self.writer.close()
