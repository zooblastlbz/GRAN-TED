# distributed_utils.py
import torch
import torch.distributed as dist
from torch.autograd import Function

# ------ 可选：拿数据并行组（DeepSpeed 时更稳），否则用 WORLD ------
def get_dp_group():
    try:
        import deepspeed
        if deepspeed.comm.is_initialized():
            return deepspeed.comm.get_data_parallel_group()
    except Exception:
        pass
    return None

# ------ 等长 batch 的 "带梯度" all_gather ------
class _AllGatherWithGrad(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group=None):
        if not dist.is_initialized():
            ctx.world_size, ctx.rank, ctx.local_n = 1, 0, x.size(0)
            return x
        ws = dist.get_world_size(group=group)
        rk = dist.get_rank(group=group)
        bufs = [torch.empty_like(x) for _ in range(ws)]
        dist.all_gather(bufs, x.contiguous(), group=group)
        ctx.world_size, ctx.rank, ctx.local_n = ws, rk, x.size(0)
        return torch.cat(bufs, dim=0)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        # 仅把属于本 rank 的那一段梯度切片回传（无需通信）
        if ctx.world_size == 1:
            return grad_out, None
        n = ctx.local_n
        s = ctx.rank * n
        e = s + n
        return grad_out[s:e].contiguous(), None

def all_gather_with_grad(x: torch.Tensor, group=None) -> torch.Tensor:
    return _AllGatherWithGrad.apply(x, group)

# ------ 等长 batch 的 "不带梯度" all_gather（推荐做负样本池） ------
@torch.no_grad()
def all_gather_nograd(x: torch.Tensor, group=None) -> torch.Tensor:
    if not dist.is_initialized():
        return x
    ws = dist.get_world_size(group=group)
    bufs = [torch.empty_like(x) for _ in range(ws)]
    dist.all_gather(bufs, x.contiguous(), group=group)
    return torch.cat(bufs, dim=0)

# ------ 统一接口：返回 (z1_all, z2_all) ------
def gather_features(z1: torch.Tensor, z2: torch.Tensor, with_grad: bool = False):
    """
    等长 batch 假设；各 rank 的 z1/z2 形状均为 [B, D]。
    with_grad=False：跨卡样本仅作“列”的负样本池，**不回传梯度**（更稳，open-clip 默认）。
    with_grad=True ：跨卡样本也接收梯度（更重，不稳定，慎用）。
    """
    group = get_dp_group()
    if with_grad:
        return all_gather_with_grad(z1, group), all_gather_with_grad(z2, group)
    else:
        return all_gather_nograd(z1, group), all_gather_nograd(z2, group)
