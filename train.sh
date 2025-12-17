#!/bin/bash
# set -e  # 如需遇错退出可取消注释

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export LD_PRELOAD=/share/mayanqi/libnccl.so.2.23.4
# export NCCL_TOPO_FILE=/share/baidu-nccl/topo_a800_hpc_bcc.xml
export DISABLE_MLFLOW_INTEGRATION=TRUE

export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export NCCL_ALGO=^NVLS,NVLSTree
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

# 设置 PYTHONPATH，确保可以导入 granted 包
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH}"

# 按需调整路径
DEEPSPEED_BIN=""
PYTHON_BIN=""
HOSTFILE="/etc/mpi/hostfile"

# 训练脚本与配置目录
TRAIN_SCRIPT="train_single_node_multi_cuda.py"
CONFIG_DIR="configs"

# 需要顺序跑的配置列表
CONFIGS=(
  example_qwen3vl.yaml
)

for cfg in "${CONFIGS[@]}"; do
  echo "=============================="
  echo ">>> $(date '+%F %T')  开始运行: $cfg"
  echo "=============================="

  "${DEEPSPEED_BIN}" --hostfile "${HOSTFILE}" --master_port 29599 \
      "${TRAIN_SCRIPT}" \
      --config "${CONFIG_DIR}/${cfg}"

  echo ">>> $(date '+%F %T')  结束运行: $cfg"
  # 如需训练后立刻评测，可取消下方注释
  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" eval_embed_statement.py --config "${CONFIG_DIR}/${cfg}"
done
