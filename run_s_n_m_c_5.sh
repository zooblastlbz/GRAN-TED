#!/bin/bash
# set -e                                  # 任何一步报错就退出
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LD_PRELOAD=/share/mayanqi/libnccl.so.2.23.4
export NCCL_TOPO_FILE=/share/baidu-nccl/topo_a800_hpc_bcc.xml
export DISABLE_MLFLOW_INTEGRATION=TRUE

export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export NCCL_ALGO=^NVLS,NVLSTree
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

# 路径可按需调整
DEEPSPEED_BIN="/ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/miniconda3/envs/t5gamma/bin/deepspeed"
PYTHON_BIN="/ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/miniconda3/envs/t5gamma/bin/python"
hostfile=/etc/mpi/hostfile

TRAIN_SCRIPT="train_single_node_multi_cuda.py"
CONFIG_DIR="configs_911"

python /ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/proj_text_enc/attn_pool_contrastive/clear.py


# 你要依次跑的 18 个 yaml（按想要的顺序列出来就行）
CONFIGS=( 
    internvl3_5_last1_2layers-hf.yaml
    internvl3_5_last1_2layers.yaml
    internvl3_5_last2_2layers-hf.yaml
    internvl3_5_last2_2layers.yaml
    internvl3_5_normavg_2layers-hf.yaml
    internvl3_5_normavg_2layers.yaml
    internvl3_last1_2layers-hf.yaml
    internvl3_last1_2layers.yaml
    internvl3_last2_2layers-hf.yaml
    internvl3_last2_2layers.yaml
    internvl3_normavg_2layers-hf.yaml
    internvl3_normavg_2layers.yaml
    internvl3_ours_last1_2layers-4500.yaml
    internvl3_ours_last1_2layers-7000.yaml
    internvl3_ours_last2_2layers-4500.yaml
    internvl3_ours_last2_2layers-7000.yaml
    internvl3_ours_normavg_2layers-4500.yaml
    internvl3_ours_normavg_2layers-7000.yaml
    ovis2_5_last1_2layers.yaml
    ovis2_5_last2_2layers.yaml
    ovis2_5_normavg_2layers.yaml
)

for cfg in "${CONFIGS[@]}"; do
  echo "=============================="
  echo ">>> $(date '+%F %T')  开始运行: $cfg"
  echo "=============================="

  # 真正调用 deepspeed
  "${DEEPSPEED_BIN}" --hostfile $hostfile --master_port 29501 \
      "${TRAIN_SCRIPT}" \
      --config "${CONFIG_DIR}/${cfg}"

  echo ">>> $(date '+%F %T')  结束运行: $cfg"
  python /ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/proj_text_enc/attn_pool_contrastive/clear.py

  # echo
  # 如需间隔休息可取消下一行注释
  sleep 5

#   CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" \
#       eval_embed_statement.py \
#       --config "${CONFIG_DIR}/${cfg}"

  python /ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/proj_text_enc/attn_pool_contrastive/clear.py

done
