#!/bin/bash
# set -e                                  # 任何一步报错就退出
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export LD_PRELOAD=/share/mayanqi/libnccl.so.2.23.4
# export NCCL_TOPO_FILE=/share/baidu-nccl/topo_a800_hpc_bcc.xml
export DISABLE_MLFLOW_INTEGRATION=TRUE

export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export NCCL_ALGO=^NVLS,NVLSTree
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

# 路径可按需调整
DEEPSPEED_BIN="/ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/miniconda3/envs/t5gamma-new/bin/deepspeed"
PYTHON_BIN="/ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/miniconda3/envs/t5gamma-new/bin/python"
hostfile=/etc/mpi/hostfile

TRAIN_SCRIPT="train_single_node_multi_cuda.py"
CONFIG_DIR="configs_per_layer"

#python /ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/proj_text_enc/attn_pool_contrastive/clear.py


# 你要依次跑的 18 个 yaml（按想要的顺序列出来就行）
CONFIGS=( 
#qwen3vl_thinking_last_8_2layers.yaml
qwen3vl_last-7_2layers.yaml
qwen3vl_last-1_2layers.yaml
qwen3vl_last-5_2layers.yaml
qwen3vl_last-28_2layers.yaml
qwen3vl_thinking_last-28_2layers.yaml
qwen3vl_last-10_2layers.yaml
qwen3vl_last-22_2layers.yaml
qwen3vl_last-25_2layers.yaml
qwen3vl_thinking_last-35_2layers.yaml
qwen3vl_last-31_2layers.yaml
qwen3vl_thinking_last-24_2layers.yaml
qwen3vl_thinking_last-4_2layers.yaml
qwen3vl_thinking_last-10_2layers.yaml
qwen3vl_thinking_last-23_2layers.yaml
qwen3vl_thinking_last-33_2layers.yaml
qwen3vl_last-6_2layers.yaml
qwen3vl_last-4_2layers.yaml
qwen3vl_thinking_last-13_2layers.yaml
qwen3vl_last-32_2layers.yaml
qwen3vl_thinking_last-6_2layers.yaml
qwen3vl_thinking_last-36_2layers.yaml
qwen3vl_last-26_2layers.yaml
qwen3vl_last-3_2layers.yaml
qwen3vl_last-8_2layers.yaml
qwen3vl_thinking_last-30_2layers.yaml
qwen3vl_thinking_last-8_2layers.yaml
qwen3vl_thinking_last-16_2layers.yaml
qwen3vl_thinking_last-19_2layers.yaml
qwen3vl_thinking_last-14_2layers.yaml
qwen3vl_thinking_last-1_2layers.yaml
qwen3vl_last-23_2layers.yaml
qwen3vl_last-21_2layers.yaml
qwen3vl_last-16_2layers.yaml
qwen3vl_last-24_2layers.yaml
qwen3vl_thinking_last-32_2layers.yaml
qwen3vl_thinking_last-2_2layers.yaml
qwen3vl_last-36_2layers.yaml
qwen3vl_thinking_last-11_2layers.yaml
qwen3vl_thinking_last-17_2layers.yaml
qwen3vl_last-34_2layers.yaml
qwen3vl_last-2_2layers.yaml
qwen3vl_last-11_2layers.yaml
qwen3vl_thinking_last-7_2layers.yaml
qwen3vl_last-18_2layers.yaml
qwen3vl_last-12_2layers.yaml
qwen3vl_thinking_last-21_2layers.yaml
qwen3vl_thinking_last-31_2layers.yaml
qwen3vl_thinking_last-20_2layers.yaml
qwen3vl_last-13_2layers.yaml
qwen3vl_last-15_2layers.yaml
qwen3vl_thinking_last-26_2layers.yaml
qwen3vl_thinking_last-15_2layers.yaml
qwen3vl_last-17_2layers.yaml
qwen3vl_thinking_last-9_2layers.yaml
qwen3vl_last-9_2layers.yaml
qwen3vl_thinking_last-22_2layers.yaml
qwen3vl_thinking_last-27_2layers.yaml
qwen3vl_thinking_last-34_2layers.yaml
qwen3vl_last-33_2layers.yaml
qwen3vl_last-35_2layers.yaml
qwen3vl_last-20_2layers.yaml
qwen3vl_last-30_2layers.yaml
qwen3vl_last-27_2layers.yaml
qwen3vl_last-29_2layers.yaml
qwen3vl_thinking_last-29_2layers.yaml
qwen3vl_last-19_2layers.yaml
qwen3vl_thinking_last-18_2layers.yaml
qwen3vl_thinking_last-25_2layers.yaml
qwen3vl_thinking_last-5_2layers.yaml
qwen3vl_thinking_last-3_2layers.yaml
qwen3vl_thinking_last-12_2layers.yaml
qwen3vl_last-14_2layers.yaml
)

for cfg in "${CONFIGS[@]}"; do
  echo "=============================="
  echo ">>> $(date '+%F %T')  开始运行: $cfg"
  echo "=============================="

   真正调用 deepspeed
  "${DEEPSPEED_BIN}" --hostfile $hostfile --master_port 29599 \
      "${TRAIN_SCRIPT}" \
      --config "${CONFIG_DIR}/${cfg}"

  echo ">>> $(date '+%F %T')  结束运行: $cfg"
  #python /ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/proj_text_enc/attn_pool_contrastive/clear.py

  #echo
  #如需间隔休息可取消下一行注释
  #sleep 5

  #CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" \
  #    eval_embed_statement.py \
  #    --config "${CONFIG_DIR}/${cfg}"

  #python /ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/proj_text_enc/attn_pool_contrastive/clear.py

done
