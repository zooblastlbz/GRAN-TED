#!/bin/bash
# set -e                                  # 任何一步报错就退出
export CUDA_VISIBLE_DEVICES=4,5,6,7
export LD_PRELOAD=/share/mayanqi/libnccl.so.2.23.4
export NCCL_TOPO_FILE=/share/baidu-nccl/topo_a800_hpc_bcc.xml
export DISABLE_MLFLOW_INTEGRATION=TRUE

export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_ALGO=^NVLS,NVLSTree
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

# 路径可按需调整
DEEPSPEED_BIN="/ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/miniconda3/envs/t5gamma-new/bin/deepspeed"
PYTHON_BIN="/ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/miniconda3/envs/t5gamma-new/bin/python"


SCRIPT="eval_embed_statement.py"
CONFIG_DIR="configs_paper"

#python /ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/proj_text_enc/attn_pool_contrastive/clear.py


# 你要依次跑的 18 个 yaml（按想要的顺序列出来就行）
CONFIGS=( 
#qwen3vl_ours_last2_2layers.yaml
#qwen3_4B_norm_avg_2layers.yaml
#qwen3vl_32B_norm_avg_2layers.yaml
#qwen3_32B_last-1_2layers.yaml
#ovis2_5_last2_2layers.yaml
#ovis2_5_norm_avg_2layers.yaml
#qwen3_32B_norm_avg_2layers.yaml
#qwen3_4B_last-2_2layers.yaml
#internvl3_last1_2layers-hf.yaml
#qwen3_thinking_4B_norm_avg_2layers.yaml
#qwen3vl_4B_norm_avg_2layers.yaml
#qwen3_4B_last-1_2layers.yaml
#qwen3vl_normavg_2layers.yaml
#clip_norm_avg_2layers.yaml
#xiaomi_norm_avg_2layers.yaml
#umt5xxl_norm_avg_2layers.yaml
#qwen3vl_thinking_last_2_2layers.yaml
#qwen3vl_4B_last-2_2layers.yaml
#qwen3vl_thinking_last_1_2layers.yaml
#qwen3vl_4B_last-1_2layers.yaml
#umt5xxl_last2_2layers.yaml
#umt5xxl_last1_2layers.yaml
#qwen3_norm_avg_2layers.yaml
#xiaomi_last1_2layers.yaml
#qwen3_thinking_4B_last-2_2layers.yaml
#qwen3vl_ours_norm_avg_2layers.yaml
#xiaomi_last2_2layers.yaml
#qwen3_32B_last-2_2layers.yaml
#qwen3_Base_norm_avg_2layers.yaml
#qwen3vl_thinking_norm_avg_2layers.yaml
#qwen3vl_ours_last1_2layers.yaml
qwen3_Base_4B_last-1_2layers.yaml
qwen3_Base_4B_last-2_2layers.yaml
qwen3_Base_4B_norm_avg_2layers.yaml
qwen3_last-1_2layers.yaml
qwen3_last-2_2layers.yaml
qwen3_norm_avg_2layers.yaml
qwen3vl_ours_norm_avg_2layers.yaml
)

NUM_GPUS=8
TOTAL=${#CONFIGS[@]}

echo ">>> 总任务数: $TOTAL"
echo ">>> GPU 数量: $NUM_GPUS"
echo ">>> 将分 $(( (TOTAL + NUM_GPUS - 1) / NUM_GPUS )) 批运行"
echo

# === 主循环：按批次并发执行 ===
for (( epoch = 1; epoch < 2; epoch += 1 )); do
  for (( start = 0; start < TOTAL; start += NUM_GPUS )); do
      end=$(( start + NUM_GPUS - 1 ))
      if [[ $end -ge $TOTAL ]]; then
          end=$(( TOTAL - 1 ))
      fi
      echo "=============================="
      echo ">>> $(date '+%F %T') 启动批次: 任务 $((start + 1)) ~ $((end + 1)) (共 $((end - start + 1)) 个)"
      echo "=============================="

      # 启动本批次所有任务（后台运行）
      for (( i = start; i <= end; i++ )); do
          cfg="${CONFIGS[$i]}"
          gpu_id=$(( i % NUM_GPUS )) 
          echo "    [$((i + 1))] 分配 $cfg 到 GPU $gpu_id"

          # 后台执行任务
          (
              # 子 shell 中设置环境
              export CUDA_VISIBLE_DEVICES=$gpu_id
              # 执行命令
              "$PYTHON_BIN" "$SCRIPT" --config "${CONFIG_DIR}/${cfg}" --load_epoch "$epoch" 
              # 捕获退出状态
              exit_code=$?
              if [[ $exit_code -ne 0 ]]; then
                  echo ">>> ERROR: $cfg (GPU $gpu_id) 失败，退出码: $exit_code" >&2
              else
                  echo ">>> SUCCESS: $cfg (GPU $gpu_id) 完成" >&2
              fi
              exit $exit_code
          ) &
      done

      # 等待本批次所有任务完成
      echo ">>> 等待本批次 $((end - start + 1)) 个任务完成..."
      wait
      echo ">>> $(date '+%F %T') 批次 $((start / NUM_GPUS + 1)) 完成"
      echo
      sleep 3  # 批次间稍作休息
    done
done
