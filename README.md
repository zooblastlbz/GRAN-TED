# GRAN-TED

> 对应论文：`GRAN-TED.pdf`

本仓库提供 GRAN-TED 的训练与评测实现，支持多机多卡（deepspeed/torchrun）训练与统一的评测入口。

## 环境准备
- Python 环境下确保可安装本项目依赖（`torch`、`transformers` 等）。
- 运行时请设置 `PYTHONPATH=src`，或将本项目安装为可导入包。
- 默认使用 HuggingFace 上的模型名称（可在配置中覆盖 `model.model_name`）。

## 训练
入口：`train_single_node_multi_cuda.py`

- 配置格式：使用嵌套的 `configs/*.yaml`，示例见 `configs/example_qwen3vl.yaml`（字段：`data` / `model` / `optimizer` / `trainer` / `seed`）。
- 单机多卡（示例）：
  ```bash
  PYTHONPATH=src torchrun --nproc_per_node=8 \
    train_single_node_multi_cuda.py \
    --config configs/example_qwen3vl.yaml
  ```
- 也可使用仓库提供的 `train.sh`，按需调整 deepspeed 路径、hostfile、配置列表。

## 评测
入口：`eval_embed_statement.py`

```bash
PYTHONPATH=src python eval_embed_statement.py \
  --config configs/example_qwen3vl.yaml \
  --load_epoch 1 \
  --input_file <eval_json> \
  --output_path <out_dir>
```
评测脚本复用同一套配置格式，从指定 `epoch_*.pt` 中加载池化头。

## 目录结构（核心）
- `src/granted/`：配置 dataclass、数据管线、模型封装、训练工具。
- `train_single_node_multi_cuda.py`：唯一训练入口（DDP/deepspeed 友好）。
- `eval_embed_statement.py`：唯一评测入口。
- `configs/`：示例与自定义训练/评测配置。
- `GRAN-TED.pdf`：对应论文。

## 关键注意事项
- 训练脚本依赖嵌套配置，不再支持旧的扁平字段。
- 默认 DataLoader `drop_last=True`，如需保留尾批可自行调整。
- 在分布式环境下运行前，请确保 NCCL/网络环境已正确配置。
