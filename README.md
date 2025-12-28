<!-- <img src="image/logo.png" width="38" height="38" alt="">  -->
<h1 align="center"> GRAN-TED: Generating Robust, Aligned, and Nuanced Text Embedding for Diffusion Models </h1>

<p align="center">
<a href='https://arxiv.org/abs/2512.15560'><img src='https://img.shields.io/badge/arXiv-2512.15560-b31b1b.svg'></a>&nbsp;






## Environment Setup

```bash
conda create -n granted python=3.10 -y
conda activate granted
pip install -r requirements.txt
pip install -e .
export PYTHONPATH=$(pwd)
```

## Configure YAML for Evaluation
`eval_embed_statement.py` reads the same YAML used for training. To evaluate a model, duplicate `configs/example_qwen3vl.yaml` and edit these keys:
- `model.backbone`: backend type, e.g., `qwen3vl`, `qwen25vl`, `llama3`, `t5_gemma`, etc. (must match a wrapper in `models/llm_wrapper.py` or `models/embedding_wrapper.py`).
- `model.model_name`: HF repo ID or local path to the checkpoint you want to evaluate.
- `model.layers_to_select` / `select_all_layers_or_not`: which hidden layers to pool; `-1` or `true` mirrors the training recipe.
- `model.adapter`, `proj_dim`, `num_layers`, `use_rope`, `norm_type`: keep consistent with the training setup of the pooling head you will load.
- (Optional) `model.dim_in` and `num_llm_layers`: override if your backbone hidden size/layer count differs from the defaults inferred in the code.
- `trainer.output`: directory that stores the pooling checkpoints (`epoch_*.pt`). This must point to the run you want to load.
- `data` block is ignored in evaluation, but can stay as-is.

Minimal example for evaluation:
```yaml
model:
  backbone: qwen3vl
  model_name: /path/to/Qwen-VL-weights
  adapter: attn
  proj_dim: 1024
  layers_to_select: -1
  select_all_layers_or_not: true
  use_rope: true
  num_layers: 2
trainer:
  output: runs/qwen3vl_example   # contains epoch_1.pt, epoch_*.pt
```

## Quick Start
### Training
```bash
PYTHONPATH=$(pwd) torchrun --nproc_per_node=8 \
  train_single_node_multi_cuda.py \
  --config configs/example_qwen3vl.yaml
```
Checkpoints of the pooling head are saved under `trainer.output` as `step_*.pt` and `epoch_*.pt`.

### Evaluation (caption–statement retrieval)
```bash
PYTHONPATH=$(pwd) python eval_embed_statement.py \
  --config configs/example_qwen3vl.yaml \
  --load_epoch 1 \
  --input_file data/TED-6k.json \
  --output_path outputs/eval \
  --batch_size 24
```
The script loads the frozen backbone plus the trained pooling head from `trainer.output/epoch_<load_epoch>.pt`, computes embeddings for captions/statements, reports accuracy by type, and saves a JSON report to `output_path`.



## Citation
If you use GRAN-TED in your research, please cite:
```
@article{li2025gran,
  title={GRAN-TED: Generating Robust, Aligned, and Nuanced Text Embedding for Diffusion Models},
  author={Li, Bozhou and Yang, Sihan and Guan, Yushuo and An, Ruichuan and Chen, Xinlong and Shi, Yang and Wan, Pengfei and Zhang, Wentao and others},
  journal={arXiv preprint arXiv:2512.15560},
  year={2025}
}
```
