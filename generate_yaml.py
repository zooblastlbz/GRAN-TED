from regex import F
import yaml
import os

# 基础配置（除 layers_to_select 外的所有字段）

base_config={
    "batch_size": 16,
    "epochs": 1,
    "lr": 1.0e-05,
    "proj_dim": 1024,
    "max_len": 1024,
    "use_rope": True,
    "weight_decay": 1.0e-05,
    "accum_steps": 1,
    "warmup_epochs": 0.0,
    "amp": False,
    "save_every_steps": 10000,
    "num_max_saved": 25,    
    "data": "data/raw/processed_adapter_data.jsonl",
    "output": "runs_per_layer/qwen3vl_last-8_2layers",
    "select_all_layers_or_not": False,
    "use_softmax_weights": False,
    "load_softmax_weights": None,
    "backbone": "qwen3vl",
    "adapter": "attn",
    "num_layers": 2,
    "layers_to_select": -8,
    "use_norm": False,

}

output_dir = "/ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/proj_text_enc/attn_pool_contrastive/configs_per_layer/"
os.makedirs(output_dir, exist_ok=True)
file_list=[]
# 生成 layers_to_select 从 -1 到 -36 的配置文件
for i in range(1, 37):  # i 从 1 到 36
    layer_val = -i
    config = base_config.copy()
    config["layers_to_select"] = layer_val
    config["output"]=f"runs_per_layer/qwen3vl_last{layer_val}_2layers"

    # 构造文件名，例如: config_layer_-1.yaml
    filename = f"qwen3vl_last{layer_val}_2layers.yaml"
    filepath = os.path.join(output_dir, filename)

    # 写入 YAML 文件
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    file_list.append(filename)
print(file_list)

list=os.listdir(output_dir)

for i in list:
    print(i)

print(f"✅ 已生成 36 个 YAML 配置文件，保存在 '{output_dir}' 目录中。")