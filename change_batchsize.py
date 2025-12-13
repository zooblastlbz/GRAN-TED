import os
import yaml

def update_batch_size_in_yaml_files(folder_path, new_batch_size):
    """
    遍历指定文件夹下所有 .yaml 或 .yml 文件，
    修改其中的 batch_size 字段，并写回原文件。
    
    :param folder_path: 包含 YAML 文件的目录路径
    :param new_batch_size: 要设置的新 batch_size 值（如 32）
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"指定路径不是有效目录: {folder_path}")

    # 支持 .yaml 和 .yml 后缀
    yaml_files = [f for f in os.listdir(folder_path)
                  if f.lower().endswith(('.yaml', '.yml'))]

    for filename in yaml_files:
        filepath = os.path.join(folder_path, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if data is None:
                print(f"⚠️ 警告: {filename} 为空，跳过。")
                continue

            if 'batch_size' not in data:
                print(f"⚠️ 警告: {filename} 中没有 'batch_size' 字段，跳过。")
                continue

            old_value = data['batch_size']
            data['batch_size'] = new_batch_size
            print(f"✅ {filename}: batch_size 从 {old_value} 修改为 {new_batch_size}")

            # 写回文件（保持原有格式风格，但 YAML 格式可能略有变化）
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, allow_unicode=True, sort_keys=False)

        except Exception as e:
            print(f"❌ 处理 {filename} 时出错: {e}")

# ===== 使用示例 =====
if __name__ == "__main__":
    folder = "/ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/proj_text_enc/attn_pool_contrastive/configs_cos"          # 替换为你自己的目录路径
    new_bs = 32                 # 替换为你想要的 batch_size
    update_batch_size_in_yaml_files(folder, new_bs)