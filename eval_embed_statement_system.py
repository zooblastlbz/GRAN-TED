import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'  # 设置可见的GPU设备
import json
import argparse
import torch
from typing import List, Dict
from collections import defaultdict
import numpy as np

# ------------------------------------------------------------
# ❶ 解析命令行 —— 只保留 config、输入/输出、device、batch_size
# ------------------------------------------------------------
def parse_args():
    import argparse, os
    p = argparse.ArgumentParser("Caption-Statement Embedding Eval")
    p.add_argument("--config", required=True, help="训练用的 yaml")
    p.add_argument("--input_file", type=str, default='/ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/proj_text_enc/attn_pool_contrastive/data/eval_data/annotated_question_answer_statements_filtered.json')
    p.add_argument("--output_path", type=str, default='./eval_output')
    p.add_argument("--load_step", type=str, default='final')
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=24)
    return p.parse_args()


# ------------------------------------------------------------
# ❷ 根据 config 创建 wrapper
# ------------------------------------------------------------
from models.encoder_wrapper import T5GemmaWrapper
from models.llm_wrapper import Qwen25Wrapper, Qwen25VLWrapper, Qwen3Wrapper
from models.embedding_wrapper import Qwen3EmbedEmbeddingWrapper, Qwen3EmbedSequenceWrapper, JINAv4Wrapper

def build_wrapper(cfg: dict, device: str):
    bb = cfg.get("backbone")
    kw = dict(
        device=device,
        layers_to_select     = cfg.get("layers_to_select", -1),
        select_all_layers_or_not = cfg.get("select_all_layers_or_not", False),
        bf16 = False,
    )
    if bb == "qwen25":
        return Qwen25Wrapper(**kw)
    elif bb == "qwen25vl":
        return Qwen25VLWrapper(**kw)
    elif bb == "qwen3embed_seq":
        return Qwen3EmbedSequenceWrapper(**kw)
    elif bb == "qwen3embed_embed":
        # 句级 embedding 可设置 normalize=True 避免重复归一化
        return Qwen3EmbedEmbeddingWrapper(**kw)
    elif bb == "qwen3":
        return Qwen3Wrapper(**kw)
    elif bb == "jina_v4":
        if cfg["task"] == "retrieval":
            task = "retrieval"  # 或 "text-matching"
        else:
            task = "text-matching"
        wrapper = JINAv4Wrapper(device=device,
                                task=task,  # 或 "retrieval"
                                prompt_name=None)
        return wrapper
    elif bb == "t5_gemma":
        return T5GemmaWrapper(model_name=cfg.get("model_name", "/ytech_m2v5_hdd/workspace/kling_mm/Models/t5gemma-2b-2b-ul2"),
                              device=device,
                              layers_to_select=cfg["layers_to_select"],
                              select_all_layers_or_not=cfg["select_all_layers_or_not"])
    else:
        raise ValueError(f"Unsupported backbone: {bb}")


# ------------------------------------------------------------
# ❸ 根据 config 创建 pooling 头
# ------------------------------------------------------------
from models.attention_pooling import AttnPooling, MeanPooling


def build_pool(cfg: dict, hidden_dim: int, device: str, load_step:str, num_llm_layers: int = 32) -> torch.nn.Module:
    if cfg["adapter"] == "attn":
        m = AttnPooling(dim=hidden_dim,
                        dim_out=cfg["proj_dim"],
                        num_layers=cfg.get("num_layers", 2),
                        use_rope=cfg.get("use_rope", True),
                        select_all_layers_or_not=cfg["select_all_layers_or_not"],
                        num_llm_layers=num_llm_layers)
    else:
        m = MeanPooling(dim=hidden_dim,
                        dim_out=cfg["proj_dim"])
    ckpt_path = os.path.join(cfg["output"], f"step_{load_step}.pt")
    if os.path.exists(ckpt_path):
        print(f"Loading pooling weights from {ckpt_path}")
        state_dict = torch.load(ckpt_path)
        m.load_state_dict(state_dict['model'], strict=True)
    else:
        print(f"Warning: Pooling checkpoint {ckpt_path} does not exist. Using uninitialized pooling.")
    m = m.to(torch.float16)
    return m.to(device)


# ------------------------------------------------------------
# ❹ 封装推理模型
# ------------------------------------------------------------
from models.embedding_model import EmbeddingModel
import yaml

def deploy_embedding_model(cfg_path: str, device: str, load_step: str) -> EmbeddingModel:
    cfg = yaml.safe_load(open(cfg_path))
    wrapper = build_wrapper(cfg, device).to(torch.float16)
    if cfg["backbone"] == "jina_v4":
        dim_in = 128
    elif cfg["backbone"] == "t5_gemma":
        dim_in = wrapper.model.config.encoder.hidden_size
    else:
        dim_in  = wrapper.model.config.hidden_size
    hidden_dim = dim_in
    if cfg["backbone"] == "t5_gemma":
        num_llm_layers = wrapper.model.config.encoder.num_hidden_layers
    else:
        num_llm_layers = wrapper.model.config.num_hidden_layers
    pool = build_pool(cfg, hidden_dim, device, load_step, num_llm_layers).to(torch.float16)
    return EmbeddingModel(wrapper, pool,
                          normalize=True,      # 最后再 L2-norm
                          device=device)


# ------------------------------------------------------------
# compute_embeddings 用 EmbeddingModel.encode
# ------------------------------------------------------------
def compute_embeddings(model: EmbeddingModel,
                       texts, batch_size: int) -> torch.Tensor:
    print(f"Encoding {len(texts)} texts ...")
    return model.encode(texts, batch_size=batch_size)


def load_statement_data(data_path: str) -> List[Dict]:
    """加载statement数据并转换为测评格式"""
    print(f"加载statement数据文件: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        statement_data = json.load(f)
    
    eval_data = []
    
    for caption_key, caption_info in statement_data.items():
        caption = caption_info['caption']
        caption_type = caption_info['caption_type']
        statements_dict = caption_info['statements']
        
        for question_type, statement_info in statements_dict.items():
            question = statement_info['question']
            
            # 检查是否有正确答案的statement
            if statement_info['correct_statement'] is None:
                continue
            
            correct_statement = statement_info['correct_statement']
            if correct_statement['status'] != 'success' or correct_statement['statement'] is None:
                continue
            
            # 收集所有选项（正确答案 + 错误答案）
            options = []
            option_letters = []
            correct_answer_letter = 'A'  # 正确答案总是第一个选项
            
            # 添加正确答案
            options.append(correct_statement['statement'])
            option_letters.append('A')
            
            # 添加错误答案
            incorrect_statements = statement_info.get('incorrect_statements', [])
            available_letters = ['B', 'C', 'D', 'E', 'F', 'G', 'H']
            
            for idx, incorrect_statement in enumerate(incorrect_statements):
                if (incorrect_statement['status'] == 'success' and 
                    incorrect_statement['statement'] is not None and 
                    idx < len(available_letters)):
                    options.append(incorrect_statement['statement'])
                    option_letters.append(available_letters[idx])
            
            # 至少需要2个选项才能进行测评
            if len(options) < 2:
                continue
            
            # 构建选项字符串
            options_str = '\n'.join([f"{letter}. {statement}" for letter, statement in zip(option_letters, options)])
            
            eval_item = {
                'caption': caption,
                'caption_type': caption_type,
                'question': question,
                'question_type': question_type,
                'options': options_str,
                'correct_answer': correct_answer_letter,
                'caption_key': caption_key,
                'statements': options,  # 保存所有statement用于嵌入计算
                'option_mapping': {i: letter for i, letter in enumerate(option_letters)}
            }
            
            eval_data.append(eval_item)
    
    print(f"转换得到 {len(eval_data)} 个测评样本")
    return eval_data


def filter_data(data: List[Dict]) -> List[Dict]:
    """过滤掉image和Temporal的组合"""
    filtered_data = []
    excluded_count = 0
    
    for item in data:
        caption_type = item['caption_type']
        question_type = item['question_type']
        
        # 去除image和Temporal的组合
        if caption_type == 'image' and question_type == 'Temporal':
            excluded_count += 1
            continue
        
        filtered_data.append(item)
    
    print(f"过滤掉 {excluded_count} 个 image+Temporal 组合，剩余 {len(filtered_data)} 个样本")
    return filtered_data


def prepare_captions_and_statements(data: List[Dict]) -> tuple:
    """准备caption和statement文本用于嵌入计算"""
    #task_description = "Given a caption, find the most relevant statement that describes the content"
    
    # 准备caption查询
    caption_queries = []
    all_statements = []
    statement_mappings = []  # 记录每个样本的statement映射
    
    for i, item in enumerate(data):
        caption_type = item['caption_type']
        caption = item['caption']
        
        # 构建caption查询文本
        caption_text = f"Caption Type: {caption_type}\nCaption: {caption}"
        #caption_query = get_detailed_instruct(task_description, caption_text)
        caption_query=caption_text  # 直接使用文本作为查询
        caption_query = f"You are an assistant designed to generate high-quality images or videos based on user prompts. {caption_query} "
        
        caption_queries.append(caption_query)
        
        # 获取当前样本的所有statements
        statements = item['statements']
        for statement in statements:
            statement = f"You are an assistant designed to generate high-quality images or videos based on user prompts. {statement} "
            all_statements.append(statement)
        option_mapping = item['option_mapping']
        
        # all_statements.extend(statements)
        statement_mappings.append({
            'start_idx': len(all_statements) - len(statements),
            'end_idx': len(all_statements),
            'mapping': option_mapping,
            'num_options': len(statements)
        })
    
    return caption_queries, all_statements, statement_mappings


def find_best_statements(caption_embeddings: torch.Tensor, statement_embeddings: torch.Tensor, 
                        statement_mappings: List[Dict]) -> List[tuple]:
    """为每个caption找到最相似的statement"""
    results = []
    
    for i, mapping in enumerate(statement_mappings):
        start_idx = mapping['start_idx']
        end_idx = mapping['end_idx']
        option_mapping = mapping['mapping']
        
        # 获取当前样本的statement嵌入
        sample_statement_embeddings = statement_embeddings[start_idx:end_idx]
        
        # 计算相似度分数
        caption_emb = caption_embeddings[i:i+1]  # 保持维度
        similarities = torch.mm(caption_emb, sample_statement_embeddings.T)
        
        # 找到最高相似度的statement
        best_idx = similarities.argmax().item()
        best_score = similarities[0, best_idx].item()
        best_option = option_mapping.get(best_idx, str(best_idx))
        
        results.append((best_option, best_score, similarities[0].tolist()))
    
    return results


def calculate_detailed_accuracy(data: List[Dict], predictions: List[str]) -> Dict:
    """计算详细的准确率统计"""
    # 初始化统计字典
    caption_type_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    question_type_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    combined_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    total_correct = 0
    total_num = 0
    
    for i, (item, pred) in enumerate(zip(data, predictions)):
        true_answer = item.get('correct_answer', '').strip().upper()
        pred_answer = pred.strip().upper()
        is_correct = (true_answer == pred_answer)
        
        caption_type = item['caption_type']
        question_type = item['question_type']
        combined_key = f"{caption_type}_{question_type}"
        
        # 更新总体统计
        total_num += 1
        if is_correct:
            total_correct += 1
        
        # 更新caption type统计
        caption_type_stats[caption_type]['total'] += 1
        if is_correct:
            caption_type_stats[caption_type]['correct'] += 1
        
        # 更新question type统计
        question_type_stats[question_type]['total'] += 1
        if is_correct:
            question_type_stats[question_type]['correct'] += 1
        
        # 更新组合统计
        combined_stats[combined_key]['total'] += 1
        if is_correct:
            combined_stats[combined_key]['correct'] += 1
        
        # 添加预测结果到数据中
        item['prediction'] = pred_answer
        item['is_correct'] = is_correct
    
    # 计算准确率
    overall_accuracy = total_correct / total_num if total_num > 0 else 0
    
    # 计算各类别准确率
    caption_type_accuracy = {}
    for ctype, stats in caption_type_stats.items():
        caption_type_accuracy[ctype] = {
            'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0,
            'correct': stats['correct'],
            'total': stats['total']
        }
    
    question_type_accuracy = {}
    for qtype, stats in question_type_stats.items():
        question_type_accuracy[qtype] = {
            'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0,
            'correct': stats['correct'],
            'total': stats['total']
        }
    
    combined_accuracy = {}
    for combined_key, stats in combined_stats.items():
        combined_accuracy[combined_key] = {
            'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0,
            'correct': stats['correct'],
            'total': stats['total']
        }
    
    return {
        'overall_accuracy': overall_accuracy,
        'total_correct': total_correct,
        'total_samples': total_num,
        'caption_type_accuracy': caption_type_accuracy,
        'question_type_accuracy': question_type_accuracy,
        'combined_accuracy': combined_accuracy
    }


def print_detailed_results(accuracy_stats: Dict):
    """打印详细的准确率结果"""
    print(f"\n{'='*60}")
    print(f"详细准确率统计报告 (Caption-Statement Embedding)")
    print(f"{'='*60}")
    
    # 总体准确率
    print(f"总体准确率: {accuracy_stats['overall_accuracy']:.2%} ({accuracy_stats['total_correct']}/{accuracy_stats['total_samples']})")
    
    # Caption Type准确率
    print(f"\n按Caption Type分类:")
    print(f"{'-'*40}")
    for ctype, stats in sorted(accuracy_stats['caption_type_accuracy'].items()):
        print(f"  {ctype}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    
    # Question Type准确率
    print(f"\n按Question Type分类:")
    print(f"{'-'*40}")
    for qtype, stats in sorted(accuracy_stats['question_type_accuracy'].items()):
        print(f"  {qtype}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    
    # 组合统计
    print(f"\n按Caption Type + Question Type组合分类:")
    print(f"{'-'*50}")
    for combined_key, stats in sorted(accuracy_stats['combined_accuracy'].items()):
        caption_type, question_type = combined_key.split('_', 1)
        print(f"  {caption_type} + {question_type}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")


def save_results(data: List[Dict], predictions: List[tuple], output_file: str, accuracy_stats: Dict):
    """保存结果"""
    results = []
    
    for i, (item, (pred_option, pred_score, all_scores)) in enumerate(zip(data, predictions)):
        result = {
            'index': i,
            'caption_key': item['caption_key'],
            'caption_type': item['caption_type'],
            'question_type': item['question_type'],
            'caption': item['caption'],
            'question': item['question'],
            'options': item['options'],
            'statements': item['statements'],
            'correct_answer': item.get('correct_answer', ''),
            'predicted_answer': pred_option,
            'confidence_score': pred_score,
            'all_scores': all_scores,
            'is_correct': item.get('is_correct', False)
        }
        results.append(result)
    
    # 添加详细统计信息
    summary = {
        'model_type': 'Caption-Statement-Embedding',
        'total_samples': len(results),
        'accuracy_statistics': accuracy_stats,
        'average_confidence': np.mean([r['confidence_score'] for r in results])
    }
    
    output_data = {
        'summary': summary,
        'results': results
    }
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)


# ============================ main ===========================
def main():
    args = parse_args()

    # 1) 数据加载 + 过滤（保持原逻辑）
    data = load_statement_data(args.input_file)
    print(f"加载了 {len(data)} 个样本")
    data = filter_data(data)
    caption_queries, statements, mappings = prepare_captions_and_statements(data)
    print(f"准备了 {len(caption_queries)} 个caption查询和 {len(statements)} 个statement")
    print(caption_queries[0])
    print(statements[:5])  # 打印前5个statement以检查格式

    # 2) 部署推理模型
    embed_model = deploy_embedding_model(args.config, args.device, args.load_step)

    # 3) 计算嵌入
    caption_emb = compute_embeddings(embed_model, caption_queries, args.batch_size)
    stmt_emb    = compute_embeddings(embed_model, statements,    args.batch_size)

    # 4-9) 相似度、评测、保存（同旧脚本）
    predictions = find_best_statements(caption_emb, stmt_emb, mappings)
    pred_opts   = [p[0] for p in predictions]
    acc_stats   = calculate_detailed_accuracy(data, pred_opts)
    print_detailed_results(acc_stats)
    output_file = os.path.join(args.output_path, f'{args.config.split("/")[-1].replace(".yaml", "")}_{args.load_step}.json')
    save_results(data, predictions, output_file, acc_stats)
    print(f"结果已保存到 {output_file}")

if __name__ == "__main__":
    main()
