import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'  # 设置可见的GPU设备
import json
import argparse
import torch
from typing import List, Dict
from collections import defaultdict
from vllm import LLM
import numpy as np


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Statement-Based Embedding 模型测评脚本')
    
    parser.add_argument('--model_path', type=str, default='/pfs/Models/Qwen3-Embedding-4B/', help='模型路径')
    parser.add_argument('--input_file', type=str, default='/ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/proj_text_enc/attn_pool_contrastive/data/eval_data/annotated_question_answer_statements_filtered.json', help='输入 statement JSON 文件路径')
    parser.add_argument('--output_file', type=str, default='/ytech_m2v5_hdd/workspace/kling_mm/libozhou/text_encoder/eval/output/qwen3-embedding-4b-statement-embedding-anno.json', help='输出文件路径')
    parser.add_argument('--tensor_parallel_size', type=int, default=4, help='张量并行大小')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help='GPU 内存利用率')
    
    return parser.parse_args()


def get_detailed_instruct(task_description: str, query: str) -> str:
    """构建详细指令格式"""
    return f'Instruct: {task_description}\nQuery: {query}'


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
        caption_queries.append(caption_query)
        
        # 获取当前样本的所有statements
        statements = item['statements']
        option_mapping = item['option_mapping']
        
        all_statements.extend(statements)
        statement_mappings.append({
            'start_idx': len(all_statements) - len(statements),
            'end_idx': len(all_statements),
            'mapping': option_mapping,
            'num_options': len(statements)
        })
    
    return caption_queries, all_statements, statement_mappings


def deploy_embedding_model(model_path: str, tensor_parallel_size: int, gpu_memory_utilization: float) -> LLM:
    """部署嵌入模型"""
    print(f"正在加载嵌入模型: {model_path}")
    
    llm = LLM(
        model=model_path,
        task="embed",
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        dtype="auto"
    )
    
    print("嵌入模型加载完成")
    return llm


def compute_embeddings(llm: LLM, texts: List[str]) -> torch.Tensor:
    """计算文本嵌入"""
    print(f"计算 {len(texts)} 个文本的嵌入...")
    
    outputs = llm.embed(texts)
    embeddings = torch.tensor([o.outputs.embedding for o in outputs])
    
    # 归一化嵌入向量
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    return embeddings


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


def main():
    """主函数"""
    args = parse_args()
    
    # 1. 加载statement数据
    print("加载statement数据...")
    data = load_statement_data(args.input_file)
    print(f"加载了 {len(data)} 个样本")
    
    # 2. 过滤数据，去除image和Temporal的组合
    data = filter_data(data)
    
    # 3. 准备caption和statement
    print("准备caption和statement...")
    caption_queries, statements, statement_mappings = prepare_captions_and_statements(data)
    print(f"准备了 {len(caption_queries)} 个caption查询和 {len(statements)} 个statement")
    
    # 4. 部署嵌入模型
    llm = deploy_embedding_model(args.model_path, args.tensor_parallel_size, args.gpu_memory_utilization)
    
    # 5. 计算嵌入
    print("计算caption嵌入...")
    caption_embeddings = compute_embeddings(llm, caption_queries)
    
    print("计算statement嵌入...")
    statement_embeddings = compute_embeddings(llm, statements)
    
    # 6. 找到最佳statement
    print("计算相似度并选择最佳statement...")
    predictions = find_best_statements(caption_embeddings, statement_embeddings, statement_mappings)
    
    # 7. 计算详细准确率
    pred_options = [pred[0] for pred in predictions]
    accuracy_stats = calculate_detailed_accuracy(data, pred_options)
    
    # 8. 打印详细结果
    print_detailed_results(accuracy_stats)
    
    # 9. 保存结果
    save_results(data, predictions, args.output_file, accuracy_stats)
    print(f"结果已保存到: {args.output_file}")


if __name__ == "__main__":
    main()