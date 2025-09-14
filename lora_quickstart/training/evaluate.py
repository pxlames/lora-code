"""
模型评估脚本
评估LoRA微调后的模型性能
"""

import os
import yaml
import torch
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.utils import load_chinese_instructions, generate_response


def load_config(config_path="config/config.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_finetuned_model(model_path, base_model_name="Qwen/Qwen2-0.5B"):
    """加载微调后的模型"""
    print(f"正在加载基础模型: {base_model_name}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 加载LoRA权重
    print(f"正在加载LoRA权重: {model_path}")
    model = PeftModel.from_pretrained(base_model, model_path)
    
    return model, tokenizer


def evaluate_model_performance(model, tokenizer, test_data, max_samples=50):
    """评估模型性能"""
    print(f"正在评估模型性能，使用 {min(len(test_data), max_samples)} 个样本...")
    
    results = {
        'total_samples': 0,
        'successful_generations': 0,
        'avg_length': 0,
        'generation_times': [],
        'sample_results': []
    }
    
    model.eval()
    
    for i, sample in enumerate(test_data[:max_samples]):
        if i % 10 == 0:
            print(f"处理进度: {i}/{min(len(test_data), max_samples)}")
        
        instruction = sample['instruction']
        expected_output = sample['output']
        
        # 格式化输入
        prompt = f"### 指令:\n{instruction}\n\n### 输出:\n"
        
        try:
            # 生成回复
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if start_time:
                start_time.record()
            
            inputs = tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=len(inputs.input_ids[0]) + 100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                generation_time = start_time.elapsed_time(end_time) / 1000.0  # 转换为秒
            else:
                generation_time = 0.0
            
            # 解码输出
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_output = generated_text.split("### 输出:\n")[-1].strip()
            
            results['total_samples'] += 1
            results['successful_generations'] += 1
            results['avg_length'] += len(generated_output)
            results['generation_times'].append(generation_time)
            
            # 保存样本结果
            results['sample_results'].append({
                'instruction': instruction,
                'expected': expected_output,
                'generated': generated_output,
                'generation_time': generation_time,
                'length': len(generated_output)
            })
            
        except Exception as e:
            print(f"生成失败 (样本 {i}): {e}")
            results['sample_results'].append({
                'instruction': instruction,
                'expected': expected_output,
                'generated': f"生成失败: {e}",
                'generation_time': 0.0,
                'length': 0
            })
    
    # 计算平均指标
    if results['successful_generations'] > 0:
        results['avg_length'] /= results['successful_generations']
        results['avg_generation_time'] = np.mean(results['generation_times'])
        results['success_rate'] = results['successful_generations'] / results['total_samples']
    else:
        results['avg_length'] = 0
        results['avg_generation_time'] = 0
        results['success_rate'] = 0
    
    return results


def calculate_similarity_metrics(results):
    """计算相似性指标"""
    from difflib import SequenceMatcher
    
    similarities = []
    
    for sample in results['sample_results']:
        if sample['generated'] and not sample['generated'].startswith("生成失败"):
            # 计算字符串相似度
            similarity = SequenceMatcher(None, sample['expected'], sample['generated']).ratio()
            similarities.append(similarity)
    
    if similarities:
        return {
            'avg_similarity': np.mean(similarities),
            'max_similarity': np.max(similarities),
            'min_similarity': np.min(similarities),
            'std_similarity': np.std(similarities)
        }
    else:
        return {
            'avg_similarity': 0,
            'max_similarity': 0,
            'min_similarity': 0,
            'std_similarity': 0
        }


def plot_evaluation_results(results, output_dir="outputs"):
    """绘制评估结果图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('LoRA微调模型评估结果', fontsize=16)
    
    # 1. 生成长度分布
    lengths = [sample['length'] for sample in results['sample_results'] if sample['length'] > 0]
    if lengths:
        axes[0, 0].hist(lengths, bins=20, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('生成长度分布')
        axes[0, 0].set_xlabel('生成长度')
        axes[0, 0].set_ylabel('频次')
    
    # 2. 生成时间分布
    times = [sample['generation_time'] for sample in results['sample_results'] if sample['generation_time'] > 0]
    if times:
        axes[0, 1].hist(times, bins=20, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('生成时间分布')
        axes[0, 1].set_xlabel('生成时间 (秒)')
        axes[0, 1].set_ylabel('频次')
    
    # 3. 性能指标
    metrics = ['成功率', '平均长度', '平均时间']
    values = [results['success_rate'], results['avg_length'], results['avg_generation_time']]
    
    bars = axes[1, 0].bar(metrics, values, color=['orange', 'purple', 'brown'])
    axes[1, 0].set_title('性能指标')
    axes[1, 0].set_ylabel('数值')
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.2f}', ha='center', va='bottom')
    
    # 4. 相似度分布
    similarities = []
    for sample in results['sample_results']:
        if sample['generated'] and not sample['generated'].startswith("生成失败"):
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, sample['expected'], sample['generated']).ratio()
            similarities.append(similarity)
    
    if similarities:
        axes[1, 1].hist(similarities, bins=20, alpha=0.7, color='pink')
        axes[1, 1].set_title('相似度分布')
        axes[1, 1].set_xlabel('相似度')
        axes[1, 1].set_ylabel('频次')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/evaluation_results.png", dpi=300, bbox_inches='tight')
    plt.show()


def save_detailed_results(results, output_dir="outputs"):
    """保存详细评估结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存JSON格式结果
    with open(f"{output_dir}/evaluation_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 保存文本格式结果
    with open(f"{output_dir}/evaluation_summary.txt", 'w', encoding='utf-8') as f:
        f.write("LoRA微调模型评估结果\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"总样本数: {results['total_samples']}\n")
        f.write(f"成功生成数: {results['successful_generations']}\n")
        f.write(f"成功率: {results['success_rate']:.2%}\n")
        f.write(f"平均生成长度: {results['avg_length']:.2f}\n")
        f.write(f"平均生成时间: {results['avg_generation_time']:.2f}秒\n\n")
        
        f.write("详细样本结果:\n")
        f.write("-" * 50 + "\n")
        
        for i, sample in enumerate(results['sample_results']):
            f.write(f"\n样本 {i+1}:\n")
            f.write(f"指令: {sample['instruction']}\n")
            f.write(f"期望输出: {sample['expected']}\n")
            f.write(f"生成输出: {sample['generated']}\n")
            f.write(f"生成长度: {sample['length']}\n")
            f.write(f"生成时间: {sample['generation_time']:.2f}秒\n")


def main():
    """主评估函数"""
    parser = argparse.ArgumentParser(description='LoRA微调模型评估')
    parser.add_argument('--model_path', type=str, required=True, help='微调模型路径')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='配置文件路径')
    parser.add_argument('--test_data', type=str, default='data/chinese_instructions.json', help='测试数据路径')
    parser.add_argument('--max_samples', type=int, default=50, help='最大测试样本数')
    parser.add_argument('--output_dir', type=str, default='outputs', help='输出目录')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    print("=" * 50)
    print("开始评估LoRA微调模型")
    print("=" * 50)
    
    # 加载模型
    model, tokenizer = load_finetuned_model(args.model_path)
    
    # 加载测试数据
    test_data = load_chinese_instructions(args.test_data)
    print(f"加载了 {len(test_data)} 个测试样本")
    
    # 评估模型
    results = evaluate_model_performance(model, tokenizer, test_data, args.max_samples)
    
    # 计算相似性指标
    similarity_metrics = calculate_similarity_metrics(results)
    results.update(similarity_metrics)
    
    # 打印结果
    print("\n评估结果:")
    print("=" * 30)
    print(f"总样本数: {results['total_samples']}")
    print(f"成功生成数: {results['successful_generations']}")
    print(f"成功率: {results['success_rate']:.2%}")
    print(f"平均生成长度: {results['avg_length']:.2f}")
    print(f"平均生成时间: {results['avg_generation_time']:.2f}秒")
    print(f"平均相似度: {results['avg_similarity']:.2f}")
    
    # 绘制图表
    plot_evaluation_results(results, args.output_dir)
    
    # 保存结果
    save_detailed_results(results, args.output_dir)
    
    print(f"\n评估完成！结果已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
