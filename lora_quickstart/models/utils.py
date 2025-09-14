"""
工具函数
包含数据处理、评估指标等实用函数
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DataCollatorForLanguageModeling
import json
import random


def create_data_collator(tokenizer, mlm=False):
    """创建数据整理器"""
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=mlm,
        pad_to_multiple_of=8
    )


def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    
    # 获取预测结果
    predictions = np.argmax(predictions, axis=2)
    
    # 移除padding标签
    true_predictions = []
    true_labels = []
    
    for prediction, label in zip(predictions, labels):
        for (p, l) in zip(prediction, label):
            if l != -100:
                true_predictions.append(p)
                true_labels.append(l)
    
    # 计算指标
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, true_predictions, average='weighted'
    )
    accuracy = accuracy_score(true_labels, true_predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def format_instruction(instruction, input_text="", output=""):
    """格式化指令数据"""
    if input_text:
        return f"### 指令:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 输出:\n{output}"
    else:
        return f"### 指令:\n{instruction}\n\n### 输出:\n{output}"


def tokenize_function(examples, tokenizer, max_length=512):
    """分词函数"""
    # 组合输入和输出
    texts = []
    for i in range(len(examples['instruction'])):
        text = format_instruction(
            examples['instruction'][i],
            examples.get('input', [''])[i] if 'input' in examples else '',
            examples['output'][i]
        )
        texts.append(text)
    
    # 分词
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # 设置标签（与输入相同，用于语言建模）
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized


def load_chinese_instructions(file_path, max_samples=None):
    """加载中文指令数据集"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    return data


def create_sample_dataset():
    """创建示例数据集"""
    sample_data = [
        {
            "instruction": "请解释什么是人工智能",
            "input": "",
            "output": "人工智能（AI）是指让机器模拟人类智能行为的技术。它包括机器学习、深度学习、自然语言处理等领域，能够执行推理、学习、感知等任务。"
        },
        {
            "instruction": "写一首关于春天的诗",
            "input": "",
            "output": "春风拂面花满园，\n绿柳垂丝舞翩跹。\n鸟语花香人自醉，\n万物复苏展新颜。"
        },
        {
            "instruction": "翻译以下英文句子",
            "input": "Hello, how are you today?",
            "output": "你好，你今天怎么样？"
        },
        {
            "instruction": "计算以下数学题",
            "input": "15 + 27 = ?",
            "output": "15 + 27 = 42"
        },
        {
            "instruction": "解释Python中的列表推导式",
            "input": "",
            "output": "列表推导式是Python中创建列表的简洁语法。格式为：[表达式 for 元素 in 可迭代对象 if 条件]。例如：[x**2 for x in range(10) if x % 2 == 0] 会生成0到9中偶数的平方。"
        }
    ]
    
    return sample_data


def save_dataset(data, file_path):
    """保存数据集到文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def split_dataset(data, validation_split=0.1):
    """分割训练集和验证集"""
    random.shuffle(data)
    split_idx = int(len(data) * (1 - validation_split))
    
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    return train_data, val_data


def generate_response(model, tokenizer, prompt, max_length=200, temperature=0.7):
    """生成回复"""
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


if __name__ == "__main__":
    # 测试工具函数
    sample_data = create_sample_dataset()
    print("示例数据集:")
    for i, item in enumerate(sample_data[:2]):
        print(f"\n样本 {i+1}:")
        print(f"指令: {item['instruction']}")
        print(f"输出: {item['output']}")
