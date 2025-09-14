"""
LoRA微调训练脚本
基于Transformers和PEFT库实现参数高效微调
"""

import os
import yaml
import torch
import wandb
from transformers import (
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import json
import argparse
from datetime import datetime

# 添加项目根目录到路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lora_model import LoRAModel
from models.utils import (
    tokenize_function, 
    compute_metrics, 
    create_data_collator,
    load_chinese_instructions,
    split_dataset
)


def load_config(config_path="config/config.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def prepare_dataset(config, tokenizer):
    """准备训练数据集"""
    print("正在准备数据集...")
    
    # 加载数据
    data = load_chinese_instructions(config['data']['train_file'])
    
    # 限制样本数量（便于快速学习）
    if config['data'].get('max_samples'):
        data = data[:config['data']['max_samples']]
    
    # 分割训练集和验证集
    train_data, val_data = split_dataset(data, config['data']['validation_split'])
    
    print(f"训练样本数: {len(train_data)}")
    print(f"验证样本数: {len(val_data)}")
    
    # 转换为Dataset格式
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    # 分词
    def tokenize_fn(examples):
        return tokenize_function(
            examples, 
            tokenizer, 
            config['model']['max_length']
        )
    
    train_dataset = train_dataset.map(
        tokenize_fn, 
        batched=True, 
        remove_columns=train_dataset.column_names
    )
    
    val_dataset = val_dataset.map(
        tokenize_fn, 
        batched=True, 
        remove_columns=val_dataset.column_names
    )
    
    return train_dataset, val_dataset


def setup_wandb(config):
    """设置Wandb日志"""
    if config['logging']['use_wandb']:
        wandb.init(
            project=config['logging']['project_name'],
            name=config['logging']['run_name'],
            config=config
        )


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='LoRA微调训练')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, help='从检查点恢复训练')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置Wandb
    setup_wandb(config)
    
    print("=" * 50)
    print("开始LoRA微调训练")
    print("=" * 50)
    
    # 初始化LoRA模型
    lora_model = LoRAModel(args.config)
    model, tokenizer = lora_model.load_model_and_tokenizer()
    peft_model = lora_model.setup_lora()
    
    # 准备数据集
    train_dataset, val_dataset = prepare_dataset(config, tokenizer)
    
    # 创建数据整理器
    data_collator = create_data_collator(tokenizer)
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_ratio=config['training']['warmup_ratio'],
        logging_steps=config['training']['logging_steps'],
        save_steps=config['training']['save_steps'],
        eval_steps=config['training']['eval_steps'],
        save_total_limit=config['training']['save_total_limit'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        metric_for_best_model=config['training']['metric_for_best_model'],
        greater_is_better=config['training']['greater_is_better'],
        fp16=config['training']['fp16'],
        dataloader_num_workers=config['training']['dataloader_num_workers'],
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_dir=f"{config['training']['output_dir']}/logs",
        report_to="wandb" if config['logging']['use_wandb'] else None,
        remove_unused_columns=False,
    )
    
    # 创建训练器
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # 开始训练
    print("\n开始训练...")
    print(f"训练参数: {training_args}")
    
    try:
        if args.resume:
            trainer.train(resume_from_checkpoint=args.resume)
        else:
            trainer.train()
        
        # 保存最终模型
        trainer.save_model()
        tokenizer.save_pretrained(config['training']['output_dir'])
        
        print("\n训练完成！")
        print(f"模型已保存到: {config['training']['output_dir']}")
        
        # 显示训练结果
        train_results = trainer.evaluate()
        print("\n最终评估结果:")
        for key, value in train_results.items():
            print(f"{key}: {value:.4f}")
            
    except KeyboardInterrupt:
        print("\n训练被中断，正在保存当前模型...")
        trainer.save_model()
        tokenizer.save_pretrained(config['training']['output_dir'])
        print("模型已保存")
    
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        raise
    
    finally:
        if config['logging']['use_wandb']:
            wandb.finish()


def test_generation(model, tokenizer, test_prompts):
    """测试模型生成能力"""
    print("\n测试模型生成能力:")
    print("=" * 50)
    
    model.eval()
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n测试 {i+1}:")
        print(f"输入: {prompt}")
        
        # 生成回复
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"输出: {response}")


if __name__ == "__main__":
    main()
