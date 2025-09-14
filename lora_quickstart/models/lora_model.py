"""
LoRA微调模型实现
基于PEFT库实现参数高效微调
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import yaml
import os


class LoRAModel:
    """LoRA微调模型类"""
    
    def __init__(self, config_path="config/config.yaml"):
        """初始化LoRA模型
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
    def _load_config(self, config_path):
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def load_model_and_tokenizer(self):
        """加载预训练模型和分词器"""
        print("正在加载模型和分词器...")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['name'],
            trust_remote_code=self.config['model']['trust_remote_code']
        )
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['name'],
            trust_remote_code=self.config['model']['trust_remote_code'],
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print(f"模型加载完成: {self.config['model']['name']}")
        return self.model, self.tokenizer
    
    def setup_lora(self):
        """设置LoRA配置"""
        print("正在设置LoRA配置...")
        
        # LoRA配置
        lora_config = LoraConfig(
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['lora_alpha'],
            target_modules=self.config['lora']['target_modules'],
            lora_dropout=self.config['lora']['lora_dropout'],
            bias=self.config['lora']['bias'],
            task_type=TaskType.CAUSAL_LM,
        )
        
        # 应用LoRA到模型
        self.peft_model = get_peft_model(self.model, lora_config)
        
        # 打印可训练参数
        self.print_trainable_parameters()
        
        return self.peft_model
    
    def print_trainable_parameters(self):
        """打印可训练参数信息"""
        trainable_params = 0
        all_param = 0
        
        for _, param in self.peft_model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print(f"可训练参数: {trainable_params:,} || 总参数: {all_param:,} || 可训练参数占比: {100 * trainable_params / all_param:.2f}%")
    
    def get_model_info(self):
        """获取模型信息"""
        if self.peft_model is None:
            return "模型未初始化"
        
        info = {
            "base_model": self.config['model']['name'],
            "lora_rank": self.config['lora']['r'],
            "lora_alpha": self.config['lora']['lora_alpha'],
            "target_modules": self.config['lora']['target_modules'],
            "max_length": self.config['model']['max_length']
        }
        return info


def mark_only_lora_as_trainable(model):
    """只训练LoRA参数，冻结其他参数"""
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 解冻LoRA参数
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
            print(f"解冻参数: {name}")


if __name__ == "__main__":
    # 测试LoRA模型
    lora_model = LoRAModel()
    model, tokenizer = lora_model.load_model_and_tokenizer()
    peft_model = lora_model.setup_lora()
    
    print("\n模型信息:")
    print(lora_model.get_model_info())
