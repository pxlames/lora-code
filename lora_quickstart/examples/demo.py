"""
LoRA微调演示脚本
展示如何使用微调后的模型进行推理
"""

import os
import sys
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.utils import format_instruction


def load_model_and_tokenizer(model_path, base_model_name="Qwen/Qwen2-0.5B"):
    """加载微调后的模型和分词器"""
    print(f"正在加载模型: {base_model_name}")
    
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
    model = PeftModel.from_pretrained(base_model, model_path)
    
    print("模型加载完成！")
    return model, tokenizer


def generate_response(model, tokenizer, instruction, input_text="", max_length=200, temperature=0.7):
    """生成回复"""
    # 格式化输入
    prompt = format_instruction(instruction, input_text)
    
    # 分词
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 生成回复
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_p=0.9,
            top_k=50
        )
    
    # 解码输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取生成的部分
    generated_text = response.split("### 输出:\n")[-1].strip()
    
    return generated_text


def interactive_demo(model, tokenizer):
    """交互式演示"""
    print("\n" + "="*60)
    print("LoRA微调模型交互式演示")
    print("="*60)
    print("输入 'quit' 或 'exit' 退出演示")
    print("输入 'help' 查看帮助信息")
    print("="*60)
    
    while True:
        try:
            # 获取用户输入
            instruction = input("\n请输入指令: ").strip()
            
            if instruction.lower() in ['quit', 'exit', '退出']:
                print("感谢使用！再见！")
                break
            
            if instruction.lower() in ['help', '帮助']:
                print_help()
                continue
            
            if not instruction:
                print("请输入有效的指令")
                continue
            
            # 可选输入
            input_text = input("请输入额外信息 (可选，直接回车跳过): ").strip()
            
            # 生成回复
            print("\n正在生成回复...")
            response = generate_response(model, tokenizer, instruction, input_text)
            
            print(f"\n生成的回复:\n{response}")
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\n演示被中断，再见！")
            break
        except Exception as e:
            print(f"\n生成过程中出现错误: {e}")
            print("请重试或输入 'help' 查看帮助")


def print_help():
    """打印帮助信息"""
    help_text = """
帮助信息:
---------
1. 基本用法:
   - 直接输入指令，如: "请解释什么是人工智能"
   - 可以添加额外信息，如: "翻译以下句子" + "Hello World"

2. 示例指令:
   - 问答: "什么是机器学习？"
   - 创作: "写一首关于春天的诗"
   - 翻译: "将'Hello'翻译成中文"
   - 编程: "用Python写一个排序函数"
   - 数学: "计算 2+2 等于多少"

3. 命令:
   - help: 显示此帮助信息
   - quit/exit: 退出演示

4. 提示:
   - 指令越具体，生成效果越好
   - 可以尝试不同的表达方式
   - 如果生成效果不理想，可以重新表述问题
    """
    print(help_text)


def batch_demo(model, tokenizer, test_cases):
    """批量演示"""
    print("\n" + "="*60)
    print("LoRA微调模型批量演示")
    print("="*60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试案例 {i}:")
        print(f"指令: {test_case['instruction']}")
        if test_case.get('input'):
            print(f"输入: {test_case['input']}")
        
        # 生成回复
        response = generate_response(
            model, 
            tokenizer, 
            test_case['instruction'], 
            test_case.get('input', '')
        )
        
        print(f"生成回复: {response}")
        print("-" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LoRA微调模型演示')
    parser.add_argument('--model_path', type=str, required=True, help='微调模型路径')
    parser.add_argument('--base_model', type=str, default='Qwen/Qwen2-0.5B', help='基础模型名称')
    parser.add_argument('--mode', type=str, choices=['interactive', 'batch'], default='interactive', 
                       help='演示模式: interactive(交互式) 或 batch(批量)')
    
    args = parser.parse_args()
    
    # 检查模型路径
    if not os.path.exists(args.model_path):
        print(f"错误: 模型路径不存在: {args.model_path}")
        print("请先运行训练脚本生成模型")
        return
    
    # 加载模型
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.base_model)
    
    if args.mode == 'interactive':
        # 交互式演示
        interactive_demo(model, tokenizer)
    
    elif args.mode == 'batch':
        # 批量演示
        test_cases = [
            {
                "instruction": "请解释什么是人工智能",
                "input": ""
            },
            {
                "instruction": "写一首关于春天的诗",
                "input": ""
            },
            {
                "instruction": "翻译以下句子",
                "input": "Hello, how are you?"
            },
            {
                "instruction": "用Python写一个计算斐波那契数列的函数",
                "input": ""
            },
            {
                "instruction": "计算以下数学题",
                "input": "15 + 27 = ?"
            }
        ]
        
        batch_demo(model, tokenizer, test_cases)


if __name__ == "__main__":
    main()
