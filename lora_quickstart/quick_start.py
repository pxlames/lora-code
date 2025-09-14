#!/usr/bin/env python3
"""
LoRA微调快速开始脚本
一键运行完整的LoRA微调流程
"""

import os
import sys
import subprocess
import argparse
import yaml
from datetime import datetime


def run_command(command, description):
    """运行命令并显示进度"""
    print(f"\n{'='*50}")
    print(f"正在执行: {description}")
    print(f"命令: {command}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ 执行成功!")
        if result.stdout:
            print("输出:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 执行失败: {e}")
        if e.stderr:
            print("错误信息:", e.stderr)
        return False


def check_dependencies():
    """检查依赖是否安装"""
    print("检查依赖包...")
    
    required_packages = [
        'torch', 'transformers', 'datasets', 'peft', 
        'accelerate', 'wandb', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("所有依赖包已安装!")
    return True


def prepare_environment():
    """准备环境"""
    print("准备环境...")
    
    # 创建必要的目录
    directories = ['outputs', 'outputs/logs', 'data']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ 创建目录: {directory}")
    
    # 检查配置文件
    if not os.path.exists('config/config.yaml'):
        print("❌ 配置文件不存在: config/config.yaml")
        return False
    
    print("✅ 环境准备完成!")
    return True


def prepare_data():
    """准备数据"""
    print("准备训练数据...")
    
    if not os.path.exists('data/chinese_instructions.json'):
        print("数据文件不存在，正在创建...")
        success = run_command(
            "python data/prepare_data.py",
            "创建中文指令数据集"
        )
        if not success:
            return False
    else:
        print("✅ 数据文件已存在")
    
    return True


def train_model(config_file="config/config.yaml"):
    """训练模型"""
    print("开始训练LoRA模型...")
    
    # 检查是否有GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ 检测到GPU: {torch.cuda.get_device_name()}")
        else:
            print("⚠️  未检测到GPU，将使用CPU训练（速度较慢）")
    except ImportError:
        print("⚠️  无法检测GPU状态")
    
    success = run_command(
        f"python training/train.py --config {config_file}",
        "训练LoRA模型"
    )
    
    return success


def evaluate_model(model_path="outputs"):
    """评估模型"""
    print("评估模型性能...")
    
    success = run_command(
        f"python training/evaluate.py --model_path {model_path} --max_samples 20",
        "评估模型性能"
    )
    
    return success


def visualize_results():
    """可视化结果"""
    print("生成可视化图表...")
    
    success = run_command(
        "python visualization/plot_results.py",
        "生成可视化图表"
    )
    
    return success


def demo_model(model_path="outputs"):
    """演示模型"""
    print("演示微调后的模型...")
    
    success = run_command(
        f"python examples/demo.py --model_path {model_path} --mode batch",
        "演示模型效果"
    )
    
    return success


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LoRA微调快速开始')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='配置文件路径')
    parser.add_argument('--skip-train', action='store_true', help='跳过训练步骤')
    parser.add_argument('--skip-eval', action='store_true', help='跳过评估步骤')
    parser.add_argument('--skip-viz', action='store_true', help='跳过可视化步骤')
    parser.add_argument('--skip-demo', action='store_true', help='跳过演示步骤')
    parser.add_argument('--model-path', type=str, default='outputs', help='模型路径')
    
    args = parser.parse_args()
    
    print("🚀 LoRA微调快速开始")
    print("="*60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 步骤1: 检查依赖
    if not check_dependencies():
        print("\n❌ 依赖检查失败，请先安装依赖包")
        return
    
    # 步骤2: 准备环境
    if not prepare_environment():
        print("\n❌ 环境准备失败")
        return
    
    # 步骤3: 准备数据
    if not prepare_data():
        print("\n❌ 数据准备失败")
        return
    
    # 步骤4: 训练模型
    if not args.skip_train:
        if not train_model(args.config):
            print("\n❌ 模型训练失败")
            return
    else:
        print("⏭️  跳过训练步骤")
    
    # 步骤5: 评估模型
    if not args.skip_eval:
        if not evaluate_model(args.model_path):
            print("\n❌ 模型评估失败")
            return
    else:
        print("⏭️  跳过评估步骤")
    
    # 步骤6: 可视化结果
    if not args.skip_viz:
        if not visualize_results():
            print("\n❌ 结果可视化失败")
            return
    else:
        print("⏭️  跳过可视化步骤")
    
    # 步骤7: 演示模型
    if not args.skip_demo:
        if not demo_model(args.model_path):
            print("\n❌ 模型演示失败")
            return
    else:
        print("⏭️  跳过演示步骤")
    
    print("\n" + "="*60)
    print("🎉 LoRA微调流程完成!")
    print("="*60)
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n生成的文件:")
    print("- outputs/ (训练输出)")
    print("- outputs/evaluation_results.json (评估结果)")
    print("- outputs/*.png (可视化图表)")
    print("\n下一步:")
    print("1. 查看 outputs/ 目录中的结果")
    print("2. 运行 python examples/demo.py --model_path outputs 进行交互式演示")
    print("3. 尝试修改 config/config.yaml 中的参数进行实验")


if __name__ == "__main__":
    main()
