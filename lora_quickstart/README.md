# LoRA微调快速入门项目

这是一个精简的LoRA微调学习项目，基于Qwen2-0.5B模型，适合快速学习和实践LoRA技术。

## 项目特点

- 🚀 **快速上手**: 使用轻量级Qwen2-0.5B模型，训练速度快
- 📚 **中文友好**: 包含中文指令微调数据集
- 🔧 **完整流程**: 从数据准备到模型评估的完整流程
- 📊 **可视化**: 集成wandb进行训练监控
- 💡 **学习导向**: 代码注释详细，适合学习LoRA原理

## 项目结构

```
lora_quickstart/
├── requirements.txt          # 依赖包
├── README.md                # 项目说明
├── config/
│   └── config.yaml          # 配置文件
├── data/
│   ├── prepare_data.py      # 数据准备脚本
│   └── chinese_instructions.json  # 中文指令数据集
├── models/
│   ├── lora_model.py        # LoRA模型定义
│   └── utils.py             # 工具函数
├── training/
│   ├── train.py             # 训练脚本
│   └── evaluate.py          # 评估脚本
├── visualization/
│   └── plot_results.py      # 结果可视化
└── examples/
    └── demo.py              # 使用示例
```

## 快速开始

1. **安装依赖**
```bash
pip install -r requirements.txt
```

2. **准备数据**
```bash
python data/prepare_data.py
```

3. **开始训练**
```bash
python training/train.py
```

4. **评估模型**
```bash
python training/evaluate.py
```

## 学习要点

- LoRA原理和实现
- 参数高效微调技巧
- 中文指令微调最佳实践
- 模型评估和可视化

## 数据集说明

使用中文指令微调数据集，包含：
- 问答对
- 指令跟随任务
- 对话生成任务

## 技术栈

- PyTorch + Transformers
- PEFT (Parameter Efficient Fine-Tuning)
- Wandb (实验跟踪)
- 中文大模型Qwen2
