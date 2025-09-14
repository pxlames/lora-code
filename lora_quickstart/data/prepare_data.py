"""
数据准备脚本
创建和准备中文指令微调数据集
"""

import json
import os
from models.utils import create_sample_dataset, save_dataset, split_dataset


def create_chinese_instruction_dataset():
    """创建中文指令微调数据集"""
    
    # 扩展的示例数据集，包含多种类型的指令
    dataset = [
        # 问答类
        {
            "instruction": "请解释什么是机器学习",
            "input": "",
            "output": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习并做出预测或决策，而无需被明确编程。主要包括监督学习、无监督学习和强化学习三种类型。"
        },
        {
            "instruction": "什么是深度学习？",
            "input": "",
            "output": "深度学习是机器学习的一个子领域，使用多层神经网络来模拟人脑的工作方式。它能够自动学习数据的复杂特征，在图像识别、自然语言处理等领域取得了突破性进展。"
        },
        {
            "instruction": "解释什么是神经网络",
            "input": "",
            "output": "神经网络是受生物神经网络启发的计算模型，由相互连接的节点（神经元）组成。每个连接都有权重，网络通过调整这些权重来学习输入和输出之间的关系。"
        },
        
        # 创作类
        {
            "instruction": "写一首关于秋天的诗",
            "input": "",
            "output": "秋风萧瑟叶飘零，\n金黄满地映斜阳。\n硕果累累枝头挂，\n丰收喜悦满心房。"
        },
        {
            "instruction": "写一个关于友谊的短故事",
            "input": "",
            "output": "小明和小红是同桌，他们一起学习、一起玩耍。有一天小明生病了，小红主动帮他补习功课，还给他带水果。从此他们的友谊更加深厚，成为了最好的朋友。"
        },
        {
            "instruction": "创作一个科幻故事的开头",
            "input": "",
            "output": "2157年，人类已经掌握了星际旅行技术。宇航员李博士驾驶着最新型的量子飞船，准备前往距离地球4.3光年的比邻星。然而，在穿越虫洞时，飞船突然失去了控制..."
        },
        
        # 翻译类
        {
            "instruction": "将以下中文翻译成英文",
            "input": "今天天气很好，我们去公园散步吧。",
            "output": "The weather is very nice today, let's go for a walk in the park."
        },
        {
            "instruction": "将以下英文翻译成中文",
            "input": "Artificial intelligence will change the world.",
            "output": "人工智能将改变世界。"
        },
        {
            "instruction": "翻译这个句子",
            "input": "Hello, nice to meet you!",
            "output": "你好，很高兴认识你！"
        },
        
        # 数学计算类
        {
            "instruction": "计算以下数学题",
            "input": "25 × 4 + 10 = ?",
            "output": "25 × 4 + 10 = 100 + 10 = 110"
        },
        {
            "instruction": "解这个方程",
            "input": "2x + 5 = 13",
            "output": "2x + 5 = 13\n2x = 13 - 5\n2x = 8\nx = 4"
        },
        {
            "instruction": "计算圆的面积",
            "input": "半径r = 5cm",
            "output": "圆的面积 = π × r² = π × 5² = π × 25 = 78.54 cm²"
        },
        
        # 编程类
        {
            "instruction": "用Python写一个计算斐波那契数列的函数",
            "input": "",
            "output": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\n# 使用示例\nfor i in range(10):\n    print(fibonacci(i))"
        },
        {
            "instruction": "解释Python中的装饰器",
            "input": "",
            "output": "装饰器是Python中的一种设计模式，它允许在不修改原函数代码的情况下，为函数添加新功能。装饰器本质上是一个返回函数的高阶函数。"
        },
        {
            "instruction": "写一个简单的排序算法",
            "input": "",
            "output": "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr"
        },
        
        # 生活常识类
        {
            "instruction": "如何保持健康的生活方式？",
            "input": "",
            "output": "保持健康的生活方式包括：1）规律作息，保证充足睡眠；2）均衡饮食，多吃蔬菜水果；3）适量运动，每周至少150分钟中等强度运动；4）保持良好心态，学会减压；5）定期体检，预防疾病。"
        },
        {
            "instruction": "如何学习一门新语言？",
            "input": "",
            "output": "学习新语言的方法：1）制定学习计划，每天坚持练习；2）多听多说，提高语感；3）阅读原版书籍，扩大词汇量；4）找语言伙伴练习对话；5）观看原版影视作品；6）使用语言学习APP辅助学习。"
        },
        {
            "instruction": "如何提高工作效率？",
            "input": "",
            "output": "提高工作效率的技巧：1）制定优先级清单，先做重要紧急的事；2）使用番茄工作法，专注25分钟后休息5分钟；3）减少干扰，关闭不必要的通知；4）学会说'不'，避免过度承诺；5）定期回顾和调整工作方法。"
        },
        
        # 逻辑推理类
        {
            "instruction": "解决这个逻辑题",
            "input": "如果所有的鸟都会飞，企鹅是鸟，那么企鹅会飞吗？",
            "output": "这是一个逻辑推理题。根据给定的前提：1）所有的鸟都会飞；2）企鹅是鸟。按照逻辑推理，企鹅应该会飞。但实际上我们知道企鹅不会飞，这说明前提'所有的鸟都会飞'是错误的。"
        },
        {
            "instruction": "分析这个因果关系",
            "input": "下雨和带伞的关系",
            "output": "下雨和带伞之间存在因果关系：下雨是因，带伞是果。人们为了避免被雨淋湿，所以选择带伞。这是一种预防性的因果关系，体现了人类对环境的适应和应对。"
        },
        
        # 创意写作类
        {
            "instruction": "写一个产品广告文案",
            "input": "智能手表",
            "output": "【智能手表，健康生活新选择】\n24小时健康监测，让健康触手可及\n运动数据实时记录，科学指导健身\n智能提醒，不错过重要时刻\n时尚外观，商务休闲两相宜\n选择智能手表，开启智慧生活！"
        },
        {
            "instruction": "写一封感谢信",
            "input": "感谢老师的教导",
            "output": "亲爱的老师：\n\n感谢您这段时间的悉心教导和耐心指导。您的专业知识和敬业精神让我受益匪浅，不仅学到了知识，更学会了做人的道理。\n\n您的教诲将伴随我一生，再次感谢您的付出！\n\n此致\n敬礼！\n\n学生：XXX"
        }
    ]
    
    return dataset


def main():
    """主函数"""
    print("正在准备中文指令微调数据集...")
    
    # 创建数据集
    dataset = create_chinese_instruction_dataset()
    print(f"创建了 {len(dataset)} 个样本")
    
    # 分割训练集和验证集
    train_data, val_data = split_dataset(dataset, validation_split=0.2)
    print(f"训练集: {len(train_data)} 个样本")
    print(f"验证集: {len(val_data)} 个样本")
    
    # 保存数据集
    os.makedirs("data", exist_ok=True)
    
    # 保存完整数据集
    save_dataset(dataset, "data/chinese_instructions.json")
    
    # 保存训练集和验证集
    save_dataset(train_data, "data/train_data.json")
    save_dataset(val_data, "data/val_data.json")
    
    print("数据集准备完成！")
    print("文件保存位置:")
    print("- data/chinese_instructions.json (完整数据集)")
    print("- data/train_data.json (训练集)")
    print("- data/val_data.json (验证集)")
    
    # 显示几个示例
    print("\n数据集示例:")
    for i, item in enumerate(dataset[:3]):
        print(f"\n示例 {i+1}:")
        print(f"指令: {item['instruction']}")
        if item['input']:
            print(f"输入: {item['input']}")
        print(f"输出: {item['output']}")


if __name__ == "__main__":
    main()
