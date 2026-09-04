# PyTorch Deep Learning Practice & Demos (py38learn_demo)

本项目是一个基于 **PyTorch** 的深度学习算法实践与代码实现仓库。包含了从基础机器学习模型到卷积神经网络（CNN）、循环神经网络（RNN）、注意力机制（Attention）以及自然语言处理（NLP）模型的完整代码练习与工程实践。

---

## 📂 项目目录结构 (Directory Structure)

```text
linear-regression-pytorch/
├── py38learn_demo/
│   ├── L3_Regression/          # 线性回归、权重衰退 (L2正则化) 与暂退法 (Dropout)
│   ├── L4_Perceptron/          # 多层感知机 (MLP) 与 GPU 训练效率优化
│   ├── L5_Computation/         # 深度学习计算 (层与块、参数管理、GPU训练优化)
│   ├── L6_cnn/                 # 基础卷积神经网络 (LeNet 等)
│   ├── L7_modern_cnn/          # 现代卷积神经网络 (AlexNet, VGG, NiN, ResNet 等)
│   ├── L8_rnn/                 # 基础循环神经网络 (RNN 序列模型)
│   ├── L9_mordern_rnn/         # 现代循环神经网络 (GRU, LSTM 等)
│   ├── L10_Self_Attention/     # 自注意力机制 (Self-Attention) 与 Transformer 架构基础
│   ├── L13_Cnn_cv/             # 计算机视觉应用 (图像增广、目标检测、语义分割等)
│   ├── L14_NLP_Pre-training/   # NLP 预训练 (Word2Vec, BERT 预训练实战)
│   ├── L15_NLP_application/    # NLP 实际应用 (文本分类、情感分析、序列标注等)
│   └── Learn_pre_NLP.txt       # NLP 预备知识与学习笔记
├── main.py                     # 主程序入口/测试脚本
└── README.md                   # 项目说明文档



🚀 核心涵盖内容 (Key Features)

    基础模型与正则化 (L3-L4)：

        线性回归 (Linear Regression) 与 Softmax 回归。

        权重衰退 (Weight Decay) 与 Dropout (暂退法) 防过拟合代码实现。

        初次使用 GPU (cuda) 训练及多设备代码执行效率优化。

    计算机视觉 CNN 演进 (L6-L7, L13)：

        从基础卷积层、池化层到经典 LeNet 的搭建。

        深入现代 CNN 架构（AlexNet, VGG, ResNet 等）的 PyTorch 手动实现。

        图像增强与计算机视觉下游任务尝试。

    序列模型与 NLP 演进 (L8-L10, L14-L15)：

        RNN、GRU 与双向 LSTM 模型构建。

        自注意力机制 (Self-Attention) 核心组件解析。

        Word2Vec / BERT 等 NLP 预训练流程与文本分类应用。

🛠️ 环境准备 (Prerequisites)

    Python: 3.8+

    PyTorch: 1.10+ (推荐配置 CUDA 支持以运行 GPU 加速)

    Dependencies: torch, torchvision, torchaudio, numpy, matplotlib

安装依赖：
Bash

pip install torch torchvision numpy matplotlib

💻 使用说明 (Usage)

    克隆仓库：
    Bash

    git clone [https://github.com/your-username/linear-regression-pytorch.git](https://github.com/your-username/linear-regression-pytorch.git)
    cd linear-regression-pytorch/py38learn_demo

    运行特定章节代码（例如运行 GPU 优化或 CNN 模块）：
    Bash

    python L4_Perceptron/xxx.py
    # 或运行根目录测试
    python main.py

📌 学习记录与 Commit 规范说明

本仓库记录了从 CPU 搭建基础模型到 GPU 训练优化的完整调试过程：

    L3_Regression: 提交关于权重衰退和暂退法的实现代码。

    L4_Perceptron: 提交初次使用 GPU 进行训练并优化执行效率的代码。

    L5_Computation: 提交 GPU 训练流程与模型结构的优化代码。
