import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

# 使用深度学习框架来简洁的实现 线性回归模型，生成数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# 调用框架中现有的API来读取数据
