import random
from pyexpat import features

from d2l import torch as d2l
import matplotlib.pyplot as plt
import torch
from scipy.special.cython_special import y1


# 1、生成数据
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))      # normal生成一个矩阵
    y = torch.matmul(X, w) + b                          # matmul单纯的数学公式
    y1 = y + torch.normal(0, 0.01, y.shape)             # 添加
    print("生成的大X是：",X)
    print("生成的y是：", y)
    print("加了噪声之后的y是：",y1)
    print("生成的y1.reshape是:",y1.reshape(-1,1))
    return X, y1.reshape((-1, 1))                       # reshape是改变矩阵的形状，具体可参考 Typora 的线性回归从0开始



true_w = torch.tensor([2.0, -3.4])                 # w权重有可能是一个向量。
true_b = 4.2

trian_features, train_labels = synthetic_data(true_w, true_b, 10)
print('最终trian_features的数据是：',trian_features)
print('最终train_labels的数据是',train_labels)



# 2、定义一个data_iter函数来读取小批量，该函数接收批量大小、特征矩阵和标签向量作为输入，生成大小为batch_size的小批量
def data_iter(batch_size, train_features, train_labels):
    """
    核心步骤拆解
    获取总量 (num_examples)：
    代码检测到你有 10 组特征，打印 num_examples的值是：10。

    创建索引 (indices)：
    生成一个列表 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]。每个数字对应你那 10 条数据的“行号”。
    洗牌 (random.shuffle)：
    这一步非常关键！它把顺序打乱，比如变成 [3, 0, 9, 1, 7, 4, 8, 2, 5, 6]。目的是让模型训练时不被数据的原始顺序干扰。

    分块切片 (range(0, 10, 15))：
    这是最核心的逻辑。因为 batch_size=15 大于你的数据总量 10，所以：

        循环只会运行 1 次。
        indices[0 : min(0+15, 10)] 实际上就是抓取了索引 0 到 10 的全部乱序编号。
        反之，如果num_example>batch_size,则会按照每次定好的batch_size来进行执行，
        1. 从“跳过”的角度看（循环步长）
            在代码 range(0, 10, 15) 中，batch_size 确实充当了步长（Step）：
            如果 batch_size = 3，循环会从索引 0 开始，跳到 3，再跳到 6，再跳到 9。
            它告诉程序：“这一批我已经处理完了，请跳过这 n 个，去处理下一批。”

        2. 从“容量”的角度看（样本数量）
            batch_size 更重要的身份是**“每一批次包含的样本数”**：
            它是你每次喂给模型“吃”的数据量。
            模型在更新一次参数之前，会先看完这 batch_size 个样本并计算它们的平均误差。
            产出数据 (yield)：
            根据这 10 个乱序编号，从原始的 train_features 和 train_labels 中把对应的数据抽出来。
    :param batch_size:
    :param train_features:
    :param train_labels:
    :return:
    """
    num_examples = len(train_features)
    # print('num_examples的值是：',num_examples)
    indices = list(range(num_examples))
    # print('indices的值是：',indices)
    # 这些样本是随即读取的，没有特定顺序
    # random.shuffle(indices)
    # 每一次从0开始，到num_example,跳过batch_size个大小
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        yield train_features[batch_indices], train_labels[batch_indices]

batch_size = 10

for X, y in data_iter(batch_size, trian_features, train_labels):
    print(X, '\n', y, '\n')
    break

