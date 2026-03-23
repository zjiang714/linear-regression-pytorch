import torch
from torch import nn
from d2l import torch as d2l

from py38learn_demo.L1_Regression.SoftMax_Regression import train_ch3

# 下载图片数据集
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)



# 初始化模型参数，每一层记录一个权重和偏置向量。
# 实现一个具有单隐藏层的多层感知机，它包含256个隐藏单元
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)            # 第一层输入的权重 w
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))     # 第一个输入的偏差 b

W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)           # 第二层的输入就是第一层的输出，最后得出权重 w
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))     # 第二层的输入就是第一层的输出，最后得出权重 b

params = [W1, b1, W2, b2]

# 激活函数
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

# 模型
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1)                                             # 这里“@”代表矩阵乘法，X代表输入的一个矩阵
    return (H@W2 + b2)

# 损失函数
loss = nn.CrossEntropyLoss(reduction='none')


# 开始训练
num_epochs, lr = 10, 0.3
updater = torch.optim.SGD(params, lr=lr)
train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)


