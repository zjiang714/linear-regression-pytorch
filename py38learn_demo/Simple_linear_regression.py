import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn


# 使用深度学习框架来简洁的实现 线性回归模型，生成数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# 1、调用框架中现有的API来读取数据
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)                         # 把19行已经生成的features, labels做成一个list,传到TensorDataset里面
    return data.DataLoader(dataset, batch_size, shuffle=is_train)      # dataloader这个函数就是每次从dataset里面随机挑选batch_size个样本出来

batch_size = 10
data_iter = load_array((features, labels), batch_size)
print(next(iter(data_iter)))

# 2、使用框架的预定义好的层，nn是神经网络的缩写。
net = torch.nn.Sequential(nn.Linear(2, 1))   # list of layers

# 3、初始化模型参数
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill(0,0.01)

# 4、定义均方误差，使用MSELoss，也称为平方范数
loss = nn.MSELoss()

# 5、实例化sgd实例
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 6、开始训练
def start_training():
    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features),labels)
        print(f'epoch {epoch}, loss{l:f}')




