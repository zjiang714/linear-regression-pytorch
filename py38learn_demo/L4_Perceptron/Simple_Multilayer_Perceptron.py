import torch
from torch import nn
from d2l import torch as d2l

from py38learn_demo.L3_Regression.SoftMax_Regression import train_ch3
from py38learn_demo.L4_Perceptron.Multilayer_Perceptron_From_Zero import batch_size

# 1、生成数据
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 2、定义模型
net = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(28 * 28, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10),
                    )
# 3、初始化模型参数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)



# 4、定义损失函数
loss = nn.CrossEntropyLoss()

# 5、实例化SGD实例
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

# 6、定义超参数
batch_size, lr, num_epochs = 256, 0.1, 10
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
