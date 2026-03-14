import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt # 导入绘图库


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型（定义见第3章）"""
    # 创建可视化动画器
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                            legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        # 训练一个周期
        train_metrics = d2l.train_epoch_ch3(net, train_iter, loss, updater)
        # 在测试集上评估精度
        test_acc = d2l.evaluate_accuracy(net, test_iter)
        # 将结果添加到动画器中
        animator.add(epoch + 1, train_metrics + (test_acc,))

    train_loss, train_acc = train_metrics
    # 断言语句，确保逻辑没有明显错误
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);

loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10

# 确保使用的是 ch3 的训练函数
try:
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    plt.show() # 关键：显示图像
except AttributeError:
    print("如果提示找不到 train_ch3，请参考我上一个回答手动定义该函数。")