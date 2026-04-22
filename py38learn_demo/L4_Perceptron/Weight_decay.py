# 权重衰退法：
#   核心思想：在损失函数里加“罚款”，让权重尽量变小。
#   方法：它的数学实现方式，就是给损失函数加上一个权重平方和的惩罚项。这个惩罚项在数学上就是 L2​ 范数的平方。所以它们在逻辑上是等价的。

import torch
from torch import nn
from d2l import torch as d2l

# 1、先生成一些真实数据
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

# 2、定义模型并初始化模型参数
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

# 3、权重惩罚项
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2

# 4、开始训练
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加了L2范数惩罚项，
            # 广播机制使l2_penalty(w)成为一个长度为batch_size的向量
            l = loss(net(X), y) + lambd * l2_penalty(w)  # 权重衰退的直接用法，lambd就是权重的具体值，
                                                         # 通过修改 lambd 的大小来告诉模型：“我有多在意权重的大小”
                                                         # w 的更新是一个“自驱动”的过程
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是：', torch.norm(w).item())

train(lambd=10)


