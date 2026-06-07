import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from d2l import torch as d2l


"""2、使用框架的预定义好的层"""
class LeNet5(nn.Module):
    """LeNet-5: 输入 1x28x28 → 输出 10 类"""
    def __init__(self):
        super().__init__()
        # ---- 卷积块：提取空间特征 ----
        # 卷积：用可学习的卷积核扫描图像，提取局部模式（边缘、纹理等）
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)   # (1,28,28) → (6,28,28)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)       # 池化（汇聚层）：下采样，降低分辨率、减少计算量
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)             # (6,14,14) → (16,10,10)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)       # (16,10,10) → (16,5,5)
        # ---- 全连接分类块：将特征映射到类别 ----
        self.flatten = nn.Flatten()                              # 16*5*5 = 400 展平为一维向量
        self.fc1 = nn.Linear(16 * 5 * 5, 120)                    # 400 → 120  全连接1
        self.fc2 = nn.Linear(120, 84)                             # 120 → 84   全连接2
        self.fc3 = nn.Linear(84, 10)                              # 84 → 10    全连接3(输出层，无激活)
        # Xavier 初始化：让每层输出的方差 ≈ 输入的方差，缓解梯度消失/爆炸
        self.apply(self._init_weights)

    """3、初始化模型参数"""
    def _init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    #前向传播
    def forward(self, x):
        # 卷积块
        x = self.conv1(x); x = torch.sigmoid(x)
        x = self.pool1(x)                                         # 池化层就是汇聚层
        x = self.conv2(x); x = torch.sigmoid(x)
        x = self.pool2(x)
        # 全连接块
        x = self.flatten(x)
        x = self.fc1(x); x = torch.sigmoid(x)
        x = self.fc2(x); x = torch.sigmoid(x)
        x = self.fc3(x)
        return x

net = LeNet5()


# 用随机数据过一遍网络，确认每层输出形状符合预期
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for name, layer in net.named_children():
    X = layer(X)
    print(f"{name:8s} | {str(layer.__class__.__name__):12s} | {list(X.shape)}")


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """在测试集上评估模型准确率"""
    if isinstance(net, nn.Module):
        net.eval()  # 切换到评估模式：禁用 dropout、固定 batch norm
        if not device:
            device = next(iter(net.parameters())).device  # 自动获取模型所在设备
    # Accumulator(2): 累加 2 个值 → (正确预测数, 总样本数)
    metric = d2l.Accumulator(2)
    with torch.no_grad():         # 评估时不计算梯度（省显存、加速）
        for X, y in data_iter:    # 遍历测试集的每个 batch
            X, y = X.to(device), y.to(device)            # 数据搬到 GPU
            metric.add(d2l.accuracy(net(X), y), y.numel())  # 累加正确数 + 总样本数
    return metric[0] / metric[1]  # 准确率 = 总正确数 / 总样本数



def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型"""
    
    # Xavier 初始化在 LeNet5.__init__ 中已完成（self.apply(_init_weights)）

    print('training on', device)
    net.to(device)       # 把整个网络的参数搬到 GPU

    """5、实例化sgd实例"""
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)  # SGD: 沿着梯度反方向更新参数
    """4、定义损失函数"""
    loss = nn.CrossEntropyLoss()  # 交叉熵 = LogSoftmax + NLLLoss，适合多分类

    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)

    """6、开始训练"""
    for epoch in range(num_epochs):                        # 遍历完整数据集 num_epochs 轮
        # Accumulator(3): 累加 (总损失, 正确数, 总样本数)
        metric = d2l.Accumulator(3)
        net.train()       # 切换到训练模式（启用 dropout 等）

        for i, (X, y) in enumerate(train_iter):           # 逐 batch 取数据
            timer.start()
            optimizer.zero_grad()                         # 清空上一步的梯度（否则梯度会累加）
            X, y = X.to(device), y.to(device)             # 数据搬到 GPU
            y_hat = net(X)                                # ① 前向传播：模型预测
            l = loss(y_hat, y)                            # ② 计算损失（预测值 vs 真实值）
            l.backward()                                  # ③ 反向传播：计算各层参数的梯度
            optimizer.step()                              # ④ 梯度下降：沿梯度反方向更新参数

            with torch.no_grad():                         # 统计指标时不参与计算图（省显存）
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]               # 平均损失
            train_acc = metric[1] / metric[2]              # 训练准确率

            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))

        test_acc = evaluate_accuracy_gpu(net, test_iter)  # 每轮结束，在测试集上评估
        animator.add(epoch + 1, (None, None, test_acc))

    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


"""1、生成数据"""
DATA_ROOT = "/home/zjianglinux/PycharmProjects/py38_dep_ln_code/py38learn_demo/data"
# ToTensor(): PIL 图像 → 形状 (C,H,W) 的 tensor，像素值缩放到 [0,1]
transform = transforms.Compose([transforms.ToTensor()])
# download=False：使用本地已下载的数据集，避免重复下载
train_dataset = datasets.FashionMNIST(root=DATA_ROOT, train=True, transform=transform, download=False)
test_dataset = datasets.FashionMNIST(root=DATA_ROOT, train=False, transform=transform, download=False)
# DataLoader: 自动将数据集切分成 batch，shuffle=True 打乱训练顺序（提高泛化）
train_iter = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
test_iter = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

lr, num_epochs = 0.9, 10
# d2l.try_gpu(): 优先用 GPU，没有 GPU 则回退到 CPU
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
d2l.plt.savefig('LeNet-5_training.png')  # 保存训练曲线图
