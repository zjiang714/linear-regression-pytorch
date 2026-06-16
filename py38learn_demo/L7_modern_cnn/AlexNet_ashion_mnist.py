import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from d2l import torch as d2l

"""2、使用框架的预定义好的层"""
class AlexNet(nn.Module):
    """AlexNet: 输入 1x224x224 → 输出 10 类 (FashionMNIST)"""
    def __init__(self):
        super().__init__()
        # ---- 卷积块：提取空间特征 ----
        # 相比 LeNet，AlexNet 用了更多的卷积层、更大的通道数，且引入了 Dropout 防止过拟合
        self.features = nn.Sequential(
            # 第一阶段：大卷积核抓取宏观粗糙特征
            # 输入 (1, 224, 224) → 输出 (96, 54, 54)
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(),
            # 汇聚层缩小宽高：输出 (96, 26, 26)
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # 第二阶段：中等卷积核提炼局部形状
            # 输入 (96, 26, 26) → 输出 (256, 26, 26)
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            # 汇聚层缩小宽高：输出 (256, 12, 12)
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # 第三阶段：连续三层卷积组合更高级、更复杂的概念
            # 输入 (256, 12, 12) → 输出 (384, 12, 12)
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            # 输入 (384, 12, 12) → 输出 (384, 12, 12)
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            # 输入 (384, 12, 12) → 输出 (256, 12, 12)
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            # 最后一层汇聚：输出 (256, 5, 5)
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        # ---- 全连接分类块：将特征映射到类别 ----
        self.flatten = nn.Flatten() # 256 * 5 * 5 = 6400 展平为一维向量
        
        self.classifier = nn.Sequential(
            # 全连接1 + Dropout（防止大网络过拟合的灵魂组件）
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            
            # 全连接2 + Dropout
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            
            # 全连接3(输出层，无激活)
            nn.Linear(4096, 10)
        )
        
        # Xavier 初始化参数
        self.apply(self._init_weights)

    """3、初始化模型参数"""
    def _init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    # 前向传播
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

net = AlexNet()


# 用随机数据过一遍网络，确认每层输出形状（由于内部使用了 nn.Sequential，我们分开打印大块）
X = torch.rand(size=(1, 1, 224, 224), dtype=torch.float32)
print("=" * 60)
print("【AlexNet 网络大块维度验证】")
print("=" * 60)
for name, layer in net.named_children():
    if name != '_init_weights':
        X = layer(X)
        print(f"{name:10s} | {str(layer.__class__.__name__):12s} | {list(X.shape)}")
print("=" * 60 + "\n")


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """在测试集上评估模型准确率"""
    if isinstance(net, nn.Module):
        net.eval()  # 切换到评估模式
        if not device:
            device = next(iter(net.parameters())).device
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型"""
    print('training on', device)
    net.to(device)

    """5、实例化sgd实例"""
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    """4、定义损失函数"""
    loss = nn.CrossEntropyLoss()

    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)

    """6、开始训练"""
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        net.train() # 切换到训练模式（启用 dropout）

        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)                                # ① 前向传播
            l = loss(y_hat, y)                            # ② 计算损失
            l.backward()                                  # ③ 反向传播
            optimizer.step()                              # ④ 梯度下降

            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]

            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))

        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))

    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


"""1、生成数据"""
DATA_ROOT = "/home/zjianglinux/PycharmProjects/py38_dep_ln_code/py38learn_demo/data"

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])

train_dataset = datasets.FashionMNIST(root=DATA_ROOT, train=True, transform=transform, download=False)
test_dataset = datasets.FashionMNIST(root=DATA_ROOT, train=False, transform=transform, download=False)

train_iter = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
test_iter = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

lr, num_epochs = 0.01, 10

train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())


d2l.plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AlexNet_training.png'))



"""
第一步,图片进厂,强行整容(Resize 预处理）【图片正在经历什么】：图片刚被你手写输入进去时，它只是个 $28 \times 28$ 像素的小方块。但 AlexNet 这个巨型工厂的流水线太粗犷了，它的第一个大卷积核根本吃不下这么小的图。于是，图片在进门的一瞬间，先被四周拉扯，强行放大拉伸了 64 倍，变成了一个 $224 \times 224$ 的大矩阵，并被扒光衣服（归一化），每个像素点都变成了 0 到 1 之间的数字。🔍 对应你的代码块（在文件最底部的加载区）：Pythontransform = transforms.Compose([
    transforms.Resize(224), # 🔴 图片在这里被强行拉伸变大
    transforms.ToTensor()   # 🔴 像素在这里变成 0~1 的数字向量
])
【为什么这样写】：不放大，后面那个 $11 \times 11$、步长为 4 的卷积核一抬脚就会直接跨过整张图，网络直接就瘫痪了。第二步：图片通电，大刀阔斧切片（Features 块的前半段）【图片正在经历什么】：整容后的图片正式被推上了履带。它迎面撞上了 96 个巨大的 $11 \times 11$ 的滤镜（第一层卷积）。这些滤镜在它身上疯狂扫过，由于步长很大（每跨一步就是 4 个像素），图片被切得稀碎。扫完之后，原图消失了，取而代之的是 96 张 $54 \times 54$ 的“线条草图”（有的只留下了衣服的左边缘，有的只留下了纽扣的亮光）。接着，这些草图经过了“最大池化”，每 $3 \times 3$ 个像素里只留下最亮的那一个。图片体积再次缩水，变成了 96 张 $26 \times 26$ 的精简图。紧接着，又迎面撞上 256 个微调滤镜和第二次池化，图片被进一步揉捏成了 256 张 $12 \times 12$ 的局部特征图。🔍 对应你的代码块（在 self.features 的前半部分）：Python# 🔴 图片在这里被切成 96 张草图
nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
nn.ReLU(),
nn.MaxPool2d(kernel_size=3, stride=2), # 🔴 这里完成第一次暴瘦

# 🔴 细节深化，变成 256 张更小的局部图
nn.Conv2d(96, 256, kernel_size=5, padding=2),
nn.ReLU(),
nn.MaxPool2d(kernel_size=3, stride=2),
【为什么这样写】：这就是大名鼎鼎的特征提取。图片不能直接拿去分类，必须先用大卷积把它的“骨架”（宏观轮廓）和“血肉”（局部细节）分别剥离出来。第三步：高级逻辑脑补，榨干最后一点多余体积（Features 块的后半段）【图片正在经历什么】：此时的图片已经看不出任何人形/衣服形了，它变成了 256 张 $12 \times 12$ 的碎片矩阵。接下来，它要经历一段黑森林——连续三层极其密集的微型卷积扫描。这三层扫描非常贼，它们故意不缩减图片的宽高，就着 12x12 的尺寸疯狂地做逻辑组合。在这层里，网络的大脑开始疯狂运转，把第二步提炼出的“袖口、领子、拉链”这些碎片在空间上拼装组合。走完这三层，经历最后一次最大池化，图片迎来了它在卷积大车间的终点：256 张只有 $5 \times 5$ 像素的终极概念图。🔍 对应你的代码块（在 self.features 的后半部分）：Python# 🔴 连续三层不瘦身、只做脑补的高级组合
nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
# 🔴 卷积车间的最后一关，压榨成 5x5
nn.MaxPool2d(kernel_size=3, stride=2)
* **【为什么这样写】**：连续卷积不加池化，是为了**让网络想得更深**。如果是毛衣，它不仅要知道有毛线，还要通过连续三层卷积意识到“毛线织成了领口，且领口连着肩膀”，这是高阶的语义逻辑。

---

### 第四步：方块被“拍扁拉直”，进入数字脑风暴（Flatten 与 Classifier 前段）
**【图片正在经历什么】**：
从上一步出来的数据，形状是一个 $256 \times 5 \times 5$ 的三维小方块。
在进入最终决策室前，它被一块叫 `Flatten` 的大板子**“啪”的一声拍成了一根平平整整、拥有 6400 个数字的超长一维向量**。
这根 6400 维的细长数字链，立刻被拉进了拥有 4096 个神经元的超级大脑（全连接层）里。每一个数字都通过密密麻麻的传导线和全连接节点进行复杂的加权运算。在训练时，为了防止大家作弊死记硬背，还会有 50% 的传导线被随机掐断（Dropout）。



* **🔍 对应你的代码块**（中间的过渡与分类器前段）：
    ```python
    self.flatten = nn.Flatten() # 🔴 3D方块在这里被拍扁拉直成一根线
    
    # 🔴 6400个特征进入超级大脑进行综合思考，并随机掐断线（Dropout）
    nn.Linear(256 * 5 * 5, 4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),
【为什么这样写】：前面的卷积层都是“各管各的区域”（局部特征）。到了这里，必须把全图所有的信息混在一起，做一次全局大总管式的综合大推导。第五步：终极宣判，得分最高的胜出（Classifier 尾段）【图片正在经历什么】：经过 4096 维大脑的两轮疯狂洗礼，关于这张图的所有想法，最终被逼着灌进最后 10 个出口（输出通道）。这 10 个出口对应着 10 个衣服箱子（T恤、裤子、皮鞋等）。数据从这 10 个出口喷射出来，变成了 10 个具体的得分数字（比如：[-1.2, 0.5, 12.8, -3.1, ...]）。系统定睛一看：发现第三个位置（代表毛衣）的得分是最高的 12.8！于是大喇叭广播：“报告老板，这图识别出来了，它是一件毛衣！” 图片的一生至此画上完美句号。🔍 对应你的代码块（分类器的最后一行）：Pythonnn.Linear(4096, 10) # 🔴 4096个想法收束成10个箱子的得分
* **【为什么这样写】**：这是网络的退场口。不管中间想得多复杂，最后必须落实到具体的分类数量上，有多少个类，最后一层就必须吐出多少个得分。

---

### 💡 乱入提问：那你在代码中段写的那个 `for name, layer in net.named_children():` 是干嘛的？
既然你懂了图片识别的物理顺序，你就能彻底明白那段测试代码的用意了：

```python
# 这一段，就是我们在图片“整容进厂”后，在它身上贴了三个“追踪器”
for name, layer in net.named_children():
    X = layer(X) # 让图片（或假数据）流过这个大车间
    print(...)   # 🔴 在图片刚走完这个大车间时，立刻拦截它，量一下它当前的尺寸并打印出来！
"""