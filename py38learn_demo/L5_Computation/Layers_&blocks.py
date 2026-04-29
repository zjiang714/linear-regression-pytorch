#  块和层的写法，
#  **这种“块”思维的强大之处在于：** 当你设计一个非常复杂的网络（比如有 100 层）时，你不需要写 100 行 `nn.Linear`，
#  你只需要定义一个好用的“块”，然后像复印一样通过循环把它们叠起来即可。
#  还有方向传播的代码块
# 块和层的理解：块就是一个带有记忆的公共函数，前向传播函数放在这个块里面，每一层或者每一个其他的函数在调用这个块的时候，会自动向前传递out

import torch
from torch import nn


class MyDeepNetwork(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        # 定义 4 个线性层（这就是我们要用的 4 个零件）
        self.layer1 = nn.Linear(in_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, out_size)

        # 定义激活函数
        self.relu = nn.ReLU()

    # 这就是“前向传播算法”的本体
    def forward(self, x):
        print(f"输入初始形状: {x.shape}")

        # 第一层传播
        x = self.layer1(x)
        x = self.relu(x)
        print(f"过完第 1 层后的形状: {x.shape}")

        # 第二层传播
        x = self.layer2(x)
        x = self.relu(x)
        print(f"过完第 2 层后的形状: {x.shape}")

        # 第三层传播
        x = self.layer3(x)
        x = self.relu(x)
        print(f"过完第 3 层后的形状: {x.shape}")

        # 第四层（输出层）
        # 注意：最后一层通常不加 ReLU，除非你有特殊需求
        x = self.layer4(x)
        print(f"最终输出形状: {x.shape}")

        return x


# --- 模拟运行 ---

# 假设输入特征是 20 个（比如房价预测的 20 个特征），隐藏层 64 个神经元，最后输出 1 个房价结果
model = MyDeepNetwork(in_size=20, hidden_size=64, out_size=1)

# 模拟 8 条数据同时进入网络（Batch Size = 8）
input_data = torch.randn(8, 20)

# 只要调用 model(input_data)，PyTorch 就会自动执行 forward 函数
output = model(input_data)

import torch
from torch import nn




# --- 第一部分：定义“块” (每个块里有 2 层) ---
class MyTwoLayerBlock(nn.Module):
    def __init__(self, units):
        super().__init__()
        # 定义 2 个线性层
        self.layer1 = nn.Linear(units, units)
        self.layer2 = nn.Linear(units, units)
        self.relu = nn.ReLU()

    def forward(self, X):
        # 这里的逻辑：层1 -> 激活 -> 层2 -> 激活
        out = self.relu(self.layer1(X))
        out = self.relu(self.layer2(out))
        return out


# --- 第二部分：使用“块”来组装 4 层模型 ---
# 我们想要 4 层，只需要堆叠 2 个 MyTwoLayerBlock 即可
num_features = 64

net = nn.Sequential(
    # 第 1-2 层：第一个块
    MyTwoLayerBlock(num_features),

    # 第 3-4 层：第二个块
    MyTwoLayerBlock(num_features),

    # 最后的输出层（严格算的话，现在总共 5 层了）
    nn.Linear(num_features, 1)
)

# --- 测试 ---
X = torch.rand(1, num_features)
output = net(X)
print(f"输入形状: {X.shape}")
print(f"模型输出值: {output.item()}")

