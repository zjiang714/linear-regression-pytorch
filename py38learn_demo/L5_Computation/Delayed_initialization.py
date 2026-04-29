#**延后初始化的思维**是：**“我现在不知道输入是多少，等数据真正流过来的时候，你自己看着办吧！”**
import torch
from torch import nn
#以前的写法是：
class DelayedInitialization(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64),
            nn.Linear(64, 128), # 你得保证这里的 64 和上面对得上
            nn.Linear(128, 1)   # 你得保证这里的 128 和上面对得上
        )

# 引入延后初始化思想后的写法是：
class DelayedInitialization_new(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(64),  # 输入是多少？不知道，等数据来了再说
            nn.LazyLinear(128), # 输入是多少？自动对齐上一层的 64
            nn.LazyLinear(1)
        )