import torch
import torch.nn as nn


class BatchNormLeNet5(nn.Module):
    def __init__(self):
        super(BatchNormLeNet5, self).__init__()
        # 1. 卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        # BatchNorm2d 的 num_features 对应卷积的 out_channels
        self.bn1 = nn.BatchNorm2d(num_features=6) 
        
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        
        # 2. 全连接层
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        # BatchNorm1d 的 num_features 对应全连接的 out_features
        self.bn3 = nn.BatchNorm1d(num_features=120)
        
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.bn4 = nn.BatchNorm1d(num_features=84)
        
        self.fc3 = nn.Linear(in_features=84, out_features=10)
        
        # 3. 公用层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.act = nn.ReLU() # 现代网络加了BN后通常换用ReLU以获得更佳效果

    def forward(self, x):
        # 严格执行：卷积 -> 规范化 -> 激活 -> 池化
        x = self.pool(self.act(self.bn1(self.conv1(x))))
        x = self.pool(self.act(self.bn2(self.conv2(x))))
        
        x = x.view(-1, 16 * 5 * 5)
        
        # 全连接 -> 规范化 -> 激活
        x = self.act(self.bn3(self.fc1(x)))
        x = self.act(self.bn4(self.fc2(x)))
        x = self.fc3(x)
        return x