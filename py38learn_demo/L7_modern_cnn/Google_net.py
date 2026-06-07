import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 重点 1: 定义 Inception 块 (并行多分支结构)
# ==========================================
class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        """
        Args:
            in_channels: 输入通道数
            c1: 分支 1 (1x1 卷积) 的输出通道数
            c2: 分支 2 (1x1 卷积 + 3x3 卷积) 的输出通道数，传入元组 (c2_1x1, c2_3x3)
            c3: 分支 3 (1x1 卷积 + 5x5 卷积) 的输出通道数，传入元组 (c3_1x1, c3_5x5)
            c4: 分支 4 (3x3 最大池化 + 1x1 卷积) 的输出通道数
        """
        super(Inception, self).__init__()
        
        # 分支 1：纯 1x1 卷积
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        
        # 分支 2：1x1 卷积降维 + 3x3 卷积（padding=1 保证宽高不变）
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        
        # 分支 3：1x1 卷积降维 + 5x5 卷积（padding=2 保证宽高不变）
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        
        # 分支 4：3x3 最大池化（padding=1且stride=1宽高不变） + 1x1 卷积
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        # 并行计算 4 个分支
        out1 = F.relu(self.p1_1(x))
        out2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        out3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        out4 = F.relu(self.p4_2(self.p4_1(x)))
        
        # 在通道（Dimension 1）上将 4 个分支的结果拼接起来
        return torch.cat((out1, out2, out3, out4), dim=1)


# ==========================================
# 2. 构建 GoogLeNet 骨干网络
# ==========================================
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogLeNet, self).__init__()
        
        # Block 1: 基础卷积层（降低分辨率）
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Block 2: 基础卷积层
        self.b2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Block 3: 开始堆叠 Inception 块
        # 输入 192 -> 拼接后输出 256 / 输入 256 -> 拼接后输出 480
        self.b3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),    # 3a
            Inception(256, 128, (128, 192), (32, 96), 64),  # 3b
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Block 4: 更多的 Inception 块
        self.b4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),   # 4a
            Inception(512, 160, (112, 224), (24, 64), 64),  # 4b
            Inception(512, 128, (128, 256), (24, 64), 64),  # 4c
            Inception(512, 112, (144, 288), (32, 64), 64),  # 4d
            Inception(528, 256, (160, 320), (32, 128), 128),# 4e
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Block 5: 最后的 Inception 块与全局平均池化
        self.b5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),# 5a
            Inception(832, 384, (192, 384), (48, 128), 128),# 5b
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.classifier(x)
        return x

# --- 验证模型 ---
if __name__ == "__main__":
    model = GoogLeNet(num_classes=10)
    # 模拟输入 224x224 图像
    X = torch.randn(2, 3, 224, 224)
    output = model(X)
    print(f"输入形状: {X.shape}")
    print(f"输出形状: {output.shape}")