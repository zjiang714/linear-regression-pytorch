import torch
import torch.nn as nn

# 1. 定义 NiN 块 (NiN Block)
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    """
    NiN 块的核心结构：
    一个常规卷积层 -> 两个 1x1 卷积层 (充当局部 MLP)
    """
    return nn.Sequential(
        # 核心卷积层，用于提取空间特征
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(inplace=True),
        
        # 第一个 1x1 卷积，用于通道间特征组合（相当于跨通道的跨像素全连接）
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(inplace=True),
        
        # 第二个 1x1 卷积，进一步增强非线性表达
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(inplace=True)
    )

# 2. 构建 NiN 网络
class NiN(nn.Module):
    def __init__(self, num_classes=10):
        super(NiN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1: 模拟 AlexNet 的第一层大卷积，输入通常为 3 通道 (如 224x224)
            nin_block(in_channels=3, out_channels=96, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Block 2
            nin_block(in_channels=96, out_channels=256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Block 3
            nin_block(in_channels=256, out_channels=384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(p=0.5),
            
            # Block 4: 最后一个 NiN 块的输出通道数直接等于你的分类类别数 (num_classes)
            # 这一步是 NiN 的精髓，替代了传统最后的 Linear 层
            nin_block(in_channels=384, out_channels=num_classes, kernel_size=3, strides=1, padding=1),
            
            # 全局平均池化 (Global Average Pooling)
            # 将特征图的高宽从 (H, W) 变成 (1, 1)，保留通道数不变
            nn.AdaptiveAvgPool2d((1, 1)),
            
            # 压平输出，将形状从 (Batch, num_classes, 1, 1) 变成 (Batch, num_classes)
            nn.Flatten()
        )

    def forward(self, x):
        return self.features(x)

# --- 验证模型 ---
if __name__ == "__main__":
    # 实例化模型（假设处理 CIFAR-10 或类似的 10 分类任务）
    model = NiN(num_classes=10)
    print(model)
    
    # 模拟标准 ImageNet 尺寸输入：batch_size=2, 3通道, 224x224
    X = torch.randn(2, 3, 224, 224)
    output = model(X)
    
    print("-" * 30)
    print(f"输入数据形状: {X.shape}")
    print(f"输出数据形状: {output.shape}")  # 预期输出为 [2, 10]