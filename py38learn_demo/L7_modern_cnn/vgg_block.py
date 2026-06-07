# 该函数有三个参数，
# 分别对应于卷积层的数量num_convs、输入通道的数量in_channels 和输出通道的数量out_channels.



import torch
import torch.nn as nn

# 1. 定义标准的 VGG 块
def vgg_block(num_convs, in_channels, out_channels):
    """
    Args:
        num_convs (int): 块内卷积层的数量
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
    """
    layers = []
    for i in range(num_convs):
        # 第一层卷积负责改变通道数，后续的卷积通道数保持不变
        layers.append(nn.Conv2d(in_channels if i == 0 else out_channels, 
                                out_channels, 
                                kernel_size=3, 
                                padding=1))
        layers.append(nn.ReLU(inplace=True))
    
    # 每个 VGG 块最后连接一个 2x2 的最大池化层，高宽减半
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    
    return nn.Sequential(*layers)


# 2. 组合成 VGG 网络
class TinyVGG(nn.Module):
    def __init__(self, block_arch, num_classes=10):
        """
        Args:
            block_arch (list of tuples): 网络架构配置，形如 [(num_convs, out_channels), ...]
            num_classes (int): 分类类别数（例如 CIFAR-10 设为 10）
        """
        super(TinyVGG, self).__init__()
        
        self.features = nn.Sequential()
        
        # 动态构建卷积特征提取层
        in_channels = 3  # 初始输入通道数（如 RGB 图像）
        for i, (num_convs, out_channels) in enumerate(block_arch):
            self.features.add_module(f"vgg_block_{i+1}", 
                                     vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels  # 下一个块的输入通道是当前块的输出通道
            
        # 3. 定义全连接层（分类器）
        # 假设输入图像大小在经过多次池化后降为了 7x7 (例如标准 ImageNet 输入 224x224 经 5次池化后是 7x7)
        # 如果你使用的是 32x32 的图片（如 CIFAR），请根据实际池化次数调整处的 7*7
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_channels * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- 验证模型 ---
if __name__ == "__main__":
    # 定义网络架构：(卷积层数, 输出通道数)
    # 这里我们定义 4 个 VGG 块，模拟一个迷你版的 VGG
    conv_arch = [(1, 64), (1, 128), (2, 256), (2, 512)]
    
    # 实例化模型
    model = TinyVGG(conv_arch, num_classes=10)
    print(model)
    
    # 模拟一个输入：batch_size=2, 3通道, 224x224大小的图像
    X = torch.randn(2, 3, 224, 224)
    output = model(X)
    print(f"\n输入形状: {X.shape}")
    print(f"输出形状: {output.shape} (符合 10 个类别的预测)")
