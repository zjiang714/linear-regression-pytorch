import torch
import torch.nn as nn

# ==========================================
# 1. 定义 BasicBlock（用于 ResNet-18 / ResNet-34）
# ==========================================
class BasicBlock(nn.Module):
    expansion = 1  # 输出通道膨胀系数

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        # 两个独立的 3x3 卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample  # 接收外部传入的快捷分支对齐层

    def forward(self, x):
        identity = x

        # 主干特征提取分支
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 快捷旁路分支：若维度或分辨率不对，通过外部传入的 1x1 卷积对齐
        if self.downsample is not None:
            identity = self.downsample(x)

        # 核心：残差融合，然后进行最后一次激活
        out += identity
        out = self.relu(out)

        return out


# ==========================================
# 2. 定义 BottleneckBlock（用于 ResNet-50 / 101 / 152）
# ==========================================
class BottleneckBlock(nn.Module):
    expansion = 4  # 最终输出通道是中间计算通道的 4 倍

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        
        # 1x1 卷积压缩通道（降维）
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3 卷积进行核心特征提取
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1 卷积将通道数放大 4 倍（升维）
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        # 主干特征提取分支（三明治结构）
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 快捷旁路分支对齐
        if self.downsample is not None:
            identity = self.downsample(x)

        # 核心：残差融合
        out += identity
        out = self.relu(out)

        return out


# ==========================================
# 3. 统一封装的 ResNet 大类（包含自动动态组装逻辑）
# ==========================================
class GeneralResNet(nn.Module):
    def __init__(self, block_type, layers_list, num_classes=1000):
        """
        Args:
            block_type: 传入上面的 BasicBlock 或 BottleneckBlock 基础块类
            layers_list: 每个 Stage 包含几个 Block。例如 ResNet-18 为 [2, 2, 2, 2]，ResNet-50 为 [3, 4, 6, 3]
            num_classes: 分类任务的类别总数
        """
        super(GeneralResNet, self).__init__()
        self.in_channels = 64  # 网络初始输入的特征图通道数

        # 1. 初始输入输入头（通常对图像进行前期的快速降采样）
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 2. 核心四大 Stage 层的搭建
        # 每一个 Stage 的第一个 Block 负责动态判断并组装 downsample 快捷层
        self.stage1 = self._make_layer(block_type, out_channels=64,  blocks_num=layers_list[0], stride=1)
        self.stage2 = self._make_layer(block_type, out_channels=128, blocks_num=layers_list[1], stride=2)
        self.stage3 = self._make_layer(block_type, out_channels=256, blocks_num=layers_list[2], stride=2)
        self.stage4 = self._make_layer(block_type, out_channels=512, blocks_num=layers_list[3], stride=2)

        # 3. 后端分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block_type.expansion, num_classes)

    def _make_layer(self, block_type, out_channels, blocks_num, stride=1):
        """
        🚀 动态组装快捷层（downsample）的核心业务函数
        """
        downsample = None
        
        # 判断条件：如果步长不为 1 (代表分辨率减半)，或者输入通道与残差输出通道对不上
        if stride != 1 or self.in_channels != out_channels * block_type.expansion:
            # 瞧！当初你想贴的代码，在大型网络里就是作为高阶逻辑被统一动态生成出来⬇️
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block_type.expansion, 
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block_type.expansion)
            )

        layers = []
        # 创建该 Stage 的第一个 Block（只有它需要对齐维度，所以传 downsample）
        layers.append(block_type(self.in_channels, out_channels, stride, downsample))
        
        # 更新追踪：当前这一组 Block 吐出来的输出通道，作为下几个 Block 的输入通道
        self.in_channels = out_channels * block_type.expansion

        # 该 Stage 剩余的 Block 维度已经完全对齐，不再需要 downsample 传参
        for _ in range(1, blocks_num):
            layers.append(block_type(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 1. 输入数据预处理
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        # 2. 依次趟过四个阶段的残差积木堆
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # 3. 分类输出
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ==========================================
# 4. 验证测试主函数
# ==========================================
if __name__ == '__main__':
    # 模拟输入一张 3 通道的 RGB 图片（Batch=1, Size=224x224）
    fake_img = torch.randn(1, 3, 224, 224)

    print("--- 1. 测试组装标准的 ResNet-18 ---")
    resnet18 = GeneralResNet(BasicBlock, [2, 2, 2, 2], num_classes=1000)
    out18 = resnet18(fake_img)
    print(f"ResNet-18 最终输出特征图 shape: {out18.shape}\n")

    print("--- 2. 测试组装深层的 ResNet-50 ---")
    resnet50 = GeneralResNet(BottleneckBlock, [3, 4, 6, 3], num_classes=1000)
    out50 = resnet50(fake_img)
    print(f"ResNet-50 最终输出特征图 shape: {out50.shape}")