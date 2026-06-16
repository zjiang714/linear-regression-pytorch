import torch
import torch.nn as nn

# =====================================================================
# 1. 核心工业积木块：BottleneckBlock（专注于深层网络的瓶颈结构）
# =====================================================================
class BottleneckBlock(nn.Module):
    expansion = 4  # 最终输出通道是中间计算通道的 4 倍

    def __init__(self, in_channels, mid_channels, stride=1):
        """
        Args:
            in_channels: 真正输入这个积木块的通道数
            mid_channels: 被 1x1 压缩后的中间核心计算通道数
        """
        super(BottleneckBlock, self).__init__()
        out_channels = mid_channels * self.expansion # 最终输出的通道数
        
        # 1.1 主干特征提取分支（三明治压缩结构）
        # 第一层：1x1 卷积压缩通道（降维）
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        # 第二层：3x3 卷积（负责真正的空间特征提取，可在此处通过 stride 降低特征图分辨率）
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        # 第三层：1x1 卷积重新放大通道（升维）
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # 1.2 快捷旁路分支（Shortcut / Identity）：判断是否需要购买 1x1 变换层
        self.downsample = None
        # 如果步长不为 1（分辨率变了），或者输入的总通道和最终输出的总通道对不上（通道变了）
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        # 趟过主干的三明治加工厂
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        # 检查旁路分支是否需要做维度对齐
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差相加并激活
        out += identity
        out = self.relu(out)
        return out


# =====================================================================
# 2. 纯 Bottleneck 组装的 ResNet 骨干网络（以经典 ResNet-50 为基准）
# =====================================================================
class IndustryResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super(IndustryResNet50, self).__init__()
        
        # 记录当前管道内的数据通道流向，刚进 Stage1 的时候是 64 维
        self.current_in_channels = 64  

        # 【第一道工序：大门安检（前置特征提取）】
        # 输入 3 通道图片，快速转换并抽取到 64 通道，分辨率减半（stride=2）
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 用最大池化再次把分辨率减半
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 【第二道工序：核心四大 Stage 残差积木群】
        # layers_list = [3, 4, 6, 3] 是 ResNet-50 的标准骨架配置
        self.stage1 = self._make_stage(mid_channels=64,  block_num=3, stride=1)
        self.stage2 = self._make_stage(mid_channels=128, block_num=4, stride=2)
        self.stage3 = self._make_stage(mid_channels=256, block_num=6, stride=2)
        self.stage4 = self._make_stage(mid_channels=512, block_num=3, stride=2)

        # 【第三道工序：后端输出全连接层】
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 汇聚全图特征
        self.fc = nn.Linear(512 * 4, num_classes)     # 512 * expansion(4) = 2048 维

    def _make_stage(self, mid_channels, block_num, stride):
        """
        流水线车间：负责把一个个 Bottleneck 积木拼接成一个大的 Stage
        """
        layers = []
        
        # 1. 每个 Stage 的首个 Block：通常伴随着分辨率或通道的剧烈变化，单独建立
        layers.append(BottleneckBlock(self.current_in_channels, mid_channels, stride))
        
        # 此时通道发生了 4 倍膨胀，更新全局通道计数器，供下一个 Block 认清输入
        self.current_in_channels = mid_channels * 4

        # 2. 该 Stage 剩余的 Block：维度已经完美对齐，只需要规规矩矩堆叠，步长全部为 1
        for _ in range(1, block_num):
            layers.append(BottleneckBlock(self.current_in_channels, mid_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 1. 初始安检：图片进入模型
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        # 2. 横穿四大车间
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # 3. 提炼输出
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # 展平矩阵，方便喂给全连接层
        x = self.fc(x)
        return x


# =====================================================================
# 3. 运行测试验证
# =====================================================================
if __name__ == '__main__':
    # 模拟真实图片输入 (Batch=2, 3通道RGB, 尺寸224x224)
    fake_batch_img = torch.randn(2, 3, 224, 224)
    
    # 购买一台工业标准的 ResNet-50 机器，分类别定为 10 类（比如做猫狗花草等10分类）
    model = IndustryResNet50(num_classes=10)
    model.eval() # 切换到评估模式
    
    # 前向推理
    output = model(fake_batch_img)
    
    print("====== 纯 Bottleneck 工业级网络运行成功 ======")
    print(f"输入真实数据形状: {fake_batch_img.shape}")
    print(f"最终输出分类预测形状: {output.shape} ── (2个样本, 10类得分)")