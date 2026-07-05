# 全卷积网络

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from typing import Tuple, List

# ==========================================
# 🧱 第一阶段：工业级 FCN (全卷积网络) 拓扑构建
# ==========================================

class FullyConvolutionalNet(nn.Module):
    """
    工业级全卷积网络 (FCN-32s 变体)
    基于预训练 ResNet-18 提取特征，利用转置卷积进行全分辨率上采样
    """
    def __init__(self, num_classes: int, pretrain_weights: bool = True):
        super().__init__()
        self.num_classes = num_classes
        
        # 1. 加载主干网络 (采用最新版本的 weights 参数，消除过时的 pretrained 警告)
        if pretrain_weights:
            weights = torchvision.models.ResNet18_Weights.DEFAULT
        else:
            weights = None
        resnet = torchvision.models.resnet18(weights=weights)
        
        # 2. 特征提取骨架：剥离最后两层(全局平均池化 Flatten 和全连接分类层 FC)
        # 此时输出的特征图尺寸为 [Batch, 512, H/32, W/32]
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # 3. 分类通道转换层：通过 1x1 卷积将 512 维通道数压缩到最终的目标类别数
        self.final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        
        # 4. 核心上采样层(转置卷积)：将特征图空间分辨率精准放大 32 倍，恢复到原图尺寸
        # 算法超参数由经典的 FCN-32s 论文数学推导得出
        self.transpose_conv = nn.ConvTranspose2d(
            in_channels=num_classes, 
            out_channels=num_classes,
            kernel_size=64, 
            padding=16, 
            stride=32,
            bias=False
        )
        
        # 5. 自动触发工业级权重初始化
        self._init_bilinear_weights()

    def _init_bilinear_weights(self):
        """使用双线性插值核对转置卷积进行硬核参数复制初始化(加速网络分割边缘的收敛)"""
        W = self.bilinear_kernel(self.num_classes, self.num_classes, kernel_size=64)
        self.transpose_conv.weight.data.copy_(W)
        print("⚙️ [FCN Init] 成功使用双线性插值矩阵初始化转置卷积权重。")

    @staticmethod
    def bilinear_kernel(in_channels: int, out_channels: int, kernel_size: int) -> torch.Tensor:
        """纯数学矩阵构建：生成双线性插值滤波核"""
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = (torch.arange(kernel_size).reshape(-1, 1),
              torch.arange(kernel_size).reshape(1, -1))
        filt = (1 - torch.abs(og[0] - center) / factor) * \
               (1 - torch.abs(og[1] - center) / factor)
        
        weight = torch.zeros((in_channels, out_channels, kernel_size, kernel_size))
        weight[range(in_channels), range(out_channels), :, :] = filt
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [Batch, 3, H, W] -> [Batch, 512, H/32, W/32]
        features = self.backbone(x)
        # -> [Batch, num_classes, H/32, W/32]
        class_logits = self.final_conv(features)
        # -> [Batch, num_classes, H, W] (恢复全分辨率)
        out = self.transpose_conv(class_logits)
        return out


# ==========================================
# 🎨 第二阶段：工业级图像预处理与色彩映射引擎
# ==========================================

class SemanticSegmentationProcessor:
    """负责分割任务的标准化前处理与后处理调色盘(Colormap)映射"""
    
    # 帕斯卡 PASCAL VOC2012 官方定义的 21 种分类标准的 RGB 颜色映射表
    VOC_COLORMAP = [
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128],
        [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
        [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
        [0, 192, 0], [128, 192, 0], [0, 64, 128]
    ]

    def __init__(self, device: torch.device):
        self.device = device
        # 将色彩表固化到 GPU 显存，实现矩阵级像素填色加速
        self.colormap_tensor = torch.tensor(self.VOC_COLORMAP, device=device)
        
        # 工业标准的 ImageNet 归一化参数
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess(self, pil_img: Image.Image) -> torch.Tensor:
        """输入 PIL 图像，输出符合模型前向传播边界的标准 4D 张量"""
        # [3, H, W] -> [1, 3, H, W]
        return self.transform(pil_img).unsqueeze(0).to(self.device)

    def label_to_image(self, pred_mask: torch.Tensor) -> np.ndarray:
        """把模型输出的类别索引矩阵 [H, W] 精准映射回带色彩的 RGB 渲染图 [H, W, 3]"""
        idx_matrix = pred_mask.long()
        # 利用 PyTorch 高級索引从显存色彩表中一枪映射
        rgb_tensor = self.colormap_tensor[idx_matrix, :]
        return rgb_tensor.cpu().numpy().astype(np.uint8)


# ==========================================
# 📐 第三阶段：解耦的专属像素级多分类损失函数
# ==========================================

class FullyConvolutionalLoss(nn.Module):
    """FCN 专用的二维空间像素交叉熵损失函数"""
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 对每一个像素点的多分类概率计算交叉熵，并在 H 和 W 维度上求均值
        pixel_loss = self.criterion(inputs, targets)
        return pixel_loss.mean(dim=-1).mean(dim=-1)


# ==========================================
# 🛡️ 第四阶段：工业级部署与路径自检验证入口
# ==========================================

if __name__ == '__main__':
    # 1. 自动化测试卡配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📊 正在配置工业级执行计算卡: {device}")
    
    # 2. 动态绝对路径计算(彻底解决上一阶段的 FileNotFoundError 痛点)
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    target_img_path = os.path.abspath(os.path.join(current_file_dir, '../img/catdog.jpg'))
    
    # 3. 初始化 FCN 核心网络 (PASCAL VOC 共有 21 个类别)
    fcn_model = FullyConvolutionalNet(num_classes=21, pretrain_weights=True).to(device)
    fcn_model.eval()
    
    processor = SemanticSegmentationProcessor(device=device)
    
    # 4. 执行业务流逻辑
    if os.path.exists(target_img_path):
        print(f"✅ 成功锁定本地测试图片: {target_img_path}")
        try:
            # 读取真实的猫狗图片
            raw_image = Image.open(target_img_path).convert('RGB')
            # 缩放到标准测试裁剪尺寸 (320x480)
            resized_image = raw_image.resize((480, 320))
            
            # 前处理
            input_tensor = processor.preprocess(resized_image)
            
            # 📍 真正的算法爆发执行点：FCN 网络前向像素级推理
            with torch.no_grad():
                output_logits = fcn_model(input_tensor)
            
            # 后处理：在 Channel 维(维度 1)上取最大值，获得每个像素点胜率最高的类别索引
            pred_mask = output_logits.argmax(dim=1).squeeze(0)
            
            # 将索引图翻译成彩色可视化图片矩阵
            color_mask = processor.label_to_image(pred_mask)
            
            print("\n✅ FCN 全卷积网络物理推理闭环测试成功！")
            print(f"   - 输入图像张量维度 (Shape): {input_tensor.shape} -> [Batch, RGB, H, W]")
            print(f"   - 网络原始输出维度 (Shape): {output_logits.shape} -> [Batch, 类别数, H, W]")
            print(f"   - 像素类别索引维度 (Shape): {pred_mask.shape}   -> [H, W]")
            print(f"   - 最终彩色掩膜维度 (Shape): {color_mask.shape} -> [H, W, RGB]")
            
        except Exception as e:
            print(f"❌ 业务链路运行崩溃: {str(e)}")
    else:
        print(f"⚠️ [⚠️ 警告] 未在路径 {target_img_path} 下找到测试图片。")
        print("   -> 激活虚拟数据冒烟测试。")
        # 构造一个 4D 虚拟张量进行纯数学网格单元测试
        mock_input = torch.zeros((4, 3, 320, 480), device=device)
        with torch.no_grad():
            mock_output = fcn_model(mock_input)
        print(f"✅ 虚拟前向传播测试通过，输出特征图 Shape: {mock_output.shape}")

        