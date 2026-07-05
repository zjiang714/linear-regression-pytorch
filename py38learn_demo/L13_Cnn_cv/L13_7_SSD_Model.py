# 单发多框检测（SSD）
# SSD算法的核心：多尺度特征提取、锚框预测、扁平化拼接、自定义多任务损失函数、以及前向推理

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any

# ==========================================
# 🧱 第一阶段：SSD 核心多尺度特征提取与预测模块
# ==========================================

class SSDPredictor(nn.Module):
    """工业级预测头：支持动态类别数与锚框数配置"""
    def __init__(self, in_channels: int, num_anchors: int, num_classes: int):
        super().__init__()
        # 类别预测层：输出通道数为 锚框数 * (类别数 + Background)
        self.cls_conv = nn.Conv2d(in_channels, num_anchors * (num_classes + 1), kernel_size=3, padding=1)
        # 边界框回归层：输出通道数为 锚框数 * 4 (即偏移量 [dx, dy, dw, dh])
        self.bbox_conv = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cls_conv(x), self.bbox_conv(x)


class DownSampleBlock(nn.Module):
    """标准下采样模块，用于构建多尺度特征金字塔"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ==========================================
# 📊 第二阶段：工业级全连接/多尺度特征拼接算子
# ==========================================

class SSDFeatureAggregator:
    """特征聚合器：将不同尺度的特征图预测结果展平并完美拼接"""
    @staticmethod
    def flatten_prediction(pred: torch.Tensor) -> torch.Tensor:
        # 将通道维移到最后：[Batch, Channel, Height, Width] -> [Batch, Height, Width, Channel]
        # 随后展平成二维矩阵：[Batch, Height * Width * Channel]
        return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

    @classmethod
    def concat_predictions(cls, preds: List[torch.Tensor]) -> torch.Tensor:
        # 对金字塔所有层的所有预测框在 Dimension 1 上进行硬核拼接
        return torch.cat([cls.flatten_prediction(p) for p in preds], dim=1)


# ==========================================
# 🚀 第三阶段：TinySSD 工业级完整网络拓扑架构
# ==========================================

class TinySSD(nn.Module):
    def __init__(self, num_classes: int, num_anchors: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # 1. 主干特征金字塔多尺度骨架构建 (5个不同分辨率的特征层)
        self.backbone_stage0 = nn.Sequential(
            DownSampleBlock(3, 16),
            DownSampleBlock(16, 32),
            DownSampleBlock(32, 64)
        )
        self.stage1 = DownSampleBlock(64, 128)
        self.stage2 = DownSampleBlock(128, 128)
        self.stage3 = DownSampleBlock(128, 128)
        self.stage4 = nn.AdaptiveMaxPool2d((1, 1)) # 终极全局汇聚层 [Batch, 128, 1, 1]

        # 2. 挂载 5 个多尺度预测头
        in_channels_list = [64, 128, 128, 128, 128]
        self.predictors = nn.ModuleList([
            SSDPredictor(in_channels, num_anchors, num_classes)
            for in_channels in in_channels_list
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cls_preds, bbox_preds = [], []
        
        # 提取各个 Stage 的多尺度特征图
        x = self.backbone_stage0(x)
        c0, b0 = self.predictors[0](x); cls_preds.append(c0); bbox_preds.append(b0)
        
        x = self.stage1(x)
        c1, b1 = self.predictors[1](x); cls_preds.append(c1); bbox_preds.append(b1)
        
        x = self.stage2(x)
        c2, b2 = self.predictors[2](x); cls_preds.append(c2); bbox_preds.append(b2)
        
        x = self.stage3(x)
        c3, b3 = self.predictors[3](x); cls_preds.append(c3); bbox_preds.append(b3)
        
        x = self.stage4(x)
        c4, b4 = self.predictors[4](x); cls_preds.append(c4); bbox_preds.append(b4)

        # 聚合所有尺度的预测数据
        concat_cls = SSDFeatureAggregator.concat_predictions(cls_preds)
        concat_bbox = SSDFeatureAggregator.concat_predictions(bbox_preds)
        
        # 重塑分类预测形状，方便后续计算 CrossEntropy
        # [Batch, 总锚框数, 类别数 + 1]
        concat_cls = concat_cls.reshape(concat_cls.shape[0], -1, self.num_classes + 1)
        
        return concat_cls, concat_bbox


# ==========================================
# 📐 第四阶段：多任务损失函数处理器（Multi-Task Loss）
# ==========================================

class SSDMultitaskLoss(nn.Module):
    """解耦的 SSD 损失计算引擎：分类交叉熵 + 边界框掩膜控制 L1 损失"""
    def __init__(self):
        super().__init__()
        self.cls_criterion = nn.CrossEntropyLoss(reduction='none')
        self.bbox_criterion = nn.L1Loss(reduction='none')

    def forward(self, cls_preds: torch.Tensor, cls_labels: torch.Tensor, 
                bbox_preds: torch.Tensor, bbox_labels: torch.Tensor, 
                bbox_masks: torch.Tensor) -> torch.Tensor:
        
        batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
        
        # 1. 计算分类损失
        cls_loss_flat = self.cls_criterion(cls_preds.reshape(-1, num_classes), cls_labels.reshape(-1))
        cls_loss = cls_loss_flat.reshape(batch_size, -1).mean(dim=1)
        
        # 2. 计算边界框回归损失（利用 bbox_masks 过滤背景框，只对正样本计入梯度）
        filtered_preds = bbox_preds * bbox_masks
        filtered_labels = bbox_labels * bbox_masks
        bbox_loss = self.bbox_criterion(filtered_preds, filtered_labels).mean(dim=1)
        
        return cls_loss + bbox_loss


# ==========================================
# 🛡️ 第五阶段：工业级推理验证主入口
# ==========================================

if __name__ == '__main__':
    # 模拟真实部署流水线
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📊 正在配置工业级执行计算卡: {device}")
    
    try:
        # 初始化模型 (单类别检测，例如：只检测香蕉)
        ssd_model = TinySSD(num_classes=1, num_anchors=4).to(device)
        ssd_model.eval()
        
        # 模拟线上推理：注入一个 Batch 大小的全零模拟图像矩阵
        mock_input = torch.zeros((16, 3, 256, 256), device=device) # [Batch_size, RGB, H, W]
        
        with torch.no_grad():
            class_predictions, bounding_box_predictions = ssd_model(mock_input)
            
        print("\n✅ SSD模型前向传播单元测试通过！")
        print(f"   - 分类预测张量分布 (Shape): {class_predictions.shape}  -> [Batch, 锚框总数, 类别+1]")
        print(f"   - 边界框回归张量分布 (Shape): {bounding_box_predictions.shape} -> [Batch, 锚框总数 * 4]")
        
    except Exception as e:
        print(f"❌ 运行崩溃，抛出异常: {str(e)}")





