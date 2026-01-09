"""
NeuroTumorNet - Brain Tumor Classification using Deep Convolutional Neural Networks
===================================================================================
基于 https://github.com/h9zdev/NeuroTumorNet 的 PyTorch 实现

原始论文方法:
1. Baseline CNN: 3层卷积块 + BatchNorm + L2正则化 + Dropout
2. VGG16 Transfer Learning: 预训练VGG16 + 自定义分类头
3. 带正则化的增强CNN: BatchNorm + Dropout + L2正则化

本模块用 PyTorch 复现原始仓库的核心架构，保留其设计理念：
- 简单的3层CNN作为基线
- BatchNormalization稳定训练
- Dropout防止过拟合
- L2正则化 (通过weight_decay实现)

作者: 基于 h9zdev/NeuroTumorNet (CC BY-NC 4.0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. NeuroTumorNet Baseline CNN
# ============================================================================
class NeuroTumorNetBaseline(nn.Module):
    """
    NeuroTumorNet Baseline CNN - 原始论文的基线模型
    
    架构:
    - 3个卷积块: Conv2D -> BatchNorm -> ReLU -> MaxPool
    - 通道数: 32 -> 64 -> 128
    - 全连接层: Flatten -> Dense(128) -> Dropout -> Output
    
    基于论文 Section 3.3: Model Architecture
    """
    def __init__(self, num_classes=4, dropout_rate=0.5):
        super(NeuroTumorNetBaseline, self).__init__()
        
        # 卷积块 1: 32 filters
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # 卷积块 2: 64 filters
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # 卷积块 3: 128 filters
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # 分类头
        # 输入 224x224 -> 经过3次pool后 -> 28x28x128
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 28 * 28, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 卷积块 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # 卷积块 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # 卷积块 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # 分类
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ============================================================================
# 2. NeuroTumorNet Enhanced CNN (with Regularization)
# ============================================================================
class NeuroTumorNetEnhanced(nn.Module):
    """
    NeuroTumorNet Enhanced CNN - 带增强正则化的模型
    
    基于论文 Section 3.6: Addressing Overfitting
    - 每个卷积层后添加BatchNorm
    - 更高的Dropout率 (可配置0.3-0.7)
    - L2正则化通过optimizer的weight_decay实现
    
    架构增强:
    - 更深的全连接层
    - 多级Dropout
    """
    def __init__(self, num_classes=4, dropout_rate1=0.5, dropout_rate2=0.3):
        super(NeuroTumorNetEnhanced, self).__init__()
        
        # 卷积块 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # 卷积块 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # 卷积块 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # 增强的分类头
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_rate1)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout_rate2)
        self.fc3 = nn.Linear(128, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 卷积块
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # 增强分类头
        x = self.flatten(x)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


# ============================================================================
# 3. NeuroTumorNet Deep (5-layer CNN)
# ============================================================================
class NeuroTumorNetDeep(nn.Module):
    """
    NeuroTumorNet Deep - 更深的CNN变体
    
    5个卷积块:
    - 通道数: 32 -> 64 -> 128 -> 256 -> 512
    - 使用Global Average Pooling代替Flatten减少参数
    """
    def __init__(self, num_classes=4, dropout_rate=0.5):
        super(NeuroTumorNetDeep, self).__init__()
        
        # 卷积块序列
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


# ============================================================================
# 4. NeuroTumorNet VGG-style (类似VGG的架构)
# ============================================================================
class NeuroTumorNetVGGStyle(nn.Module):
    """
    NeuroTumorNet VGG-style - 模仿VGG的双卷积块设计
    
    参考原始仓库使用VGG16进行迁移学习的思路，
    但这里是从头训练一个VGG风格的小型网络
    
    每个stage有2个连续的3x3卷积
    """
    def __init__(self, num_classes=4, dropout_rate=0.5):
        super(NeuroTumorNetVGGStyle, self).__init__()
        
        self.features = nn.Sequential(
            # Stage 1: 2 x Conv(64)
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Stage 2: 2 x Conv(128)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Stage 3: 3 x Conv(256)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Stage 4: 3 x Conv(512)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ============================================================================
# 5. NeuroTumorNet Tiny (轻量级版本)
# ============================================================================
class NeuroTumorNetTiny(nn.Module):
    """
    NeuroTumorNet Tiny - 轻量级版本用于快速实验
    
    更少的通道数: 16 -> 32 -> 64
    使用Global Average Pooling大幅减少参数
    """
    def __init__(self, num_classes=4, dropout_rate=0.3):
        super(NeuroTumorNetTiny, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================================
# 模型工厂函数
# ============================================================================
def create_neurotumornet(variant='baseline', num_classes=4, **kwargs):
    """
    创建 NeuroTumorNet 模型
    
    Args:
        variant: 模型变体
            - 'baseline': 原始3层CNN
            - 'enhanced': 带增强正则化的CNN
            - 'deep': 5层深度CNN
            - 'vgg_style': VGG风格的CNN
            - 'tiny': 轻量级CNN
        num_classes: 输出类别数
        **kwargs: 额外参数 (如 dropout_rate)
    
    Returns:
        PyTorch模型
    """
    models = {
        'baseline': NeuroTumorNetBaseline,
        'enhanced': NeuroTumorNetEnhanced,
        'deep': NeuroTumorNetDeep,
        'vgg_style': NeuroTumorNetVGGStyle,
        'tiny': NeuroTumorNetTiny,
    }
    
    if variant not in models:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(models.keys())}")
    
    return models[variant](num_classes=num_classes, **kwargs)


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("测试 NeuroTumorNet Classifier")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    
    models_to_test = [
        ('NeuroTumorNetBaseline', 'baseline'),
        ('NeuroTumorNetEnhanced', 'enhanced'),
        ('NeuroTumorNetDeep', 'deep'),
        ('NeuroTumorNetVGGStyle', 'vgg_style'),
        ('NeuroTumorNetTiny', 'tiny'),
    ]
    
    for name, variant in models_to_test:
        print(f"\n--- {name} ---")
        model = create_neurotumornet(variant=variant, num_classes=4).to(device)
        output = model(dummy_input)
        
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"输入: {dummy_input.shape}")
        print(f"输出: {output.shape}")
        print(f"参数量: {params:.2f}M", end=" ")
        print("✅ 测试通过!")
    
    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)
