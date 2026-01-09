"""
UNet2D Classifier - 基于 LucaLumetti/UNetTransplant 的 2D UNet 图像分类器
参考: https://github.com/LucaLumetti/UNetTransplant

将原始 3D UNet 分割模型改造为 2D 图像分类模型：
1. 使用 2D 卷积 (is3d=False)
2. 仅使用 Encoder 作为特征提取器
3. 添加 Global Average Pooling + 分类头

适配你的脑肿瘤分类任务 (4类: glioma, meningioma, no_tumor, pituitary)
"""

from functools import partial
from typing import Optional, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
#  Building Blocks (从 UNetTransplant 仓库移植并适配 2D 分类)
# =============================================================================

def number_of_features_per_level(init_channel_number: int, num_levels: int) -> List[int]:
    """计算每层的特征通道数"""
    return [init_channel_number * (2 ** k) for k in range(num_levels)]


def create_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    order: str,
    num_groups: int,
    padding: int,
    dropout_prob: float,
    is3d: bool = False,
):
    """
    创建卷积层组合，支持灵活的层序配置
    order: 定义层顺序, e.g. 'gcr' -> GroupNorm + Conv + ReLU
        'c' - Conv
        'r' - ReLU
        'l' - LeakyReLU
        'e' - ELU
        'g' - GroupNorm
        'b' - BatchNorm
        'd' - Dropout
    """
    assert "c" in order, "必须包含卷积层"
    
    modules = []
    for char in order:
        if char == "r":
            modules.append(("ReLU", nn.ReLU(inplace=True)))
        elif char == "l":
            modules.append(("LeakyReLU", nn.LeakyReLU(negative_slope=0.1, inplace=True)))
        elif char == "e":
            modules.append(("ELU", nn.ELU(inplace=True)))
        elif char == "c":
            if is3d:
                conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias="g" not in order and "b" not in order)
            else:
                conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias="g" not in order and "b" not in order)
            modules.append(("Conv", conv))
        elif char == "g":
            # 自适应 num_groups 以确保可整除
            actual_num_groups = num_groups
            while out_channels % actual_num_groups != 0 and actual_num_groups > 1:
                actual_num_groups -= 1
            modules.append(("GroupNorm", nn.GroupNorm(num_groups=actual_num_groups, num_channels=out_channels)))
        elif char == "b":
            if is3d:
                modules.append(("BatchNorm", nn.BatchNorm3d(out_channels)))
            else:
                modules.append(("BatchNorm", nn.BatchNorm2d(out_channels)))
        elif char == "d":
            if dropout_prob > 0.0:
                modules.append(("Dropout", nn.Dropout(p=dropout_prob)))
        else:
            raise ValueError(f"不支持的层类型: {char}")
    
    return modules


class SingleConv(nn.Sequential):
    """
    单次卷积块：Conv + Norm + Activation + (可选)Dropout
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        order: str = "cgr",  # 默认改为 Conv + GroupNorm + ReLU
        num_groups: int = 8,
        padding: int = 1,
        dropout_prob: float = 0.1,
        is3d: bool = False,
    ):
        super().__init__()
        for name, module in create_conv(
            in_channels, out_channels, kernel_size, order, num_groups, padding, dropout_prob, is3d
        ):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    """
    双卷积块：2x (Conv + Norm + Activation)
    这是 UNet 的核心构建块
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        encoder: bool,
        kernel_size: int = 3,
        order: str = "cgr",  # 默认改为 Conv + GroupNorm + ReLU
        num_groups: int = 8,
        padding: int = 1,
        upscale: int = 2,
        dropout_prob: float = 0.1,
        is3d: bool = False,
    ):
        super().__init__()
        
        if encoder:
            conv1_in_channels = in_channels
            if upscale == 1:
                conv1_out_channels = out_channels
            else:
                conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels
        
        if isinstance(dropout_prob, (list, tuple)):
            dropout_prob1 = dropout_prob[0]
            dropout_prob2 = dropout_prob[1]
        else:
            dropout_prob1 = dropout_prob2 = dropout_prob
        
        self.add_module(
            "SingleConv1",
            SingleConv(
                conv1_in_channels, conv1_out_channels, kernel_size, order,
                num_groups, padding, dropout_prob1, is3d
            ),
        )
        self.add_module(
            "SingleConv2",
            SingleConv(
                conv2_in_channels, conv2_out_channels, kernel_size, order,
                num_groups, padding, dropout_prob2, is3d
            ),
        )


class ResNetBlock(nn.Module):
    """
    ResNet 残差块
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        encoder: bool,
        kernel_size: int = 3,
        order: str = "gcr",
        num_groups: int = 8,
        padding: int = 1,
        upscale: int = 2,
        dropout_prob: float = 0.1,
        is3d: bool = False,
    ):
        super().__init__()
        
        if is3d:
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        self.conv2 = SingleConv(
            out_channels, out_channels, kernel_size, order, num_groups, padding, dropout_prob, is3d
        )
        self.conv3 = SingleConv(
            out_channels, out_channels, kernel_size, order, num_groups, padding, dropout_prob, is3d
        )
        
        self.non_linearity = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = self.conv1(x)
        out = self.conv2(residual)
        out = self.conv3(out)
        out += residual
        out = self.non_linearity(out)
        return out


class Encoder(nn.Module):
    """
    UNet Encoder 单元: Pooling + BasicModule
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_kernel_size: int = 3,
        apply_pooling: bool = True,
        pool_kernel_size: int = 2,
        pool_type: str = "max",
        basic_module: nn.Module = DoubleConv,
        conv_layer_order: str = "cgr",
        num_groups: int = 8,
        padding: int = 1,
        upscale: int = 2,
        dropout_prob: float = 0.1,
        is3d: bool = False,
    ):
        super().__init__()
        
        assert pool_type in ["max", "avg"]
        if apply_pooling:
            if pool_type == "max":
                self.pooling = nn.MaxPool2d(kernel_size=pool_kernel_size) if not is3d else nn.MaxPool3d(kernel_size=pool_kernel_size)
            else:
                self.pooling = nn.AvgPool2d(kernel_size=pool_kernel_size) if not is3d else nn.AvgPool3d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None
        
        self.basic_module = basic_module(
            in_channels, out_channels,
            encoder=True,
            kernel_size=conv_kernel_size,
            order=conv_layer_order,
            num_groups=num_groups,
            padding=padding,
            upscale=upscale,
            dropout_prob=dropout_prob,
            is3d=is3d,
        )
    
    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


def create_encoders(
    in_channels: int,
    f_maps: List[int],
    basic_module: nn.Module,
    conv_kernel_size: int,
    conv_padding: int,
    conv_upscale: int,
    dropout_prob: float,
    layer_order: str,
    num_groups: int,
    pool_kernel_size: int,
    is3d: bool,
) -> nn.ModuleList:
    """创建 Encoder 模块列表"""
    encoders = nn.ModuleList()
    for i, out_feature_num in enumerate(f_maps):
        if i == 0:
            encoder = Encoder(
                in_channels, out_feature_num,
                apply_pooling=False,  # 第一层不需要 pooling
                basic_module=basic_module,
                conv_layer_order=layer_order,
                conv_kernel_size=conv_kernel_size,
                num_groups=num_groups,
                padding=conv_padding,
                upscale=conv_upscale,
                dropout_prob=dropout_prob,
                is3d=is3d,
            )
        else:
            encoder = Encoder(
                f_maps[i - 1], out_feature_num,
                apply_pooling=True,
                pool_kernel_size=pool_kernel_size,
                basic_module=basic_module,
                conv_layer_order=layer_order,
                conv_kernel_size=conv_kernel_size,
                num_groups=num_groups,
                padding=conv_padding,
                upscale=conv_upscale,
                dropout_prob=dropout_prob,
                is3d=is3d,
            )
        encoders.append(encoder)
    return encoders


# =============================================================================
#  UNet2D Classifier (用于图像分类任务)
# =============================================================================

class UNet2DClassifier(nn.Module):
    """
    2D UNet 分类器 - 基于 LucaLumetti/UNetTransplant
    
    使用 UNet 的 Encoder 作为特征提取器，添加 Global Pooling + 分类头
    
    Args:
        in_channels: 输入通道数 (RGB=3, 灰度=1)
        num_classes: 分类类别数
        f_maps: 每层的特征通道数 (int 或 list)
        num_levels: Encoder 层数
        basic_module: 基础卷积块类型 ('double_conv' 或 'resnet_block')
        layer_order: 层顺序 ('cgr' = Conv+GroupNorm+ReLU)
        num_groups: GroupNorm 的组数
        dropout_prob: Dropout 概率
        use_multi_scale: 是否使用多尺度特征融合
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        f_maps: Union[int, List[int]] = 32,
        num_levels: int = 4,
        basic_module: str = "double_conv",
        layer_order: str = "cgr",
        num_groups: int = 8,
        dropout_prob: float = 0.1,
        use_multi_scale: bool = True,
    ):
        super().__init__()
        
        self.use_multi_scale = use_multi_scale
        
        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels)
        
        self.f_maps = f_maps
        
        if basic_module == "double_conv":
            basic_module_class = DoubleConv
        elif basic_module == "resnet_block":
            basic_module_class = ResNetBlock
        else:
            raise ValueError(f"不支持的 basic_module: {basic_module}")
        
        # 创建 Encoder
        self.encoders = create_encoders(
            in_channels=in_channels,
            f_maps=f_maps,
            basic_module=basic_module_class,
            conv_kernel_size=3,
            conv_padding=1,
            conv_upscale=2,
            dropout_prob=dropout_prob,
            layer_order=layer_order,
            num_groups=num_groups,
            pool_kernel_size=2,
            is3d=False,  # 2D 模式
        )
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 分类头
        if use_multi_scale:
            # 多尺度特征融合：拼接所有层的特征
            total_features = sum(f_maps)
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(total_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes),
            )
        else:
            # 仅使用最后一层特征
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(f_maps[-1], 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes),
            )
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        encoder_features = []
        
        for encoder in self.encoders:
            x = encoder(x)
            encoder_features.append(x)
        
        if self.use_multi_scale:
            # 多尺度特征融合
            pooled_features = []
            for feat in encoder_features:
                pooled = self.global_pool(feat).flatten(1)
                pooled_features.append(pooled)
            x = torch.cat(pooled_features, dim=1)
        else:
            # 仅使用最后一层
            x = self.global_pool(x).flatten(1)
        
        return self.classifier(x)


class ResidualUNet2DClassifier(nn.Module):
    """
    Residual 2D UNet 分类器 - 使用 ResNet 残差块
    
    基于 ResidualUNet3D 改造为 2D 分类模型
    论文参考: https://arxiv.org/pdf/1706.00120.pdf
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        f_maps: Union[int, List[int]] = 64,
        num_levels: int = 4,
        layer_order: str = "cgdr",  # Conv + GroupNorm + Dropout + ReLU
        num_groups: int = 8,
        dropout_prob: float = 0.1,
        use_multi_scale: bool = True,
    ):
        super().__init__()
        
        self.use_multi_scale = use_multi_scale
        
        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels)
        
        self.f_maps = f_maps
        
        # 创建 Encoder (使用 ResNetBlock)
        self.encoders = create_encoders(
            in_channels=in_channels,
            f_maps=f_maps,
            basic_module=ResNetBlock,
            conv_kernel_size=3,
            conv_padding=1,
            conv_upscale=2,
            dropout_prob=dropout_prob,
            layer_order=layer_order,
            num_groups=num_groups,
            pool_kernel_size=2,
            is3d=False,
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        if use_multi_scale:
            total_features = sum(f_maps)
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(total_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(f_maps[-1], 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes),
            )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        encoder_features = []
        
        for encoder in self.encoders:
            x = encoder(x)
            encoder_features.append(x)
        
        if self.use_multi_scale:
            pooled_features = []
            for feat in encoder_features:
                pooled = self.global_pool(feat).flatten(1)
                pooled_features.append(pooled)
            x = torch.cat(pooled_features, dim=1)
        else:
            x = self.global_pool(x).flatten(1)
        
        return self.classifier(x)


# =============================================================================
#  测试代码
# =============================================================================
if __name__ == "__main__":
    # 测试模型
    print("=" * 60)
    print("测试 UNet2DClassifier")
    print("=" * 60)
    
    # 标准 UNet2D 分类器
    model = UNet2DClassifier(
        in_channels=3,
        num_classes=4,
        f_maps=32,
        num_levels=4,
        basic_module="double_conv",
        use_multi_scale=True,
    )
    
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"UNet2DClassifier 输入: {dummy_input.shape}")
    print(f"UNet2DClassifier 输出: {output.shape}")
    print(f"特征通道: {model.f_maps}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f}M")
    
    print("\n" + "=" * 60)
    print("测试 ResidualUNet2DClassifier")
    print("=" * 60)
    
    # Residual UNet2D 分类器
    res_model = ResidualUNet2DClassifier(
        in_channels=3,
        num_classes=4,
        f_maps=64,
        num_levels=4,
        use_multi_scale=True,
    )
    
    output2 = res_model(dummy_input)
    print(f"ResidualUNet2DClassifier 输入: {dummy_input.shape}")
    print(f"ResidualUNet2DClassifier 输出: {output2.shape}")
    print(f"特征通道: {res_model.f_maps}")
    
    total_params2 = sum(p.numel() for p in res_model.parameters())
    print(f"总参数量: {total_params2 / 1e6:.2f}M")
    
    print("\n✅ 模型测试通过!")
