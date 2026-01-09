"""
nnUNet v2 Classifier - 基于 MIC-DKFZ/nnUNet 的图像分类器
参考: https://github.com/MIC-DKFZ/nnUNet
论文: "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation"
      Nature Methods 2021

将原始分割模型 (ResidualEncoderUNet / PlainConvUNet) 改造为图像分类模型：
1. 使用 Encoder 作为特征提取器
2. 移除 Decoder 部分
3. 添加 Global Average Pooling + 分类头

适配你的脑肿瘤分类任务 (4类: glioma, meningioma, no_tumor, pituitary)

注意: nnUNet 使用 dynamic_network_architectures 库的网络架构
本实现直接移植核心组件，不需要额外安装 nnunetv2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Tuple, Type, Optional


# =============================================================================
#  Building Blocks (从 dynamic_network_architectures 移植)
# =============================================================================

def get_matching_instancenorm(dimension):
    """根据维度获取对应的 InstanceNorm"""
    if dimension == 2:
        return nn.InstanceNorm2d
    elif dimension == 3:
        return nn.InstanceNorm3d
    else:
        raise ValueError(f"Unsupported dimension: {dimension}")


class ConvDropoutNormReLU(nn.Module):
    """基础卷积块: Conv -> Dropout -> Norm -> ReLU"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        conv_bias: bool = True,
        norm_op: Type[nn.Module] = nn.InstanceNorm2d,
        norm_op_kwargs: dict = None,
        dropout_op: Type[nn.Module] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = None,
        conv_op: Type[nn.Module] = nn.Conv2d,
    ):
        super().__init__()
        
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {}
            
        padding = kernel_size // 2
        
        self.conv = conv_op(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, bias=conv_bias
        )
        
        self.norm = norm_op(out_channels, **norm_op_kwargs) if norm_op else None
        self.dropout = dropout_op(**dropout_op_kwargs) if dropout_op else None
        self.nonlin = nonlin(**nonlin_kwargs) if nonlin else None
        
    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.nonlin is not None:
            x = self.nonlin(x)
        return x


class StackedConvBlocks(nn.Module):
    """堆叠的卷积块"""
    def __init__(
        self,
        n_conv: int,
        conv_op: Type[nn.Module],
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        conv_bias: bool = True,
        norm_op: Type[nn.Module] = nn.InstanceNorm2d,
        norm_op_kwargs: dict = None,
        dropout_op: Type[nn.Module] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = None,
    ):
        super().__init__()
        
        convs = []
        for i in range(n_conv):
            in_c = in_channels if i == 0 else out_channels
            s = stride if i == 0 else 1
            convs.append(
                ConvDropoutNormReLU(
                    in_c, out_channels, kernel_size, s, conv_bias,
                    norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                    nonlin, nonlin_kwargs, conv_op
                )
            )
        self.convs = nn.Sequential(*convs)
        
    def forward(self, x):
        return self.convs(x)


class BasicBlockD(nn.Module):
    """
    ResNet Basic Block - nnUNet 风格
    与标准 ResNet 的区别: 使用 Instance Norm 而不是 Batch Norm
    """
    def __init__(
        self,
        conv_op: Type[nn.Module],
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        conv_bias: bool = True,
        norm_op: Type[nn.Module] = nn.InstanceNorm2d,
        norm_op_kwargs: dict = None,
        dropout_op: Type[nn.Module] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = None,
        stochastic_depth_p: float = 0.0,
        squeeze_excitation: bool = False,
        squeeze_excitation_reduction_ratio: float = 1.0 / 16,
    ):
        super().__init__()
        
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {}
            
        padding = kernel_size // 2
        
        # 主路径
        self.conv1 = conv_op(in_channels, out_channels, kernel_size, stride, padding, bias=conv_bias)
        self.norm1 = norm_op(out_channels, **norm_op_kwargs) if norm_op else nn.Identity()
        self.nonlin1 = nonlin(**nonlin_kwargs) if nonlin else nn.Identity()
        
        self.conv2 = conv_op(out_channels, out_channels, kernel_size, 1, padding, bias=conv_bias)
        self.norm2 = norm_op(out_channels, **norm_op_kwargs) if norm_op else nn.Identity()
        self.nonlin2 = nonlin(**nonlin_kwargs) if nonlin else nn.Identity()
        
        # Skip connection
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                conv_op(in_channels, out_channels, 1, stride, 0, bias=False),
                norm_op(out_channels, **norm_op_kwargs) if norm_op else nn.Identity()
            )
        else:
            self.skip = nn.Identity()
            
        # Squeeze-and-Excitation
        self.apply_se = squeeze_excitation
        if self.apply_se:
            se_channels = max(int(out_channels * squeeze_excitation_reduction_ratio), 1)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, se_channels, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(se_channels, out_channels, 1),
                nn.Sigmoid()
            )
            
        # Stochastic Depth
        self.apply_stochastic_depth = stochastic_depth_p > 0
        if self.apply_stochastic_depth:
            from timm.models.layers import DropPath
            self.drop_path = DropPath(stochastic_depth_p)
            
    def forward(self, x):
        residual = self.skip(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.nonlin1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        if self.apply_stochastic_depth:
            out = self.drop_path(out)
        if self.apply_se:
            out = out * self.se(out)
            
        out = out + residual
        out = self.nonlin2(out)
        return out


class StackedResidualBlocks(nn.Module):
    """堆叠的残差块"""
    def __init__(
        self,
        n_blocks: int,
        conv_op: Type[nn.Module],
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        conv_bias: bool = True,
        norm_op: Type[nn.Module] = nn.InstanceNorm2d,
        norm_op_kwargs: dict = None,
        dropout_op: Type[nn.Module] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = None,
        stochastic_depth_p: float = 0.0,
        squeeze_excitation: bool = False,
        squeeze_excitation_reduction_ratio: float = 1.0 / 16,
    ):
        super().__init__()
        
        blocks = []
        for i in range(n_blocks):
            in_c = in_channels if i == 0 else out_channels
            s = stride if i == 0 else 1
            blocks.append(
                BasicBlockD(
                    conv_op, in_c, out_channels, kernel_size, s,
                    conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                    nonlin, nonlin_kwargs, stochastic_depth_p, squeeze_excitation,
                    squeeze_excitation_reduction_ratio
                )
            )
        self.blocks = nn.Sequential(*blocks)
        
    def forward(self, x):
        return self.blocks(x)


# =============================================================================
#  Encoder (从 nnUNet 的 ResidualEncoder 简化)
# =============================================================================

class PlainConvEncoder(nn.Module):
    """Plain Convolutional Encoder - 用于 PlainConvUNet"""
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int]],
        conv_op: Type[nn.Module] = nn.Conv2d,
        kernel_sizes: Union[int, List[int]] = 3,
        strides: Union[int, List[int]] = (1, 2, 2, 2, 2),
        n_conv_per_stage: Union[int, List[int]] = 2,
        conv_bias: bool = True,
        norm_op: Type[nn.Module] = nn.InstanceNorm2d,
        norm_op_kwargs: dict = None,
        dropout_op: Type[nn.Module] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = None,
    ):
        super().__init__()
        
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage * (2 ** i) for i in range(n_stages)]
        if isinstance(strides, int):
            strides = [strides] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
            
        stages = []
        in_channels = input_channels
        for s in range(n_stages):
            stages.append(
                StackedConvBlocks(
                    n_conv_per_stage[s], conv_op, in_channels, features_per_stage[s],
                    kernel_sizes[s], strides[s], conv_bias, norm_op, norm_op_kwargs,
                    dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs
                )
            )
            in_channels = features_per_stage[s]
            
        self.stages = nn.ModuleList(stages)
        self.output_channels = features_per_stage
        
    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class ResidualEncoder(nn.Module):
    """Residual Encoder - 用于 ResidualEncoderUNet"""
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int]],
        conv_op: Type[nn.Module] = nn.Conv2d,
        kernel_sizes: Union[int, List[int]] = 3,
        strides: Union[int, List[int]] = (1, 2, 2, 2, 2),
        n_blocks_per_stage: Union[int, List[int]] = (1, 3, 4, 6, 6),
        conv_bias: bool = True,
        norm_op: Type[nn.Module] = nn.InstanceNorm2d,
        norm_op_kwargs: dict = None,
        dropout_op: Type[nn.Module] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = None,
        stochastic_depth_p: float = 0.0,
        squeeze_excitation: bool = False,
    ):
        super().__init__()
        
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage * (2 ** i) for i in range(n_stages)]
        if isinstance(strides, int):
            strides = [strides] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
            
        # Stem - 第一个卷积层
        self.stem = StackedConvBlocks(
            1, conv_op, input_channels, features_per_stage[0],
            kernel_sizes[0], 1, conv_bias, norm_op, norm_op_kwargs,
            dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs
        )
        
        # Residual stages
        stages = []
        for s in range(n_stages):
            in_c = features_per_stage[s - 1] if s > 0 else features_per_stage[0]
            stages.append(
                StackedResidualBlocks(
                    n_blocks_per_stage[s], conv_op, in_c, features_per_stage[s],
                    kernel_sizes[s], strides[s], conv_bias, norm_op, norm_op_kwargs,
                    dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                    stochastic_depth_p, squeeze_excitation
                )
            )
            
        self.stages = nn.ModuleList(stages)
        self.output_channels = features_per_stage
        
    def forward(self, x):
        x = self.stem(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


# =============================================================================
#  nnUNet v2 Classifier (用于图像分类任务)
# =============================================================================

class nnUNetClassifier(nn.Module):
    """
    nnUNet v2 分类器 - 基于 MIC-DKFZ/nnUNet
    
    将 nnUNet 的 Encoder 改造为分类器
    
    Args:
        in_channels: 输入通道数 (RGB=3)
        num_classes: 分类类别数
        encoder_type: 'residual' 或 'plain'
        n_stages: Encoder 阶段数
        features_per_stage: 每阶段特征数
        n_blocks_per_stage: 每阶段的残差块数量 (仅 residual)
        n_conv_per_stage: 每阶段的卷积数量 (仅 plain)
        strides: 每阶段的步长
        drop_rate: Dropout 概率
        use_multi_scale: 是否使用多尺度特征融合
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        encoder_type: str = 'residual',
        n_stages: int = 5,
        features_per_stage: List[int] = None,
        n_blocks_per_stage: List[int] = None,
        n_conv_per_stage: List[int] = None,
        strides: List[int] = None,
        kernel_sizes: int = 3,
        conv_bias: bool = True,
        norm_op: Type[nn.Module] = nn.InstanceNorm2d,
        norm_op_kwargs: dict = None,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = None,
        drop_rate: float = 0.5,
        use_multi_scale: bool = True,
        squeeze_excitation: bool = False,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_multi_scale = use_multi_scale
        self.encoder_type = encoder_type
        
        # 默认配置 (类似 nnUNet 2D)
        if features_per_stage is None:
            features_per_stage = [32, 64, 128, 256, 512][:n_stages]
        if strides is None:
            strides = [1] + [2] * (n_stages - 1)
        if n_blocks_per_stage is None:
            n_blocks_per_stage = [1, 3, 4, 6, 6][:n_stages]
        if n_conv_per_stage is None:
            n_conv_per_stage = [2] * n_stages
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {'inplace': True}
            
        self.features_per_stage = features_per_stage
        
        # 构建 Encoder
        if encoder_type == 'residual':
            self.encoder = ResidualEncoder(
                input_channels=in_channels,
                n_stages=n_stages,
                features_per_stage=features_per_stage,
                conv_op=nn.Conv2d,
                kernel_sizes=kernel_sizes,
                strides=strides,
                n_blocks_per_stage=n_blocks_per_stage,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                squeeze_excitation=squeeze_excitation,
            )
        else:
            self.encoder = PlainConvEncoder(
                input_channels=in_channels,
                n_stages=n_stages,
                features_per_stage=features_per_stage,
                conv_op=nn.Conv2d,
                kernel_sizes=kernel_sizes,
                strides=strides,
                n_conv_per_stage=n_conv_per_stage,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
            )
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 分类头
        if use_multi_scale:
            total_features = sum(features_per_stage)
            self.classifier = nn.Sequential(
                nn.Dropout(drop_rate),
                nn.Linear(total_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(drop_rate * 0.6),
                nn.Linear(512, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(drop_rate),
                nn.Linear(features_per_stage[-1], 256),
                nn.ReLU(inplace=True),
                nn.Dropout(drop_rate * 0.6),
                nn.Linear(256, num_classes),
            )
            
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.InstanceNorm2d, nn.BatchNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        # Encoder 输出多尺度特征
        features = self.encoder(x)
        
        if self.use_multi_scale:
            # 多尺度特征融合
            pooled_features = []
            for feat in features:
                pooled = self.global_pool(feat).flatten(1)
                pooled_features.append(pooled)
            x = torch.cat(pooled_features, dim=1)
        else:
            # 仅使用最后一层特征
            x = self.global_pool(features[-1]).flatten(1)
            
        return self.classifier(x)


# =============================================================================
#  预定义模型配置
# =============================================================================

class nnUNetClassifierTiny(nnUNetClassifier):
    """nnUNet Classifier Tiny - 轻量级版本"""
    def __init__(self, in_channels=3, num_classes=4, **kwargs):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            encoder_type='plain',
            n_stages=4,
            features_per_stage=[24, 48, 96, 192],
            n_conv_per_stage=[2, 2, 2, 2],
            strides=[1, 2, 2, 2],
            use_multi_scale=True,
            **kwargs,
        )


class nnUNetClassifierSmall(nnUNetClassifier):
    """nnUNet Classifier Small - 小型版本 (类似 nnUNet default)"""
    def __init__(self, in_channels=3, num_classes=4, **kwargs):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            encoder_type='plain',
            n_stages=5,
            features_per_stage=[32, 64, 128, 256, 512],
            n_conv_per_stage=[2, 2, 2, 2, 2],
            strides=[1, 2, 2, 2, 2],
            use_multi_scale=True,
            **kwargs,
        )


class nnUNetResEncClassifierSmall(nnUNetClassifier):
    """nnUNet ResEnc Classifier Small - 带残差的小型版本"""
    def __init__(self, in_channels=3, num_classes=4, **kwargs):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            encoder_type='residual',
            n_stages=5,
            features_per_stage=[32, 64, 128, 256, 320],
            n_blocks_per_stage=[1, 2, 3, 4, 4],
            strides=[1, 2, 2, 2, 2],
            use_multi_scale=True,
            **kwargs,
        )


class nnUNetResEncClassifierMedium(nnUNetClassifier):
    """nnUNet ResEnc Classifier Medium - 中型残差版本 (类似 nnUNet ResEnc M)"""
    def __init__(self, in_channels=3, num_classes=4, **kwargs):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            encoder_type='residual',
            n_stages=6,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
            strides=[1, 2, 2, 2, 2, 2],
            use_multi_scale=True,
            squeeze_excitation=False,
            **kwargs,
        )


class nnUNetResEncClassifierLarge(nnUNetClassifier):
    """nnUNet ResEnc Classifier Large - 大型残差版本 (类似 nnUNet ResEnc L)"""
    def __init__(self, in_channels=3, num_classes=4, **kwargs):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            encoder_type='residual',
            n_stages=7,
            features_per_stage=[32, 64, 128, 256, 512, 512, 512],
            n_blocks_per_stage=[1, 3, 4, 6, 6, 6, 6],
            strides=[1, 2, 2, 2, 2, 2, 2],
            use_multi_scale=True,
            squeeze_excitation=True,
            **kwargs,
        )


# =============================================================================
#  测试代码
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("测试 nnUNet Classifier")
    print("=" * 60)
    
    dummy_input = torch.randn(2, 3, 224, 224)
    
    # 测试 Tiny 版本
    print("\n--- nnUNetClassifierTiny (Plain Conv) ---")
    model_tiny = nnUNetClassifierTiny(in_channels=3, num_classes=4)
    try:
        output = model_tiny(dummy_input)
        print(f"输入: {dummy_input.shape}")
        print(f"输出: {output.shape}")
        print(f"特征维度: {model_tiny.features_per_stage}")
        total_params = sum(p.numel() for p in model_tiny.parameters())
        print(f"参数量: {total_params / 1e6:.2f}M")
        print("✅ 测试通过!")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试 Small 版本
    print("\n--- nnUNetClassifierSmall (Plain Conv) ---")
    model_small = nnUNetClassifierSmall(in_channels=3, num_classes=4)
    try:
        output = model_small(dummy_input)
        print(f"输入: {dummy_input.shape}")
        print(f"输出: {output.shape}")
        print(f"特征维度: {model_small.features_per_stage}")
        total_params = sum(p.numel() for p in model_small.parameters())
        print(f"参数量: {total_params / 1e6:.2f}M")
        print("✅ 测试通过!")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试 ResEnc Small 版本
    print("\n--- nnUNetResEncClassifierSmall (Residual) ---")
    model_resenc_small = nnUNetResEncClassifierSmall(in_channels=3, num_classes=4)
    try:
        output = model_resenc_small(dummy_input)
        print(f"输入: {dummy_input.shape}")
        print(f"输出: {output.shape}")
        print(f"特征维度: {model_resenc_small.features_per_stage}")
        total_params = sum(p.numel() for p in model_resenc_small.parameters())
        print(f"参数量: {total_params / 1e6:.2f}M")
        print("✅ 测试通过!")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试 ResEnc Medium 版本
    print("\n--- nnUNetResEncClassifierMedium (Residual) ---")
    model_resenc_m = nnUNetResEncClassifierMedium(in_channels=3, num_classes=4)
    try:
        output = model_resenc_m(dummy_input)
        print(f"输入: {dummy_input.shape}")
        print(f"输出: {output.shape}")
        print(f"特征维度: {model_resenc_m.features_per_stage}")
        total_params = sum(p.numel() for p in model_resenc_m.parameters())
        print(f"参数量: {total_params / 1e6:.2f}M")
        print("✅ 测试通过!")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)
