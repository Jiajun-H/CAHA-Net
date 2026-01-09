import torch
import torch.nn as nn
from torchvision import models
from torchvision.ops import DeformConv2d

# ================= 1. CA 模块 (Coordinate Attention) =================
class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        out = identity * a_h * a_w
        return out

# ================= 2. DCN 模块 (可变形卷积) =================
class DeformableConv2dWrapper(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DeformableConv2dWrapper, self).__init__()
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, 
                                     kernel_size=kernel_size, stride=stride, padding=padding)
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        
        self.dcn = DeformConv2d(in_channels, out_channels, kernel_size=kernel_size, 
                                stride=stride, padding=padding)

    def forward(self, x):
        offset = self.offset_conv(x)
        return self.dcn(x, offset)

def replace_conv_with_dcn(module):
    """递归将模块中的 3x3 Conv2d 替换为 DCN"""
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d) and child.kernel_size == (3, 3):
            dcn = DeformableConv2dWrapper(child.in_channels, child.out_channels, 
                                          kernel_size=3, stride=child.stride, padding=child.padding)
            setattr(module, name, dcn)
        else:
            replace_conv_with_dcn(child)

# ================= 3. 终极模型架构 (支持全模块消融) =================
class BrainTumorFinalNet(nn.Module):
    def __init__(self, num_classes=4, use_dcn=True, use_ca=True, use_symmetry=True):
        super(BrainTumorFinalNet, self).__init__()
        
        self.use_symmetry = use_symmetry
        print(f">>> 初始化模型 | DCN: {use_dcn} | CA: {use_ca} | Sym: {use_symmetry}")

        # 1. 加载预训练权重
        #base_model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        base_model = models.densenet121(weights=None)
        # 2. [开关] DCN 模块
        if use_dcn:
            replace_conv_with_dcn(base_model.features.denseblock3)
            replace_conv_with_dcn(base_model.features.denseblock4)
        
        # 3. 拆分 Feature 提取器
        features = base_model.features
        
        self.initial_layers = nn.Sequential(
            features.conv0, features.norm0, features.relu0, features.pool0
        )
        
        self.block1 = features.denseblock1
        self.trans1 = features.transition1
        
        # --- Block 2 ---
        self.block2 = features.denseblock2
        self.ca2 = CoordAtt(512) if use_ca else nn.Identity()
        self.trans2 = features.transition2
        
        # --- Block 3 ---
        self.block3 = features.denseblock3
        self.ca3 = CoordAtt(1024) if use_ca else nn.Identity()
        self.trans3 = features.transition3
        
        # --- Block 4 ---
        self.block4 = features.denseblock4
        self.ca4 = CoordAtt(1024) if use_ca else nn.Identity()
        self.norm5 = features.norm5
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 4. 分类头 (根据 Symmetry 开关动态调整输入维度)
        feat_dim = 512 + 1024 + 1024
        
        # 如果使用 Symmetry，特征是 [Original, Diff]，所以维度 x2
        final_dim = (feat_dim * 2) if use_symmetry else feat_dim
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(final_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward_single_branch(self, x):
        x = self.initial_layers(x)
        x = self.block1(x)
        x = self.trans1(x)
        
        x = self.block2(x)
        x = self.ca2(x)
        feat2 = x 
        x = self.trans2(x)
        
        x = self.block3(x)
        x = self.ca3(x) 
        feat3 = x 
        x = self.trans3(x)
        
        x = self.block4(x)
        x = self.ca4(x)
        feat4 = self.norm5(x) 
        
        return feat2, feat3, feat4

    def forward(self, x):
        # 1. 原始图像分支
        f2, f3, f4 = self.forward_single_branch(x)
        f2 = self.avgpool(f2).flatten(1)
        f3 = self.avgpool(f3).flatten(1)
        f4 = self.avgpool(f4).flatten(1)
        feat_origin = torch.cat([f2, f3, f4], dim=1)
        
        # [开关] 如果不开启对称性，直接返回单流结果 (Pure DenseNet)
        if not self.use_symmetry:
            return self.classifier(feat_origin)
        
        # 2. 翻转图像分支
        x_flip = torch.flip(x, dims=[3]) 
        f2_r, f3_r, f4_r = self.forward_single_branch(x_flip)
        f2_r = self.avgpool(f2_r).flatten(1)
        f3_r = self.avgpool(f3_r).flatten(1)
        f4_r = self.avgpool(f4_r).flatten(1)
        feat_flip = torch.cat([f2_r, f3_r, f4_r], dim=1)
        
        # 3. 差分与融合
        feat_diff = torch.abs(feat_origin - feat_flip)
        final_feat = torch.cat([feat_origin, feat_diff], dim=1)
        
        return self.classifier(final_feat)
