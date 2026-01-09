import os
import sys
import shutil
import gc
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from albumentations import Compose, CLAHE, HorizontalFlip, Rotate
from PIL import Image

from model_utils import CoordAtt

# 过滤部分不影响训练的警告，避免刷屏影响可读性
warnings.filterwarnings('ignore', category=FutureWarning, module='timm')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

try:
    from thop import profile
except ImportError:
    print("❌ 警告: 未安装 'thop' 库。无法计算 GFLOPs。请运行 pip install thop")
    profile = None


# ================= 0. VSSD 引入（external/VSSD） =================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VSSD_ROOT = os.path.join(_THIS_DIR, "external", "VSSD")
_VSSD_CLS_ROOT = os.path.join(_VSSD_ROOT, "classification")

if not os.path.isdir(_VSSD_CLS_ROOT):
    raise RuntimeError("未找到 external/VSSD/classification。请确认 external/VSSD 已存在且结构完整。")

# 只把 classification 根加入 sys.path，这样可 import models.*
if _VSSD_CLS_ROOT not in sys.path:
    sys.path.insert(0, _VSSD_CLS_ROOT)

try:
    from models.mamba2 import Backbone_VMAMBA2
except Exception as e:
    raise RuntimeError(
        "导入 VSSD 分类模型失败。\n"
        "常见原因：缺少依赖（einops/fvcore/timm/mamba_ssm 等）或 CUDA 扩展不可用。\n"
        f"原始错误: {e}"
    )


# ================= 1. 基础配置（保持 train_dinov3.py 结构） =================
BASE_CONFIG = {
    'train_dir': './data/train',
    'val_dir': './data/val',
    'log_dir': './logs_vssd_2025sota',
    'save_dir': './checkpoints_vssd_2025sota',
    'num_classes': 4,

    # VSSD（尤其 base）更吃显存；默认偏保守
    'batch_size': 4,
    'lr': 3e-4,
    'epochs': 60,
    # Windows 下 dataloader 往往是瓶颈；适当提高 workers/prefetch 可明显提升 GPU 利用率
    'num_workers': 8,
    'prefetch_factor': 4,
    'persistent_workers': True,
    'use_amp': True,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # VSSD ICCV2025 版本：tiny | small | base
    'vssd_variant': 'tiny',

    # 可选：加载 VSSD 预训练权重（本地路径）。
    # 你也可以从 VSSD readme 的 HF 链接下载到本地再填这里。
    'vssd_pretrained_ckpt': None,

    # 预训练 ckpt 的 key（VSSD 仓库默认常见是 model_ema）
    'vssd_ckpt_key': 'model_ema',

    # 效率评测（会额外跑很多次前向，默认只在初始化时评测一次）
    'efficiency_warmup': 10,
    'efficiency_iterations': 30,
    'measure_efficiency_on_best': False,
}


# ================= 2. 辅助类 (Loss, Augmentation, FMix) =================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class AlbumentationsTransform:
    def __init__(self, is_train=True, use_clahe=True):
        aug_list = []
        if use_clahe:
            aug_list.append(CLAHE(clip_limit=4.0, p=1.0))
        if is_train:
            aug_list.extend([Rotate(limit=15, p=0.5), HorizontalFlip(p=0.5)])
        self.aug = Compose(aug_list)

    def __call__(self, img):
        img_np = np.array(img)
        if self.aug:
            return Image.fromarray(self.aug(image=img_np)['image'])
        return Image.fromarray(img_np)


class IdentityTransform:
    def __call__(self, img):
        return img


def fftfreqnd(h, w=None, z=None):
    fz = fx = 0
    fy = np.fft.fftfreq(h)
    if w is not None:
        fx = np.fft.fftfreq(w)
    if z is not None:
        fz = np.fft.fftfreq(z)
    return np.meshgrid(fy, fx, indexing='ij')


def get_spectrum(freq_space, decay_power=2):
    scale = np.ones(1) / (np.maximum(freq_space, np.array([1. / max(freq_space.shape)])) ** decay_power)
    param_size = [len(freq_space)] + list(freq_space.shape)
    param = np.random.randn(*param_size)
    return np.expand_dims(scale, axis=0) * param


def make_low_freq_image(decay, shape, ch=1):
    freq_space = fftfreqnd(shape[0], shape[1])
    spectrum = get_spectrum(np.array(freq_space), decay_power=decay)
    mask = np.real(np.fft.ifft2(spectrum[:1])).astype(np.float32)
    mask = mask[0, 0] if mask.ndim == 4 else mask[0]
    if mask.ndim > 2:
        mask = mask[0]
    mask = mask - mask.min()
    return mask / (mask.max() + 1e-6)


def fmix_data(data, targets, alpha=1.0, decay_power=3.0, shape=(224, 224), device='cuda'):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    index = torch.randperm(data.size(0)).to(device)
    soft_mask = make_low_freq_image(decay_power, shape)
    mask_flat = soft_mask.flatten()
    idx = int((1 - lam) * len(mask_flat))
    threshold = np.partition(mask_flat, idx)[idx]
    binary_mask = torch.from_numpy((soft_mask > threshold).astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)
    mixed_x = data * binary_mask + data[index] * (1 - binary_mask)
    return mixed_x, targets, targets[index], binary_mask.mean().item()


# ================= 3. 效率评估函数 =================
def measure_efficiency(model, device, input_size=(1, 3, 224, 224), warmup=20, iterations=100):
    model.eval()
    dummy_input = torch.randn(input_size).to(device)

    flops_g, params_m = 0.0, 0.0
    if profile:
        try:
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
            flops_g = flops / 1e9
            params_m = params / 1e6
        except Exception as e:
            print(f"GFLOPs calculation failed: {e}")

    for _ in range(warmup):
        with torch.no_grad():
            _ = model(dummy_input)

    if device == 'cuda':
        torch.cuda.synchronize()
    t_start = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
    if device == 'cuda':
        torch.cuda.synchronize()
    t_end = time.time()

    avg_time = (t_end - t_start) / iterations
    fps = 1.0 / avg_time
    latency_ms = avg_time * 1000
    return flops_g, params_m, fps, latency_ms


# ================= 4. VSSD (ICCV2025) 模型实现（Backbone_VMAMBA2） =================
_VSSD_ICCV2025_VARIANTS = {
    # 来自 external/VSSD/classification/configs/vssd_iccv_version/*.yaml（关键参数子集）
    'tiny': {
        'embed_dim': 96,
        'depths': [2, 2, 8, 2],
        'num_heads': [4, 4, 8, 16],
        'drop_path_rate': 0.2,
        'ssd_expansion': 1,
        'ssd_chunk_size': 256,
        'linear_attn_duality': True,
        'attn_types': ['mamba2', 'mamba2', 'mamba2', 'standard'],
        'async_state': [12, 24, 48, 64],
        'mlp_ratio': 3.0,
        'rmt_downsample': True,
        'rmt_patch_embed': True,
        'use_cpe': True,
        'ssd_linear_norm': True,
        'exp_da': True,
        'rope': True,
        # build_vssd_model 里用的拼写（保持一致）
        'ssd_positve_dA': True,
    },
    'small': {
        'embed_dim': 96,
        'depths': [2, 4, 15, 4],
        'num_heads': [4, 4, 8, 16],
        'drop_path_rate': 0.4,
        'ssd_expansion': 1,
        'ssd_chunk_size': 256,
        'linear_attn_duality': True,
        'attn_types': ['mamba2', 'mamba2', 'mamba2', 'standard'],
        'async_state': [12, 24, 48, 64],
        'mlp_ratio': 3.0,
        'rmt_downsample': True,
        'rmt_patch_embed': True,
        'use_cpe': True,
        'ssd_linear_norm': True,
        'exp_da': True,
        'rope': True,
        'ssd_positve_dA': True,
    },
    'base': {
        'embed_dim': 96,
        'depths': [3, 4, 21, 5],
        'num_heads': [3, 6, 12, 24],
        'drop_path_rate': 0.6,
        'ssd_expansion': 2,
        'ssd_chunk_size': 256,
        'linear_attn_duality': True,
        'attn_types': ['mamba2', 'mamba2', 'mamba2', 'standard'],
        'mlp_ratio': 3.0,
    },
}


class BrainTumorVSSD2025(nn.Module):
    def __init__(
        self,
        num_classes=4,
        use_dcn=True,
        use_ca=True,
        use_symmetry=True,
        variant='tiny',
        pretrained_ckpt=None,
        ckpt_key='model_ema',
    ):
        super().__init__()
        self.use_symmetry = use_symmetry

        if use_dcn:
            print("⚠️ VSSD 为状态空间/注意力混合骨干，DCN 选项将被忽略。")

        if variant not in _VSSD_ICCV2025_VARIANTS:
            raise ValueError(f"未知 vssd_variant: {variant}. 可选: {list(_VSSD_ICCV2025_VARIANTS.keys())}")

        vcfg = _VSSD_ICCV2025_VARIANTS[variant]
        print(f">>> 初始化 VSSD(ICCV2025, {variant}) | CA: {use_ca} | Sym: {use_symmetry}")

        self.backbone = Backbone_VMAMBA2(
            out_indices=(1, 2, 3),
            pretrained=pretrained_ckpt,
            key=ckpt_key,
            img_size=224,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            embed_dim=vcfg['embed_dim'],
            depths=vcfg['depths'],
            num_heads=vcfg['num_heads'],
            mlp_ratio=vcfg.get('mlp_ratio', 3.0),
            drop_rate=0.0,
            drop_path_rate=vcfg.get('drop_path_rate', 0.2),
            ape=False,
            simple_downsample=False,
            simple_patch_embed=False,
            ssd_expansion=vcfg.get('ssd_expansion', 1),
            ssd_ngroups=1,
            ssd_chunk_size=vcfg.get('ssd_chunk_size', 256),
            linear_attn_duality=vcfg.get('linear_attn_duality', True),
            d_state=64,
            attn_types=vcfg.get('attn_types', ['mamba2', 'mamba2', 'mamba2', 'standard']),
            async_state=vcfg.get('async_state', [None]),
            rmt_downsample=vcfg.get('rmt_downsample', False),
            rmt_patch_embed=vcfg.get('rmt_patch_embed', False),
            use_cpe=vcfg.get('use_cpe', False),
            ssd_linear_norm=vcfg.get('ssd_linear_norm', False),
            exp_da=vcfg.get('exp_da', False),
            rope=vcfg.get('rope', False),
            ssd_positve_dA=vcfg.get('ssd_positve_dA', False),
        )

        # Backbone_VMAMBA2 的输出通道：embed_dim * 2^i
        ch2 = int(vcfg['embed_dim'] * (2 ** 1))
        ch3 = int(vcfg['embed_dim'] * (2 ** 2))
        ch4 = int(vcfg['embed_dim'] * (2 ** 3))

        self.ca2 = CoordAtt(ch2) if use_ca else nn.Identity()
        self.ca3 = CoordAtt(ch3) if use_ca else nn.Identity()
        self.ca4 = CoordAtt(ch4) if use_ca else nn.Identity()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        feat_dim = ch2 + ch3 + ch4
        final_dim = (feat_dim * 2) if use_symmetry else feat_dim

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(final_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward_single_branch(self, x):
        feat2, feat3, feat4 = self.backbone(x)
        feat2 = self.ca2(feat2)
        feat3 = self.ca3(feat3)
        feat4 = self.ca4(feat4)
        return feat2, feat3, feat4

    def forward(self, x):
        f2, f3, f4 = self.forward_single_branch(x)
        f2 = self.avgpool(f2).flatten(1)
        f3 = self.avgpool(f3).flatten(1)
        f4 = self.avgpool(f4).flatten(1)
        feat_origin = torch.cat([f2, f3, f4], dim=1)

        if not self.use_symmetry:
            return self.classifier(feat_origin)

        x_flip = torch.flip(x, dims=[3])
        f2_r, f3_r, f4_r = self.forward_single_branch(x_flip)
        f2_r = self.avgpool(f2_r).flatten(1)
        f3_r = self.avgpool(f3_r).flatten(1)
        f4_r = self.avgpool(f4_r).flatten(1)
        feat_flip = torch.cat([f2_r, f3_r, f4_r], dim=1)

        feat_diff = torch.abs(feat_origin - feat_flip)
        final_feat = torch.cat([feat_origin, feat_diff], dim=1)
        return self.classifier(final_feat)


# ================= 5. 训练逻辑（完全模仿 train_dinov3.py） =================
def train_one_experiment(config):
    device = config['device']

    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    name_parts = []
    if config['use_symmetry']:
        name_parts.append("Sym")
    else:
        name_parts.append("PureBase")
    if config['use_clahe']:
        name_parts.append("CLAHE")
    if config['use_fmix']:
        name_parts.append("FMix")
    if config['use_dcn']:
        name_parts.append("DCN")
    if config['use_ca']:
        name_parts.append("CA")

    name_parts.append(f"VSSD_{config.get('vssd_variant', 'tiny')}")

    exp_name = "_".join(name_parts)
    current_save_dir = os.path.join(config['save_dir'], exp_name)
    current_log_dir = os.path.join(config['log_dir'], exp_name)
    os.makedirs(current_save_dir, exist_ok=True)
    if os.path.exists(current_log_dir):
        shutil.rmtree(current_log_dir)
    writer = SummaryWriter(current_log_dir)

    print("\n" + "#" * 60)
    print(f"🔥 开始执行: {exp_name}")
    print("#" * 60 + "\n")

    # 训练慢且 GPU 利用率低时，最常见瓶颈是 CPU 端的数据增强/解码。
    # use_clahe=False 时改用 torchvision 纯 PIL pipeline，避免 PIL<->numpy 往返。
    if config['use_clahe']:
        train_aug = AlbumentationsTransform(is_train=True, use_clahe=True)
    else:
        train_aug = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        train_aug,
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        IdentityTransform(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    loader_kwargs = dict(
        num_workers=config['num_workers'],
        pin_memory=True,
    )
    if config['num_workers'] > 0:
        loader_kwargs['persistent_workers'] = bool(config.get('persistent_workers', True))
        loader_kwargs['prefetch_factor'] = int(config.get('prefetch_factor', 2))

    train_loader = DataLoader(
        ImageFolder(config['train_dir'], transform=train_transform),
        batch_size=config['batch_size'],
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        ImageFolder(config['val_dir'], transform=val_transform),
        batch_size=config['batch_size'],
        shuffle=False,
        **loader_kwargs,
    )

    model = BrainTumorVSSD2025(
        num_classes=config['num_classes'],
        use_dcn=config['use_dcn'],
        use_ca=config['use_ca'],
        use_symmetry=config['use_symmetry'],
        variant=config.get('vssd_variant', 'tiny'),
        pretrained_ckpt=config.get('vssd_pretrained_ckpt', None),
        ckpt_key=config.get('vssd_ckpt_key', 'model_ema'),
    ).to(device)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    flops_g, params_m, _, _ = measure_efficiency(
        model,
        device,
        warmup=int(config.get('efficiency_warmup', 10)),
        iterations=int(config.get('efficiency_iterations', 30)),
    )
    print(f"📊 模型分析: Params={params_m:.2f}M, GFLOPs={flops_g:.2f}G")

    criterion = FocalLoss(gamma=2.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_macro_f1 = 0.0
    train_start_time = time.time()

    use_amp = bool(config.get('use_amp', False)) and (device == 'cuda')
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0

        for i, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            do_fmix = config['use_fmix'] and (np.random.rand() > 0.5)
            with torch.amp.autocast('cuda', enabled=use_amp):
                if do_fmix:
                    imgs, targets_a, targets_b, lam = fmix_data(imgs, targets, device=device)
                    outputs = model(imgs)
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                else:
                    outputs = model(imgs)
                    loss = criterion(outputs, targets)

            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            if (i + 1) % 20 == 0:
                print(
                    f"\r[Exp: {exp_name}] Epoch {epoch + 1} Step {i + 1}/{len(train_loader)} Loss: {loss.item():.4f}",
                    end="",
                )

        avg_train_loss = train_loss / max(len(train_loader), 1)
        writer.add_scalar('Train/Loss', avg_train_loss, epoch)

        model.eval()
        val_loss = 0.0
        all_preds, all_targets, all_probs = [], [], []

        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(imgs)
                    val_loss += criterion(outputs, targets).item()
                probs = torch.softmax(outputs, dim=1)
                all_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        avg_val_loss = val_loss / max(len(val_loader), 1)
        scheduler.step(avg_val_loss)

        report = classification_report(all_targets, all_preds, output_dict=True, zero_division=0)
        macro_f1 = report['macro avg']['f1-score']
        macro_recall = report['macro avg']['recall']

        try:
            auc = roc_auc_score(
                np.eye(config['num_classes'])[all_targets],
                all_probs,
                multi_class='ovr',
                average='macro',
            )
        except Exception:
            auc = 0.0

        cm = confusion_matrix(all_targets, all_preds)
        FP = cm.sum(axis=0) - np.diag(cm)
        TN = cm.sum() - (FP + (cm.sum(axis=1) - np.diag(cm)) + np.diag(cm))
        macro_spec = (TN / (TN + FP + 1e-6)).mean()
        macro_fpr = (FP / (FP + TN + 1e-6)).mean()

        print(f"\nVal ({exp_name}) -> F1: {macro_f1:.4f} | AUC: {auc:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1

            model_filename = f"{exp_name}_best_model.pth"
            txt_filename = f"{exp_name}_best_metrics.txt"
            save_path_model = os.path.join(current_save_dir, model_filename)
            torch.save(model.state_dict(), save_path_model)

            total_time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - train_start_time))
            model_size_mb = os.path.getsize(save_path_model) / (1024 * 1024)
            max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
            if config.get('measure_efficiency_on_best', False):
                _, _, fps, latency = measure_efficiency(
                    model,
                    device,
                    warmup=int(config.get('efficiency_warmup', 10)),
                    iterations=int(config.get('efficiency_iterations', 30)),
                )
            else:
                fps, latency = 0.0, 0.0

            with open(os.path.join(current_save_dir, txt_filename), "w", encoding="utf-8") as f:
                f.write(f"Experiment:       {exp_name}\n")
                f.write(f"Backbone:         VSSD_{config.get('vssd_variant', 'tiny')}\n")
                f.write(f"Best Epoch:       {epoch + 1}\n")
                f.write("=" * 40 + "\n")
                f.write("--- Clinical Metrics ---\n")
                f.write(f"Macro F1:         {macro_f1:.4f}\n")
                f.write(f"AUC:              {auc:.4f}\n")
                f.write(f"Recall (Sens):    {macro_recall:.4f}\n")
                f.write(f"Specificity:      {macro_spec:.4f}\n")
                f.write(f"FPR:              {macro_fpr:.4f}\n")
                f.write(f"Val Loss:         {avg_val_loss:.4f}\n")
                f.write("\n")
                f.write("--- Computational Efficiency & Deployment ---\n")
                f.write(f"Hardware:         {gpu_name}\n")
                f.write(f"Training Time:    {total_time_str} (Cumulative)\n")
                f.write(f"Max GPU Memory:   {max_memory:.2f} GB\n")
                f.write(f"Parameters:       {params_m:.2f} M\n")
                f.write(f"GFLOPs:           {flops_g:.2f} G\n")
                f.write(f"Model Disk Size:  {model_size_mb:.2f} MB\n")
                f.write(f"Inference Speed:  {fps:.1f} FPS (Batch=1)\n")
                f.write(f"Latency:          {latency:.2f} ms\n")
                f.write("=" * 40 + "\n")
                f.write("Configuration:\n" + str(config))

            print(f"✅ [新纪录] 指标与效率数据已保存: {txt_filename}")

    writer.close()
    del model, optimizer, train_loader, val_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print(f"🏁 实验结束: {exp_name}\n")


def run_all_experiments():
    experiment_queue = [
        {'use_symmetry': False, 'use_clahe': False, 'use_fmix': False, 'use_dcn': False, 'use_ca': False},
    ]

    print(f"📋 任务队列: {len(experiment_queue)} 个实验")
    for idx, params in enumerate(experiment_queue):
        cfg = BASE_CONFIG.copy()
        cfg.update(params)
        print(f"\n🚀 执行第 {idx + 1}/{len(experiment_queue)} 个实验")
        try:
            train_one_experiment(cfg)
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback

            traceback.print_exc()


if __name__ == '__main__':
    run_all_experiments()
