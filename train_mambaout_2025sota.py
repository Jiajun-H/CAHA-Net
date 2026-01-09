import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import numpy as np
import gc
import time
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from albumentations import Compose, CLAHE, HorizontalFlip, Rotate
from PIL import Image

from model_utils import CoordAtt

try:
    import timm
except ImportError:
    print("❌ 错误: 未安装 'timm' 库。请运行 pip install timm")
    raise

try:
    from thop import profile
except ImportError:
    print("❌ 警告: 未安装 'thop' 库。无法计算 GFLOPs。请运行 pip install thop")
    profile = None


# ================= 1. 基础配置（严格沿用 train_dinov3.py 结构） =================
BASE_CONFIG = {
    'train_dir': './data/train',
    'val_dir': './data/val',
    'log_dir': './logs_mambaout_2025sota',
    'save_dir': './checkpoints_mambaout_2025sota',
    'num_classes': 4,
    'batch_size': 12,
    'lr': 1e-3,
    'epochs': 60,
    'num_workers': 4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # 2024 Mamba 系列（timm 内置）
    'mambaout_variant': 'mambaout_base',  # mambaout_tiny | mambaout_small | mambaout_base
    'mambaout_pretrained': False,

    # 可选：离线/本地预训练权重（避免 HuggingFace 下载失败）
    # 例如：'/root/autodl-tmp/weights/mambaout_base.in1k.pth' 或 safetensors 转好的 state_dict
    'mambaout_pretrained_ckpt': None,
    # ckpt 内 state_dict 的 key（常见：'state_dict'/'model'/'model_ema'），None 表示直接就是 state_dict
    'mambaout_ckpt_key': None,
    # 可选：HuggingFace/timm 的缓存目录（有网但慢时可用）
    'hf_cache_dir': None,
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
def measure_efficiency(model, device, input_size=(1, 3, 224, 224)):
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

    for _ in range(20):
        with torch.no_grad():
            _ = model(dummy_input)

    iterations = 100
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


# ================= 4. MambaOut (timm, 2024) 模型实现（严格复用三尺度逻辑） =================
class BrainTumorMambaOut2025(nn.Module):
    def __init__(
        self,
        num_classes=4,
        use_dcn=True,
        use_ca=True,
        use_symmetry=True,
        variant='mambaout_base',
        pretrained=True,
        pretrained_ckpt=None,
        ckpt_key=None,
        hf_cache_dir=None,
    ):
        super().__init__()
        self.use_symmetry = use_symmetry

        if use_dcn:
            print("⚠️ MambaOut 为 Mamba 系列 backbone，DCN 选项将被忽略。")

        print(f">>> 初始化 MambaOut | variant: {variant} | pretrained: {pretrained} | CA: {use_ca} | Sym: {use_symmetry}")

        self.backbone_name = variant

        # 1) 优先尝试 timm 自带 pretrained（会走 HF 下载）；失败则自动回退
        create_kwargs = dict(
            features_only=True,
            out_indices=(1, 2, 3),
        )
        if hf_cache_dir:
            create_kwargs['cache_dir'] = hf_cache_dir

        self.backbone = None
        if pretrained:
            try:
                self.backbone = timm.create_model(variant, pretrained=True, **create_kwargs)
            except Exception as e:
                print(
                    "⚠️ 预训练权重下载/加载失败，将回退为随机初始化（pretrained=False）。\n"
                    f"原始错误: {e}\n"
                    "如果你在无网/受限环境（如 AutoDL），建议：\n"
                    "1) 设置 BASE_CONFIG['mambaout_pretrained']=False\n"
                    "2) 或下载权重到本地并设置 BASE_CONFIG['mambaout_pretrained_ckpt']=本地路径"
                )

        if self.backbone is None:
            self.backbone = timm.create_model(variant, pretrained=False, **create_kwargs)

        # 2) 若提供本地 ckpt，则从本地加载（不依赖网络）
        if pretrained_ckpt:
            if not os.path.isfile(pretrained_ckpt):
                raise FileNotFoundError(f"未找到 mambaout_pretrained_ckpt: {pretrained_ckpt}")
            ckpt = torch.load(pretrained_ckpt, map_location='cpu')
            if ckpt_key is not None:
                if ckpt_key not in ckpt:
                    raise KeyError(f"ckpt_key='{ckpt_key}' 不在 ckpt 中，可用 keys: {list(ckpt.keys())[:20]}")
                state_dict = ckpt[ckpt_key]
            else:
                state_dict = ckpt

            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print(f"⚠️ 本地权重加载为 strict=False: missing={len(missing)}, unexpected={len(unexpected)}")

        # MambaOut 的 features_only 输出可能是 NHWC（B,H,W,C），这里做鲁棒判断并统一成 NCHW
        def _is_nhwc(t: torch.Tensor) -> bool:
            return t.ndim == 4 and t.shape[1] == t.shape[2]

        try:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                feats = self.backbone(dummy)
                f2, f3, f4 = feats
                if _is_nhwc(f2):
                    ch2, ch3, ch4 = f2.size(-1), f3.size(-1), f4.size(-1)
                else:
                    ch2, ch3, ch4 = f2.size(1), f3.size(1), f4.size(1)
        except Exception:
            # 兜底（不同版本/变体可能变化）
            ch2, ch3, ch4 = 192, 384, 576

        print(f">>> MambaOut channels: {ch2}, {ch3}, {ch4}")

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

    def _to_nchw(self, t):
        # NHWC(B,H,W,C) -> NCHW(B,C,H,W)
        if t.ndim == 4 and t.shape[1] == t.shape[2]:
            return t.permute(0, 3, 1, 2).contiguous()
        return t

    def forward_single_branch(self, x):
        feats = self.backbone(x)
        f2, f3, f4 = feats
        f2 = self._to_nchw(f2)
        f3 = self._to_nchw(f3)
        f4 = self._to_nchw(f4)
        f2 = self.ca2(f2)
        f3 = self.ca3(f3)
        f4 = self.ca4(f4)
        return f2, f3, f4

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


# ================= 5. 训练逻辑（与 train_dinov3.py 完全一致） =================
def train_one_experiment(config):
    device = config['device']

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

    name_parts.append(config.get('mambaout_variant', 'mambaout_base'))

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

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        AlbumentationsTransform(is_train=True, use_clahe=config['use_clahe']),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        AlbumentationsTransform(is_train=False, use_clahe=config['use_clahe']),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(
        ImageFolder(config['train_dir'], transform=train_transform),
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
    )

    val_loader = DataLoader(
        ImageFolder(config['val_dir'], transform=val_transform),
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
    )

    model = BrainTumorMambaOut2025(
        num_classes=config['num_classes'],
        use_dcn=config['use_dcn'],
        use_ca=config['use_ca'],
        use_symmetry=config['use_symmetry'],
        variant=config.get('mambaout_variant', 'mambaout_base'),
        pretrained=config.get('mambaout_pretrained', True),
        pretrained_ckpt=config.get('mambaout_pretrained_ckpt', None),
        ckpt_key=config.get('mambaout_ckpt_key', None),
        hf_cache_dir=config.get('hf_cache_dir', None),
    ).to(device)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    flops_g, params_m, _, _ = measure_efficiency(model, device)
    print(f"📊 模型分析: Params={params_m:.2f}M, GFLOPs={flops_g:.2f}G")

    criterion = FocalLoss(gamma=2.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_macro_f1 = 0.0
    train_start_time = time.time()

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0

        for i, (imgs, targets) in enumerate(train_loader):
            imgs, targets = imgs.to(device), targets.to(device)

            do_fmix = config['use_fmix'] and (np.random.rand() > 0.5)
            if do_fmix:
                imgs, targets_a, targets_b, lam = fmix_data(imgs, targets, device=device)
                outputs = model(imgs)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                outputs = model(imgs)
                loss = criterion(outputs, targets)

            optimizer.zero_grad()
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
                imgs, targets = imgs.to(device), targets.to(device)
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
            _, _, fps, latency = measure_efficiency(model, device)

            with open(os.path.join(current_save_dir, txt_filename), "w", encoding="utf-8") as f:
                f.write(f"Experiment:       {exp_name}\n")
                f.write(f"Backbone:         {getattr(model, 'backbone_name', 'mambaout')}\n")
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
        {'use_symmetry': False, 'use_clahe': False, 'use_fmix': False, 'use_dcn': False, 'use_ca': False}
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
