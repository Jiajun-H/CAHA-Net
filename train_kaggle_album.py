import argparse
import os
import time
import gc
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import ImageFolder

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from albumentations import Compose, CLAHE, HorizontalFlip, Rotate
from PIL import Image

from model_utils import BrainTumorFinalNet


# =====================
# 代码内开关（你只需要改这里）
# =====================
# True: 迁移学习微调（会加载 PRETRAINED_CKPT_PATH，再在新数据集上训练）
# False: 从零开始训练（不加载任何权重）
ENABLE_TRANSFER_LEARNING = True

# 论文/本文训练得到的权重（默认指向你提供的权重文件）
PRETRAINED_CKPT_PATH = "./checkpoints_ablation/PureBase_CLAHE_FMix_CA/PureBase_CLAHE_FMix_CA_best_model.pth"


# ====== 与 train_album.py 保持一致的组件 ======
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class AlbumentationsTransform:
    def __init__(self, is_train: bool = True, use_clahe: bool = True):
        aug_list = []
        if use_clahe:
            aug_list.append(CLAHE(clip_limit=4.0, p=1.0))
        if is_train:
            aug_list.extend([Rotate(limit=15, p=0.5), HorizontalFlip(p=0.5)])
        self.aug = Compose(aug_list)

    def __call__(self, img):
        img_np = np.array(img)
        if self.aug:
            return Image.fromarray(self.aug(image=img_np)["image"])
        return Image.fromarray(img_np)


# ====== FMix：与 train_album.py 保持一致 ======
def fftfreqnd(h, w=None, z=None):
    fz = fx = 0
    fy = np.fft.fftfreq(h)
    if w is not None:
        fx = np.fft.fftfreq(w)
    if z is not None:
        fz = np.fft.fftfreq(z)
    return np.meshgrid(fy, fx, indexing="ij")


def get_spectrum(freq_space, decay_power=2):
    scale = np.ones(1) / (np.maximum(freq_space, np.array([1.0 / max(freq_space.shape)])) ** decay_power)
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
    return mask / mask.max()


def fmix_data(data, targets, alpha=1.0, decay_power=3.0, shape=(224, 224), device="cuda"):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    index = torch.randperm(data.size(0)).to(device)
    soft_mask = make_low_freq_image(decay_power, shape)
    mask_flat = soft_mask.flatten()
    idx = int((1 - lam) * len(mask_flat))
    threshold = np.partition(mask_flat, idx)[idx]
    binary_mask = (
        torch.from_numpy((soft_mask > threshold).astype(np.float32))
        .to(device)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    mixed_x = data * binary_mask + data[index] * (1 - binary_mask)
    return mixed_x, targets, targets[index], float(binary_mask.mean().item())


def build_transforms(use_clahe: bool, is_train: bool):
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            AlbumentationsTransform(is_train=is_train, use_clahe=use_clahe),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    return preds.cpu().numpy(), targets.cpu().numpy(), probs.cpu().numpy()


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    num_classes: int,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            total_loss += float(criterion(outputs, targets).item())
            preds, targs, probs = compute_metrics(outputs, targets, num_classes)
            all_preds.extend(preds)
            all_targets.extend(targs)
            all_probs.extend(probs)

    avg_loss = total_loss / max(1, len(loader))
    all_targets_np = np.asarray(all_targets)
    all_preds_np = np.asarray(all_preds)

    acc = float((all_preds_np == all_targets_np).mean()) if len(all_targets_np) else 0.0

    report = classification_report(all_targets, all_preds, output_dict=True, zero_division=0)
    macro_f1 = float(report["macro avg"]["f1-score"]) if "macro avg" in report else 0.0
    macro_recall = float(report["macro avg"]["recall"]) if "macro avg" in report else 0.0

    try:
        auc = float(
            roc_auc_score(
                np.eye(num_classes)[all_targets_np],
                np.asarray(all_probs),
                multi_class="ovr",
                average="macro",
            )
        )
    except Exception:
        auc = 0.0

    cm = confusion_matrix(all_targets, all_preds)
    if cm.size:
        FP = cm.sum(axis=0) - np.diag(cm)
        TN = cm.sum() - (FP + (cm.sum(axis=1) - np.diag(cm)) + np.diag(cm))
        macro_spec = float((TN / (TN + FP + 1e-6)).mean())
        macro_fpr = float((FP / (FP + TN + 1e-6)).mean())
    else:
        macro_spec = 0.0
        macro_fpr = 0.0

    return {
        "loss": float(avg_loss),
        "acc": acc,
        "macro_f1": macro_f1,
        "macro_recall": macro_recall,
        "auc": auc,
        "macro_spec": macro_spec,
        "macro_fpr": macro_fpr,
    }


def load_state_dict_forgiving(model: nn.Module, state_dict: Dict[str, torch.Tensor]):
    """迁移学习用：跳过 shape 不匹配的参数（通常是分类头）。"""
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state_dict.items():
        if k not in model_state:
            continue
        if model_state[k].shape != v.shape:
            skipped.append(k)
            continue
        filtered[k] = v

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return skipped, missing, unexpected


def load_pretrained(model: nn.Module, ckpt_path: Path, device: str) -> None:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    if isinstance(ckpt, dict):
        if isinstance(ckpt.get("state_dict"), dict):
            state = ckpt["state_dict"]
        elif isinstance(ckpt.get("model"), dict):
            state = ckpt["model"]
        else:
            state = ckpt
    else:
        state = ckpt

    skipped, missing, unexpected = load_state_dict_forgiving(model, state)
    print(f"✅ 加载预训练权重: {ckpt_path}")
    if skipped:
        print(f"ℹ️ 跳过 shape 不匹配参数 {len(skipped)} 个（多为分类头）")
    if missing:
        print(f"ℹ️ 未加载参数 {len(missing)} 个")
    if unexpected:
        print(f"ℹ️ 未使用的预训练参数 {len(unexpected)} 个")


def infer_num_classes_from_folder(train_dir: Path) -> int:
    ds = ImageFolder(str(train_dir))
    return len(ds.classes)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train BrainTumorFinalNet on prepared Kaggle ImageFolder datasets (album-consistent).")

    p.add_argument(
        "--data_dir",
        default="./data_kaggle_prepared/Brain_MRI_Images_for_Brain_Tumor_Detection",
        help="Prepared dataset root containing train/val/test",
    )
    # 仍然保留可选参数（如需要临时覆盖），但你不传也能跑。
    p.add_argument(
        "--mode",
        default=None,
        choices=["base", "transfer"],
        help="(可选覆盖) base: train from scratch; transfer: load pretrained weights then finetune. 默认由代码内 ENABLE_TRANSFER_LEARNING 决定",
    )
    p.add_argument(
        "--pretrained_path",
        default=None,
        help="(可选覆盖) transfer learning pretrained .pth path. 默认由代码内 PRETRAINED_CKPT_PATH 决定",
    )

    # 与 train_album.py 的默认一致
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--use_clahe", action="store_true", default=True)
    p.add_argument("--no_clahe", dest="use_clahe", action="store_false")

    # 模型开关（默认对齐 train_album.py 队列里的设置）
    p.add_argument("--use_symmetry", action="store_true", default=False)
    p.add_argument("--use_fmix", action="store_true", default=True)
    p.add_argument("--no_fmix", dest="use_fmix", action="store_false")
    p.add_argument("--use_dcn", action="store_true", default=False)
    p.add_argument("--use_ca", action="store_true", default=True)
    p.add_argument("--no_ca", dest="use_ca", action="store_false")

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_dir", default="./checkpoints_kaggle")
    p.add_argument("--log_dir", default="./logs_kaggle")
    p.add_argument("--exp_name", default=None, help="Optional experiment name")

    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Expected train/val under: {data_dir}")

    num_classes = infer_num_classes_from_folder(train_dir)

    # 计算最终运行模式：命令行参数优先，其次使用代码内开关
    effective_mode = args.mode
    if effective_mode is None:
        effective_mode = "transfer" if ENABLE_TRANSFER_LEARNING else "base"

    effective_pretrained_path = args.pretrained_path or PRETRAINED_CKPT_PATH

    if effective_mode == "transfer":
        if not effective_pretrained_path:
            raise ValueError("transfer 模式需要预训练权重路径，请设置 PRETRAINED_CKPT_PATH 或传 --pretrained_path")
        if not Path(effective_pretrained_path).exists():
            raise FileNotFoundError(f"预训练权重不存在: {effective_pretrained_path}")

    # exp 名称
    if args.exp_name:
        exp_name = args.exp_name
    else:
        exp_name = f"{data_dir.name}_{effective_mode}_PureBase"
        if args.use_clahe:
            exp_name += "_CLAHE"
        if args.use_fmix:
            exp_name += "_FMix"
        if args.use_dcn:
            exp_name += "_DCN"
        if args.use_ca:
            exp_name += "_CA"

    save_dir = Path(args.save_dir) / exp_name
    log_dir = Path(args.log_dir) / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)
    if log_dir.exists():
        shutil_rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(log_dir))

    device = args.device
    print(f"📦 Dataset: {data_dir}")
    print(f"🔧 Mode: {effective_mode} | Classes: {num_classes}")
    print(f"🖥️ Device: {device}")

    train_tf = build_transforms(use_clahe=args.use_clahe, is_train=True)
    eval_tf = build_transforms(use_clahe=args.use_clahe, is_train=False)

    train_loader = DataLoader(
        ImageFolder(str(train_dir), transform=train_tf),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        ImageFolder(str(val_dir), transform=eval_tf),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_loader = None
    if test_dir.exists():
        test_loader = DataLoader(
            ImageFolder(str(test_dir), transform=eval_tf),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    model = BrainTumorFinalNet(
        num_classes=num_classes,
        use_dcn=args.use_dcn,
        use_ca=args.use_ca,
        use_symmetry=args.use_symmetry,
    ).to(device)

    if effective_mode == "transfer":
        load_pretrained(model, Path(effective_pretrained_path), device)

    criterion = FocalLoss(gamma=2.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_acc = -1.0
    best_epoch = -1
    train_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        running = 0.0

        for step, (imgs, targets) in enumerate(train_loader, start=1):
            imgs = imgs.to(device)
            targets = targets.to(device)

            do_fmix = bool(args.use_fmix) and (np.random.rand() > 0.5)
            if do_fmix:
                mixed, targets_a, targets_b, lam = fmix_data(
                    imgs,
                    targets,
                    device=device,
                    shape=(224, 224),
                )
                outputs = model(mixed)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                outputs = model(imgs)
                loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += float(loss.item())
            if step % 20 == 0:
                print(
                    f"\r[{exp_name}] Epoch {epoch+1}/{args.epochs} Step {step}/{len(train_loader)} Loss: {loss.item():.4f}",
                    end="",
                )

        avg_train_loss = running / max(1, len(train_loader))
        writer.add_scalar("Train/Loss", avg_train_loss, epoch)

        val_metrics = evaluate(model, val_loader, criterion, device, num_classes)
        scheduler.step(val_metrics["loss"])

        writer.add_scalar("Val/Loss", val_metrics["loss"], epoch)
        writer.add_scalar("Val/ACC", val_metrics["acc"], epoch)
        writer.add_scalar("Val/MacroF1", val_metrics["macro_f1"], epoch)

        print(
            f"\nVal -> ACC: {val_metrics['acc']:.4f} | F1: {val_metrics['macro_f1']:.4f} | AUC: {val_metrics['auc']:.4f} | Loss: {val_metrics['loss']:.4f}"
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch + 1
            torch.save(model.state_dict(), str(save_dir / "best_model.pth"))

            # 写 best 指标
            elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - train_start))
            with open(save_dir / "best_metrics.txt", "w", encoding="utf-8") as f:
                f.write(f"Experiment: {exp_name}\n")
                f.write(f"Best Epoch: {best_epoch}\n")
                f.write(f"Mode: {effective_mode}\n")
                f.write(f"Classes: {num_classes}\n")
                f.write("=" * 40 + "\n")
                f.write("--- Val Metrics ---\n")
                f.write(f"ACC: {val_metrics['acc']:.4f}\n")
                f.write(f"Macro F1: {val_metrics['macro_f1']:.4f}\n")
                f.write(f"AUC: {val_metrics['auc']:.4f}\n")
                f.write(f"Recall: {val_metrics['macro_recall']:.4f}\n")
                f.write(f"Specificity: {val_metrics['macro_spec']:.4f}\n")
                f.write(f"FPR: {val_metrics['macro_fpr']:.4f}\n")
                f.write(f"Val Loss: {val_metrics['loss']:.4f}\n")

                if test_loader is not None:
                    test_metrics = evaluate(model, test_loader, criterion, device, num_classes)
                    f.write("\n--- Test Metrics ---\n")
                    f.write(f"ACC: {test_metrics['acc']:.4f}\n")
                    f.write(f"Macro F1: {test_metrics['macro_f1']:.4f}\n")
                    f.write(f"AUC: {test_metrics['auc']:.4f}\n")
                    f.write(f"Test Loss: {test_metrics['loss']:.4f}\n")

                f.write("\n--- Runtime ---\n")
                f.write(f"Training Time: {elapsed}\n")

    writer.close()

    # 清理
    del model, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    print(f"✅ Done. Best Val ACC={best_val_acc:.4f} @ epoch {best_epoch}")
    print(f"Saved to: {save_dir.resolve()}")


def shutil_rmtree(p: Path) -> None:
    import shutil

    if p.exists():
        shutil.rmtree(p)


if __name__ == "__main__":
    main()
