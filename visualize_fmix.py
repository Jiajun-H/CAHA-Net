"""
FMix数据增强可视化脚本
=============================
FMix是一种基于傅里叶空间低频掩码的混合数据增强方法。
它通过在频域生成低频掩码，将两张图像进行混合，生成具有自然过渡边界的增强样本。

参考文献:
    Harris E, Marcu A, Sherrill M, et al.
    FMix: Enhancing mixed sample data augmentation[J].
    arXiv preprint arXiv:2002.12047, 2020.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from model_utils import BrainTumorFinalNet

# ================= 配置区域 =================
# 类别名称
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# 模型路径 (使用训练好的模型)
MODEL_PATH = './checkpoints_ablation/PureBase_CLAHE_FMix_CA/PureBase_CLAHE_FMix_CA_best_model.pth'
MODEL_CONFIG = {
    'num_classes': 4,
    'use_dcn': False,
    'use_ca': True,
    'use_symmetry': False
}

# 输出目录
OUTPUT_DIR = './fmix_visualization'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 示例图片
SAMPLE_IMAGES = [
    ('data/val/glioma/100820_10_751.jpg', 'glioma'),
    ('data/val/meningioma/101801_1_471.jpg', 'meningioma'),
    ('data/val/no_tumor/IXI024_Axial_52_no_tumor.jpg', 'no_tumor'),
    ('data/val/pituitary/103478_10_1500.jpg', 'pituitary'),
]
# ============================================


# ================= 模型加载函数 =================
def load_model(model_path, model_config, device):
    """加载训练好的模型"""
    print(f"📥 正在加载模型: {model_path}")
    
    model = BrainTumorFinalNet(**model_config).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        return checkpoint.to(device).eval()
    
    # 清洗state_dict
    clean_state_dict = {}
    for k, v in state_dict.items():
        if "total_ops" not in k and "total_params" not in k:
            new_key = k.replace("module.", "")
            clean_state_dict[new_key] = v
    
    model.load_state_dict(clean_state_dict, strict=False)
    model.eval()
    print("✅ 模型加载成功")
    return model


def predict_image(model, img_np, device):
    """对图像进行预测"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_pil = Image.fromarray(img_np.astype(np.uint8))
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        conf, pred_idx = torch.max(probs, 1)
    
    return CLASS_NAMES[pred_idx.item()], conf.item(), probs[0].cpu().numpy()


# ================= FMix核心函数 (与train_album.py完全相同) =================
def fftfreqnd(h, w=None, z=None):
    """生成频率网格 - 原始训练脚本版本"""
    fz = fx = 0
    fy = np.fft.fftfreq(h)
    if w is not None: 
        fx = np.fft.fftfreq(w)
    if z is not None: 
        fz = np.fft.fftfreq(z)
    return np.meshgrid(fy, fx, indexing='ij')


def get_spectrum(freq_space, decay_power=2):
    """生成频域谱 - 原始训练脚本版本"""
    scale = np.ones(1) / (np.maximum(freq_space, np.array([1. / max(freq_space.shape)])) ** decay_power)
    param_size = [len(freq_space)] + list(freq_space.shape)
    param = np.random.randn(*param_size)
    return np.expand_dims(scale, axis=0) * param


def make_low_freq_image(decay, shape, ch=1):
    """生成低频掩码 - 原始训练脚本版本"""
    freq_space = fftfreqnd(shape[0], shape[1])
    spectrum = get_spectrum(np.array(freq_space), decay_power=decay)
    mask = np.real(np.fft.ifft2(spectrum[:1])).astype(np.float32)
    mask = mask[0, 0] if mask.ndim == 4 else mask[0]
    if mask.ndim > 2: 
        mask = mask[0]
    mask = mask - mask.min()
    return mask / mask.max()


def fmix_images(img1, img2, alpha=1.0, decay_power=3.0):
    """
    对两张图像执行FMix混合
    
    参数:
        img1: 第一张图像 (H, W, C), numpy array, 值范围[0, 255]
        img2: 第二张图像 (H, W, C), numpy array, 值范围[0, 255]
        alpha: Beta分布参数，控制混合比例λ
        decay_power: 频率衰减系数
    
    返回:
        mixed_img: 混合后的图像
        soft_mask: 原始连续掩码
        binary_mask: 二值化后的掩码
        lam: 混合比例
    """
    assert img1.shape == img2.shape, "两张图像尺寸必须相同"
    h, w = img1.shape[:2]
    
    # 1. 从Beta分布采样混合比例λ
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    
    # 2. 生成低频软掩码
    soft_mask = make_low_freq_image(decay_power, (h, w))
    
    # 3. 根据λ将软掩码二值化
    mask_flat = soft_mask.flatten()
    idx = int((1 - lam) * len(mask_flat))
    threshold = np.partition(mask_flat, idx)[idx]
    binary_mask = (soft_mask > threshold).astype(np.float32)
    
    # 4. 使用二值掩码混合图像
    binary_mask_3d = binary_mask[:, :, np.newaxis]
    mixed_img = img1 * binary_mask_3d + img2 * (1 - binary_mask_3d)
    mixed_img = mixed_img.astype(np.uint8)
    
    # 计算实际混合比例
    actual_lam = binary_mask.mean()
    
    return mixed_img, soft_mask, binary_mask, actual_lam


def visualize_fmix_process(img1, img2, label1, label2, output_path, 
                           decay_power=3.0, alpha=1.0, model=None, device=None):
    """
    可视化FMix的完整处理过程
    展示：原图1、原图2、软掩码、二值掩码、混合结果、模型预测
    """
    # 执行FMix
    mixed_img, soft_mask, binary_mask, lam = fmix_images(
        img1, img2, alpha=alpha, decay_power=decay_power
    )
    
    # 模型预测
    pred_info = ""
    if model is not None and device is not None:
        pred1, conf1, _ = predict_image(model, img1, device)
        pred2, conf2, _ = predict_image(model, img2, device)
        pred_mix, conf_mix, probs_mix = predict_image(model, mixed_img, device)
        pred_info = f"\nModel Predictions: A→{pred1}({conf1:.2f}), B→{pred2}({conf2:.2f}), Mix→{pred_mix}({conf_mix:.2f})"
    
    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'FMix Data Augmentation Visualization\n'
                 f'(Harris et al., 2020) | decay_power={decay_power}, λ={lam:.3f}{pred_info}', 
                 fontsize=13, fontweight='bold')
    
    # 第一行
    axes[0, 0].imshow(img1)
    title_a = f'Image A\n({label1})'
    if model is not None:
        color_a = 'green' if pred1 == label1 else 'red'
        title_a += f'\nPred: {pred1} ({conf1:.2f})'
        axes[0, 0].set_title(title_a, fontsize=12, color=color_a)
    else:
        axes[0, 0].set_title(title_a, fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img2)
    title_b = f'Image B\n({label2})'
    if model is not None:
        color_b = 'green' if pred2 == label2 else 'red'
        title_b += f'\nPred: {pred2} ({conf2:.2f})'
        axes[0, 1].set_title(title_b, fontsize=12, color=color_b)
    else:
        axes[0, 1].set_title(title_b, fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(mixed_img)
    title_mix = f'FMix Result\nA×{lam:.2f} + B×{1-lam:.2f}'
    if model is not None:
        title_mix += f'\nPred: {pred_mix} ({conf_mix:.2f})'
    axes[0, 2].set_title(title_mix, fontsize=12)
    axes[0, 2].axis('off')
    
    # 第二行
    im_soft = axes[1, 0].imshow(soft_mask, cmap='viridis')
    axes[1, 0].set_title('Soft Mask (Low-Freq Image)\nGenerated via FFT', fontsize=12)
    axes[1, 0].axis('off')
    plt.colorbar(im_soft, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    axes[1, 1].imshow(binary_mask, cmap='gray')
    axes[1, 1].set_title(f'Binary Mask\nThreshold by λ={lam:.3f}', fontsize=12)
    axes[1, 1].axis('off')
    
    # 显示掩码叠加在混合图上
    overlay = mixed_img.copy().astype(np.float32)
    # 用红色边界标出掩码边缘
    from scipy import ndimage
    edges = ndimage.sobel(binary_mask)
    edges = (np.abs(edges) > 0.1).astype(np.float32)
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] + edges * 200, 0, 255)
    axes[1, 2].imshow(overlay.astype(np.uint8))
    axes[1, 2].set_title('Mixed Image with\nMask Boundary (Red)', fontsize=12)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"💾 已保存: {output_path}")
    return lam


def visualize_decay_power_comparison(img1, img2, label1, label2, output_path):
    """
    展示不同decay_power参数对掩码的影响
    decay_power越大，生成的掩码越平滑（低频成分越多）
    """
    decay_powers = [1.0, 2.0, 3.0, 5.0]
    
    fig, axes = plt.subplots(len(decay_powers), 4, figsize=(16, 4*len(decay_powers)))
    fig.suptitle('FMix: Effect of Decay Power on Mask Smoothness\n'
                 'Higher decay_power → Smoother boundaries (more low-frequency)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    np.random.seed(42)  # 固定随机种子以便对比
    
    for i, dp in enumerate(decay_powers):
        np.random.seed(42)  # 每次重置，保证只有decay_power不同
        mixed_img, soft_mask, binary_mask, lam = fmix_images(
            img1, img2, alpha=1.0, decay_power=dp
        )
        
        axes[i, 0].imshow(soft_mask, cmap='viridis')
        axes[i, 0].set_title(f'Soft Mask\ndecay_power={dp}', fontsize=11)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(binary_mask, cmap='gray')
        axes[i, 1].set_title(f'Binary Mask\nλ={lam:.3f}', fontsize=11)
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(mixed_img)
        axes[i, 2].set_title(f'FMix Result', fontsize=11)
        axes[i, 2].axis('off')
        
        # FFT频谱可视化
        spectrum = np.abs(np.fft.fftshift(np.fft.fft2(soft_mask)))
        spectrum_log = np.log1p(spectrum)
        axes[i, 3].imshow(spectrum_log, cmap='hot')
        axes[i, 3].set_title(f'Frequency Spectrum\n(log scale)', fontsize=11)
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"💾 已保存: {output_path}")


def visualize_multiple_samples(images_data, output_path, model=None, device=None):
    """
    展示多组FMix样本（带模型预测）
    """
    n = len(images_data) // 2
    if n < 2:
        n = 2
    
    fig, axes = plt.subplots(n, 5, figsize=(20, 4*n))
    fig.suptitle('FMix Augmentation Examples on Brain Tumor MRI\n'
                 '(Harris et al., 2020)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    for i in range(n):
        # 选择两张不同的图片进行混合
        idx1 = i % len(images_data)
        idx2 = (i + 1) % len(images_data)
        
        img1_path, label1 = images_data[idx1]
        img2_path, label2 = images_data[idx2]
        
        if not os.path.exists(img1_path) or not os.path.exists(img2_path):
            continue
        
        img1 = np.array(Image.open(img1_path).convert('RGB').resize((224, 224)))
        img2 = np.array(Image.open(img2_path).convert('RGB').resize((224, 224)))
        
        mixed_img, soft_mask, binary_mask, lam = fmix_images(
            img1, img2, alpha=1.0, decay_power=3.0
        )
        
        # 模型预测
        pred_mix_str = ""
        if model is not None and device is not None:
            pred_mix, conf_mix, _ = predict_image(model, mixed_img, device)
            pred_mix_str = f"\nPred: {pred_mix} ({conf_mix:.2f})"
        
        axes[i, 0].imshow(img1)
        axes[i, 0].set_title(f'Image A\n({label1})', fontsize=11)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(img2)
        axes[i, 1].set_title(f'Image B\n({label2})', fontsize=11)
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(binary_mask, cmap='gray')
        axes[i, 2].set_title(f'Binary Mask\nλ={lam:.3f}', fontsize=11)
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(mixed_img)
        axes[i, 3].set_title(f'FMix Result{pred_mix_str}', fontsize=11)
        axes[i, 3].axis('off')
        
        # 标签信息
        axes[i, 4].text(0.5, 0.5, 
                        f"Mix Info:\n\n"
                        f"Label A: {label1}\n"
                        f"Label B: {label2}\n\n"
                        f"λ = {lam:.3f}\n\n"
                        f"Loss:\n"
                        f"λ×L(A) + (1-λ)×L(B)",
                        ha='center', va='center', fontsize=12,
                        transform=axes[i, 4].transAxes,
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[i, 4].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"💾 已保存: {output_path}")


def visualize_fmix_vs_cutmix(img1, img2, label1, label2, output_path):
    """
    对比FMix和CutMix的区别
    """
    h, w = img1.shape[:2]
    lam = 0.5
    
    # FMix
    mixed_fmix, soft_mask, binary_mask_fmix, actual_lam = fmix_images(
        img1, img2, alpha=1.0, decay_power=3.0
    )
    
    # CutMix (简单矩形裁剪)
    cut_ratio = np.sqrt(1 - lam)
    cut_w = int(w * cut_ratio)
    cut_h = int(h * cut_ratio)
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    x1 = np.clip(cx - cut_w // 2, 0, w)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    
    binary_mask_cutmix = np.ones((h, w), dtype=np.float32)
    binary_mask_cutmix[y1:y2, x1:x2] = 0
    
    mixed_cutmix = img1.copy()
    mixed_cutmix[y1:y2, x1:x2] = img2[y1:y2, x1:x2]
    
    # 绘图
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('FMix vs CutMix Comparison\n'
                 'FMix uses low-frequency masks for natural boundaries', 
                 fontsize=14, fontweight='bold')
    
    # 原图
    axes[0, 0].imshow(img1)
    axes[0, 0].set_title(f'Image A ({label1})', fontsize=11)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img2)
    axes[0, 1].set_title(f'Image B ({label2})', fontsize=11)
    axes[0, 1].axis('off')
    
    # CutMix
    axes[0, 2].imshow(binary_mask_cutmix, cmap='gray')
    axes[0, 2].set_title('CutMix Mask\n(Rectangular)', fontsize=11)
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(mixed_cutmix)
    axes[0, 3].set_title('CutMix Result\n(Sharp edges)', fontsize=11)
    axes[0, 3].axis('off')
    
    # FMix
    axes[1, 0].imshow(soft_mask, cmap='viridis')
    axes[1, 0].set_title('FMix Soft Mask\n(Low-frequency)', fontsize=11)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(binary_mask_fmix, cmap='gray')
    axes[1, 1].set_title(f'FMix Binary Mask\nλ={actual_lam:.3f}', fontsize=11)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(mixed_fmix)
    axes[1, 2].set_title('FMix Result\n(Natural boundaries)', fontsize=11)
    axes[1, 2].axis('off')
    
    # 对比说明
    axes[1, 3].text(0.5, 0.5, 
                    "Key Differences:\n\n"
                    "CutMix:\n"
                    "• Rectangular masks\n"
                    "• Sharp, unnatural edges\n"
                    "• Simple spatial mixing\n\n"
                    "FMix:\n"
                    "• Fourier-based masks\n"
                    "• Smooth, organic shapes\n"
                    "• More realistic augmentation",
                    ha='center', va='center', fontsize=11,
                    transform=axes[1, 3].transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"💾 已保存: {output_path}")


def main():
    """主函数"""
    print("="*60)
    print("🎨 FMix数据增强可视化")
    print("   基于 Harris et al., arXiv 2020")
    print("="*60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")
    
    # 加载训练好的模型
    model = None
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH, MODEL_CONFIG, device)
    else:
        print(f"⚠️ 模型文件不存在: {MODEL_PATH}，将不显示预测结果")
    
    # 加载示例图片
    valid_images = []
    for img_path, label in SAMPLE_IMAGES:
        if os.path.exists(img_path):
            valid_images.append((img_path, label))
        else:
            print(f"⚠️ 图片不存在: {img_path}")
    
    if len(valid_images) < 2:
        # 自动从验证集选择
        print("\n🔍 自动从验证集选择图片...")
        val_dir = './data/val'
        for cls_name in CLASS_NAMES:
            cls_dir = os.path.join(val_dir, cls_name)
            if os.path.exists(cls_dir):
                files = [f for f in os.listdir(cls_dir) 
                        if f.endswith(('.jpg', '.tif', '.png'))]
                if files:
                    valid_images.append((os.path.join(cls_dir, files[0]), cls_name))
    
    if len(valid_images) < 2:
        print("❌ 需要至少2张图片来演示FMix!")
        return
    
    print(f"\n✅ 找到 {len(valid_images)} 张图片")
    
    # 加载两张示例图片
    img1_path, label1 = valid_images[0]
    img2_path, label2 = valid_images[1]
    
    img1 = np.array(Image.open(img1_path).convert('RGB').resize((224, 224)))
    img2 = np.array(Image.open(img2_path).convert('RGB').resize((224, 224)))
    
    print(f"\n📷 图片A: {label1}")
    print(f"📷 图片B: {label2}")
    
    # 1. FMix完整过程可视化 (带模型预测)
    print("\n[1/4] 生成FMix处理过程可视化...")
    visualize_fmix_process(
        img1, img2, label1, label2,
        os.path.join(OUTPUT_DIR, 'fmix_process.png'),
        decay_power=3.0,
        model=model, device=device
    )
    
    # 2. decay_power参数对比
    print("[2/4] 生成decay_power参数对比...")
    visualize_decay_power_comparison(
        img1, img2, label1, label2,
        os.path.join(OUTPUT_DIR, 'fmix_decay_power_comparison.png')
    )
    
    # 3. 多组样本展示
    print("[3/4] 生成多组FMix样本...")
    visualize_multiple_samples(
        valid_images,
        os.path.join(OUTPUT_DIR, 'fmix_multiple_samples.png'),
        model=model, device=device
    )
    
    # 4. FMix vs CutMix对比
    print("[4/4] 生成FMix vs CutMix对比...")
    visualize_fmix_vs_cutmix(
        img1, img2, label1, label2,
        os.path.join(OUTPUT_DIR, 'fmix_vs_cutmix.png')
    )
    
    print("\n" + "="*60)
    print(f"📁 所有结果已保存到: {OUTPUT_DIR}")
    print("✅ 完成!")


if __name__ == '__main__':
    main()
