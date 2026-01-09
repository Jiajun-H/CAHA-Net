"""visualize_gradcam_comparison.py

Grad-CAM 可视化对比脚本
========================
主图固定输出三列：
    (a) 原始图像
    (b) DenseNet
    (c) CMCD-Net

CLAHE 预处理效果单独输出一张图（不混入三列对比图）。

参考文献:
        Selvaraju R R, Cogswell M, Das A, et al.
        Grad-CAM: Visual explanations from deep networks via gradient-based localization[C]
        ICCV 2017.
"""

import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import albumentations as A
from model_utils import BrainTumorFinalNet

# ================= 配置区域 =================
# 模型路径
# (b) DenseNet
DENSENET_MODEL_PATH = './checkpoints_ablation/PureBase/PureBase_best_model.pth'
# (c) CMCD-Net
CMCDNET_MODEL_PATH = './checkpoints_ablation/PureBase_CLAHE_FMix_CA/PureBase_CLAHE_FMix_CA_best_model.pth'

# 类别名称
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# 输出目录
OUTPUT_DIR = './gradcam_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 测试图片列表 (每个类别选择代表性图片)
TEST_IMAGES = [
    ('data/val/glioma/100820_10_751.jpg', 'glioma'),
    ('data/val/meningioma/101801_1_471.jpg', 'meningioma'),
    ('data/val/no_tumor/IXI024_Axial_52_no_tumor.jpg', 'no_tumor'),
    ('data/val/pituitary/103478_10_1500.jpg', 'pituitary'),
]
# ============================================


def get_clahe_transform():
    """获取CLAHE变换"""
    return A.Compose([A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0)])


def load_model(model_path, model_config, device):
    """加载模型并清洗权重"""
    if not os.path.exists(model_path):
        abs_path = os.path.abspath(model_path)
        raise FileNotFoundError(
            "未找到模型权重文件：\n"
            f"  给定路径: {model_path}\n"
            f"  绝对路径: {abs_path}\n\n"
            "这是可视化脚本，不需要重新训练；但你需要把已训练好的 .pth 放到该路径，"
            "或在脚本顶部修改 DENSENET_MODEL_PATH / CMCDNET_MODEL_PATH 指向实际权重文件。"
        )

    print(f"📥 正在加载模型: {model_path}")
    print(f"   配置: {model_config}")
    
    # 初始化模型
    model = BrainTumorFinalNet(**model_config).to(device)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 获取state_dict
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        # 可能是完整模型
        return checkpoint.to(device).eval()
    
    # 清洗state_dict
    clean_state_dict = {}
    for k, v in state_dict.items():
        if "total_ops" not in k and "total_params" not in k:
            new_key = k.replace("module.", "")
            clean_state_dict[new_key] = v
    
    # 加载权重
    msg = model.load_state_dict(clean_state_dict, strict=False)
    if len(msg.missing_keys) > 0:
        print(f"⚠️ 缺失的权重键: {msg.missing_keys[:5]}...")
    
    model.eval()
    print("✅ 模型加载成功")
    return model


def generate_gradcam(model, input_tensor, target_layers, device):
    """生成Grad-CAM热力图"""
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # 获取预测结果
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred_idx = torch.max(probs, 1)
    
    # 生成热力图 (targets=None表示使用预测的最高概率类别)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]
    
    # 释放CAM资源
    del cam
    
    return grayscale_cam, pred_idx.item(), conf.item()


def get_target_layer(model, use_ca=False):
    """
    获取Grad-CAM的目标层
    对于DenseNet，使用最后一个DenseBlock (block4) 作为目标层
    这样可以获得更准确的空间激活信息
    """
    # 使用block4作为目标层，它是最后一个密集块，保留了最好的空间信息
    return [model.block4]


def _save_clahe_effect_figure(img_np, img_clahe, true_label, output_prefix):
    """单独保存 CLAHE 效果对比图（原图 vs CLAHE）。"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f'CLAHE Effect | True Label: {true_label}', fontsize=13, fontweight='bold')

    axes[0].imshow(img_np)
    axes[0].set_title('(a) Original', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(img_clahe)
    axes[1].set_title('(b) CLAHE', fontsize=12)
    axes[1].axis('off')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f'{output_prefix}_clahe_effect.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"💾 CLAHE效果图已保存: {save_path}")


def visualize_comparison(image_path, true_label, densenet_model, cmcdnet_model, device, output_prefix):
    """对比可视化 DenseNet vs CMCD-Net 的 Grad-CAM（主图三列）。"""
    print(f"\n{'='*60}")
    print(f"🖼️  处理图片: {image_path}")
    print(f"   真实标签: {true_label}")
    
    # 读取图片
    img_pil = Image.open(image_path).convert('RGB').resize((224, 224))
    img_np = np.array(img_pil)
    img_float = np.float32(img_np) / 255.0
    
    # CLAHE预处理后的图片（单独保存效果图）
    clahe_transform = get_clahe_transform()
    img_clahe = clahe_transform(image=img_np)['image']
    img_clahe_float = np.float32(img_clahe) / 255.0
    _save_clahe_effect_figure(img_np, img_clahe, true_label, output_prefix)
    
    # 预处理 - DenseNet 使用原图
    input_densenet = preprocess_image(
        img_float,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ).to(device)
    
    # 预处理 - CMCD-Net：训练时包含 CLAHE，这里用 CLAHE 图做输入
    input_cmcd = preprocess_image(
        img_clahe_float,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ).to(device)
    
    # 设置目标层 - 使用block4（最后一个DenseBlock）作为目标层
    # block4输出的是真正的特征图，比norm5更能反映模型关注的区域
    target_layers_densenet = get_target_layer(densenet_model, use_ca=False)
    target_layers_cmcd = get_target_layer(cmcdnet_model, use_ca=True)
    
    # 生成Grad-CAM
    cam_densenet, pred_densenet, conf_densenet = generate_gradcam(
        densenet_model, input_densenet, target_layers_densenet, device
    )
    cam_cmcd, pred_cmcd, conf_cmcd = generate_gradcam(
        cmcdnet_model, input_cmcd, target_layers_cmcd, device
    )
    
    # 叠加热力图到原图
    # 主对比图统一叠加在“原图”上，便于直观看出关注区域差异
    vis_densenet = show_cam_on_image(img_float, cam_densenet, use_rgb=True)
    vis_cmcd = show_cam_on_image(img_float, cam_cmcd, use_rgb=True)
    
    # 打印预测结果
    print(f"   DenseNet预测: {CLASS_NAMES[pred_densenet]} (置信度: {conf_densenet:.4f})")
    print(f"   CMCD-Net预测: {CLASS_NAMES[pred_cmcd]} (置信度: {conf_cmcd:.4f})")
    
    # ============== 绘图 ==============
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Grad-CAM Comparison | True Label: {true_label}', fontsize=14, fontweight='bold')
    
    # 原始图
    axes[0].imshow(img_np)
    axes[0].set_title('(a) Original Image', fontsize=12)
    axes[0].axis('off')
    
    # PureBase 热力图叠加
    axes[1].imshow(vis_densenet)
    pred_str = f"Pred: {CLASS_NAMES[pred_densenet]} ({conf_densenet:.2f})"
    color = 'green' if CLASS_NAMES[pred_densenet] == true_label else 'red'
    axes[1].set_title(f'(b) DenseNet\n{pred_str}', fontsize=12, color=color)
    axes[1].axis('off')
    
    # PureBase_CLAHE_FMix_CA 热力图叠加
    axes[2].imshow(vis_cmcd)
    pred_str = f"Pred: {CLASS_NAMES[pred_cmcd]} ({conf_cmcd:.2f})"
    color = 'green' if CLASS_NAMES[pred_cmcd] == true_label else 'red'
    axes[2].set_title(f'(c) CMCD-Net\n{pred_str}', fontsize=12, color=color)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(OUTPUT_DIR, f'{output_prefix}_gradcam_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"💾 已保存: {save_path}")
    
    return {
        'image_path': image_path,
        'true_label': true_label,
        'densenet_pred': CLASS_NAMES[pred_densenet],
        'densenet_conf': conf_densenet,
        'cmcd_pred': CLASS_NAMES[pred_cmcd],
        'cmcd_conf': conf_cmcd,
    }


def create_summary_figure(results, densenet_model, cmcdnet_model, device):
    """创建汇总对比图（固定三列：原图 / DenseNet / CMCD-Net）。"""
    print("\n" + "="*60)
    print("📊 生成汇总对比图...")
    
    n_samples = len(results)
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle('Grad-CAM Comparison: DenseNet vs CMCD-Net\n(Selvaraju et al., 2017)',
                 fontsize=14, fontweight='bold', y=1.02)

    clahe_transform = get_clahe_transform()
    
    for idx, result in enumerate(results):
        image_path = result['image_path']
        true_label = result['true_label']
        
        # 读取图片
        img_pil = Image.open(image_path).convert('RGB').resize((224, 224))
        img_np = np.array(img_pil)
        img_float = np.float32(img_np) / 255.0
        
        img_clahe = clahe_transform(image=img_np)['image']
        img_clahe_float = np.float32(img_clahe) / 255.0
        
        # 预处理
        input_densenet = preprocess_image(
            img_float, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ).to(device)
        input_cmcd = preprocess_image(
            img_clahe_float, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ).to(device)
        
        # 生成CAM - 使用block4作为目标层
        cam_densenet, _, _ = generate_gradcam(
            densenet_model, input_densenet, get_target_layer(densenet_model), device
        )
        cam_cmcd, _, _ = generate_gradcam(
            cmcdnet_model, input_cmcd, get_target_layer(cmcdnet_model), device
        )

        vis_densenet = show_cam_on_image(img_float, cam_densenet, use_rgb=True)
        vis_cmcd = show_cam_on_image(img_float, cam_cmcd, use_rgb=True)
        
        # 绘图
        ax_row = axes[idx]
        
        # 原图
        ax_row[0].imshow(img_np)
        ax_row[0].set_title(f'(a) Original ({true_label})', fontsize=11)
        ax_row[0].axis('off')
        
        # PureBase CAM
        ax_row[1].imshow(vis_densenet)
        color = 'green' if result['densenet_pred'] == true_label else 'red'
        ax_row[1].set_title(f"(b) DenseNet\n{result['densenet_pred']} ({result['densenet_conf']:.2f})",
                   fontsize=11, color=color)
        ax_row[1].axis('off')
        
        # CLAHE+FMix+CA CAM
        ax_row[2].imshow(vis_cmcd)
        color = 'green' if result['cmcd_pred'] == true_label else 'red'
        ax_row[2].set_title(f"(c) CMCD-Net\n{result['cmcd_pred']} ({result['cmcd_conf']:.2f})",
                   fontsize=11, color=color)
        ax_row[2].axis('off')
    
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, 'gradcam_summary_comparison.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"💾 汇总图已保存: {save_path}")


def main():
    """主函数"""
    print("="*60)
    print("🔬 Grad-CAM可视化对比分析")
    print("   (b) DenseNet vs (c) CMCD-Net")
    print("   基于 Selvaraju et al., ICCV 2017")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")
    
    # 模型配置
    purebase_config = {
        'num_classes': 4,
        'use_dcn': False,
        'use_ca': False,
        'use_symmetry': False
    }
    
    clahe_ca_config = {
        'num_classes': 4,
        'use_dcn': False,
        'use_ca': True,  # 启用CA
        'use_symmetry': False
    }
    
    # 加载模型
    print("\n📦 加载模型...")
    densenet_model = load_model(DENSENET_MODEL_PATH, purebase_config, device)
    cmcdnet_model = load_model(CMCDNET_MODEL_PATH, clahe_ca_config, device)
    
    # 检查测试图片是否存在
    valid_images = []
    for img_path, label in TEST_IMAGES:
        if os.path.exists(img_path):
            valid_images.append((img_path, label))
        else:
            print(f"⚠️ 图片不存在: {img_path}")
    
    # 如果没有有效图片，从验证集中自动选择
    if len(valid_images) == 0:
        print("\n🔍 自动从验证集选择图片...")
        val_dir = './data/val'
        for cls_name in CLASS_NAMES:
            cls_dir = os.path.join(val_dir, cls_name)
            if os.path.exists(cls_dir):
                files = [f for f in os.listdir(cls_dir) if f.endswith(('.jpg', '.tif', '.png'))]
                if files:
                    valid_images.append((os.path.join(cls_dir, files[0]), cls_name))
    
    if len(valid_images) == 0:
        print("❌ 没有找到有效的测试图片！")
        return
    
    print(f"\n✅ 找到 {len(valid_images)} 张测试图片")
    
    # 对每张图片生成对比可视化
    results = []
    for img_path, label in valid_images:
        prefix = os.path.splitext(os.path.basename(img_path))[0]
        result = visualize_comparison(
            img_path, label, 
            densenet_model, cmcdnet_model,
            device, prefix
        )
        results.append(result)
    
    # 生成汇总对比图
    create_summary_figure(results, densenet_model, cmcdnet_model, device)
    
    # 打印结果统计
    print("\n" + "="*60)
    print("📈 结果统计")
    print("="*60)
    
    densenet_correct = sum(1 for r in results if r['densenet_pred'] == r['true_label'])
    cmcd_correct = sum(1 for r in results if r['cmcd_pred'] == r['true_label'])
    
    print(f"DenseNet 正确率: {densenet_correct}/{len(results)}")
    print(f"CMCD-Net 正确率: {cmcd_correct}/{len(results)}")
    print(f"\n📁 所有结果已保存到: {OUTPUT_DIR}")
    print("✅ 完成!")


if __name__ == '__main__':
    main()
