#pip install grad-cam seaborn matplotlib scikit-learn opencv-python

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import timm
from PIL import Image

# ================= 配置区域 =================
MODEL_NAME = 'fastvit_t8.apple_in1k'
CHECKPOINT_PATH = './model_pth/best_model_fastvit_t8.pth'  # 你训练保存的最好的模型路径
IMAGE_PATH = 'data/val/glioma/93329_1_883.jpg' # 替换成你随便找的一张测试集图片路径
NUM_CLASSES = 4
# ===========================================

def get_model():
    print(f"Loading model from: {CHECKPOINT_PATH}")
    
    # 1. 加载文件
    # weights_only=False 是为了兼容旧版保存方式，忽略那个警告即可
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')

    # -----------------------------------------------------------
    # 情况 A: 你加载的是【完整模型对象】 (你现在的训练脚本就是这种情况)
    # -----------------------------------------------------------
    if isinstance(checkpoint, torch.nn.Module):
        print("✅ 检测到加载的是完整模型对象 (Whole Model)")
        model = checkpoint
        model.eval() # 切换到评估模式
        return model

    # -----------------------------------------------------------
    # 情况 B: 你加载的是【参数字典】 (标准的官方写法)
    # -----------------------------------------------------------
    print("📋 检测到加载的是参数字典 (State Dict)")
    
    # 重新构建模型结构 (空壳)
    model = timm.create_model(
        MODEL_NAME, 
        pretrained=False, 
        num_classes=NUM_CLASSES
    )
    
    # 处理字典里的 key
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint # 可能整个字典就是参数
    else:
        state_dict = checkpoint

    # 加载参数
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"⚠️ 直接加载失败，尝试忽略不匹配的键: {e}")
        # 再次尝试，允许有一些不匹配（比如 img_size 导致的头部差异）
        model.load_state_dict(state_dict, strict=False)
        
    model.eval()
    return model


def visualize():
    model = get_model()
    
    # 🎯 关键点：找到 FastViT 的最后一层特征层
    # 对于 timm 的 fastvit，通常是 model.stages[-1]
    target_layers = [model.stages[-1]]

    # 准备图片
    img = np.array(Image.open(IMAGE_PATH).convert('RGB'))
    img = cv2.resize(img, (224, 224))
    rgb_img = np.float32(img) / 255
    
    # 预处理 (标准化需与训练时一致)
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    # 初始化 GradCAM
    cam = GradCAM(model=model, target_layers=target_layers)

    # 生成热力图
    # targets=None 表示自动找概率最大的那一类
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]

    # 将热力图叠加到原图
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # === 绘图 ===
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # 原图
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # 热力图
    axes[1].imshow(visualization)
    axes[1].set_title(f"Grad-CAM Heatmap\nModel: {MODEL_NAME}")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    print("✅ 可视化完成！红色区域表示模型重点关注的地方。")

if __name__ == '__main__':
    visualize()
