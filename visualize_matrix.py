import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import numpy as np
import os

# ================= 配置区域 =================
DATA_DIR = './data' # 你的数据集根目录 (要有 train/val 或 train/test)
CHECKPOINT_PATH = './model_pth/best_model_fastvit_t8.pth'  # 你训练保存的最好的模型路径
MODEL_NAME = 'fastvit_t8.apple_in1k'
BATCH_SIZE = 32
# ===========================================

def plot_confusion_matrix():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # 1. 数据预处理
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. 自动定位验证集/测试集目录
    if os.path.exists(os.path.join(DATA_DIR, 'test')):
        target_dir = os.path.join(DATA_DIR, 'test')
        print("📂 Loading data from: /test")
    elif os.path.exists(os.path.join(DATA_DIR, 'val')):
        target_dir = os.path.join(DATA_DIR, 'val')
        print("📂 Loading data from: /val")
    else:
        print("❌ Error: Could not find 'test' or 'val' folder in dataset directory.")
        return
    
    dataset = datasets.ImageFolder(target_dir, transform=val_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    class_names = dataset.classes
    print(f"✅ Classes found: {class_names}")

    # 3. 加载模型 (修复版) ==========================================
    print(f"🔄 Loading checkpoint from: {CHECKPOINT_PATH}")
    
    # 先加载文件内容
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    # 🔍 智能判断文件内容类型
    if isinstance(checkpoint, torch.nn.Module):
        # 情况A：保存的是整个模型对象
        print("📦 Detected full model object in checkpoint.")
        model = checkpoint
    elif isinstance(checkpoint, dict):
        # 情况B：保存的是权重字典 (State Dict)
        print("🔑 Detected weight dictionary in checkpoint.")
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(class_names))
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        print("❌ Unknown checkpoint format!")
        return
    # ==============================================================
    
    model.to(device)
    model.eval()

    # 4. 预测
    y_true = []
    y_pred = []

    print("⚡ Starting inference...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # 5. 生成混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 6. 绘图
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix - {MODEL_NAME}')
    plt.tight_layout()
    plt.show()

    # 7. 打印详细报告
    print("\n📄 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

if __name__ == '__main__':
    plot_confusion_matrix()