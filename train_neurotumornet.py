"""
NeuroTumorNet 训练脚本
=====================
基于 https://github.com/h9zdev/NeuroTumorNet 的训练流程

输出格式与 train_album.py 一致，包含:
- Clinical Metrics: Macro F1, AUC, Recall, Specificity, FPR
- Computational Efficiency: GFLOPs, Params, FPS, Latency

注意: 本脚本不使用 train_album.py 中的模块，完全独立实现
"""

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

# 导入 NeuroTumorNet 模型
from neurotumornet_classifier import create_neurotumornet

# 计算 FLOPs
try:
    from thop import profile
except ImportError:
    print("❌ 警告: 未安装 'thop' 库。无法计算 GFLOPs。请运行 pip install thop")
    profile = None


# ================= 1. 基础配置 =================
BASE_CONFIG = {
    'train_dir': './data/train',
    'val_dir': './data/val',
    'log_dir': './logs_neurotumornet',
    'save_dir': './checkpoints_neurotumornet',
    'result_dir': './result',  # 结果输出目录
    'num_classes': 4,
    'batch_size': 32,  # 原始仓库使用 batch_size=32
    'lr': 1e-3,  # 原始仓库使用 Adam with lr=0.001
    'epochs': 50,  # 原始仓库使用 20-50 epochs
    'num_workers': 8,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'weight_decay': 1e-3,  # L2 正则化 (原始仓库使用 l2=0.001)
}


# ================= 2. 辅助函数 =================
class FocalLoss(nn.Module):
    """Focal Loss for class imbalance"""
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def measure_efficiency(model, device, input_size=(1, 3, 224, 224)):
    """
    计算 GFLOPs, 参数量, FPS, Latency
    """
    model.eval()
    dummy_input = torch.randn(input_size).to(device)
    
    # 1. 计算 GFLOPs 和 参数量
    flops_g, params_m = 0.0, 0.0
    if profile:
        try:
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
            flops_g = flops / 1e9
            params_m = params / 1e6
        except Exception as e:
            print(f"GFLOPs calculation failed: {e}")
            params_m = sum(p.numel() for p in model.parameters()) / 1e6
    else:
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
    
    # 2. 计算推理速度 (FPS & Latency)
    # 预热
    for _ in range(20):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # 正式测量
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


def compute_metrics(all_labels, all_preds, all_probs, num_classes=4):
    """
    计算临床指标: Macro F1, AUC, Recall, Specificity, FPR
    """
    # 1. 基础指标
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    macro_f1 = report['macro avg']['f1-score']
    macro_recall = report['macro avg']['recall']
    
    # 2. AUC (One-vs-Rest)
    try:
        auc = roc_auc_score(
            np.eye(num_classes)[all_labels],
            all_probs,
            average='macro',
            multi_class='ovr'
        )
    except ValueError:
        auc = 0.0
    
    # 3. 混淆矩阵计算 Specificity 和 FPR
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    specificities = []
    fprs = []
    
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        specificities.append(spec)
        fprs.append(fpr)
    
    macro_spec = np.mean(specificities)
    macro_fpr = np.mean(fprs)
    
    return macro_f1, auc, macro_recall, macro_spec, macro_fpr


# ================= 3. 训练函数 =================
def train_one_experiment(config):
    device = config['device']
    
    # 生成实验名称
    exp_name = f"NeuroTumorNet_{config['variant']}"
    if config.get('use_augmentation', False):
        exp_name += "_Aug"
    
    current_save_dir = os.path.join(config['save_dir'], exp_name)
    current_log_dir = os.path.join(config['log_dir'], exp_name)
    os.makedirs(current_save_dir, exist_ok=True)
    os.makedirs(config['result_dir'], exist_ok=True)
    
    if os.path.exists(current_log_dir):
        shutil.rmtree(current_log_dir)
    writer = SummaryWriter(current_log_dir)
    
    print("\n" + "#" * 60)
    print(f"🧠 开始执行: {exp_name}")
    print("#" * 60 + "\n")
    
    # 数据增强 - 参考原始仓库的 ImageDataGenerator 设置
    if config.get('use_augmentation', False):
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(20),  # rotation_range=20
            transforms.RandomAffine(
                degrees=0,
                translate=(0.2, 0.2),  # width_shift, height_shift
                shear=20,  # shear_range=0.2 (20%)
                scale=(0.8, 1.2)  # zoom_range=0.2
            ),
            transforms.RandomHorizontalFlip(p=0.5),  # horizontal_flip=True
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 数据加载
    train_dataset = ImageFolder(config['train_dir'], transform=train_transform)
    val_dataset = ImageFolder(config['val_dir'], transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # 创建模型
    model = create_neurotumornet(
        variant=config['variant'],
        num_classes=config['num_classes'],
        dropout_rate=config.get('dropout_rate', 0.5)
    ).to(device)
    
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    
    # 评估模型复杂度
    flops_g, params_m, _, _ = measure_efficiency(model, device)
    print(f"📊 模型分析: Params={params_m:.2f}M, GFLOPs={flops_g:.2f}G")
    
    # 损失函数和优化器
    # 原始仓库使用 categorical_crossentropy，这里使用 CrossEntropyLoss
    criterion = nn.CrossEntropyLoss().to(device)
    
    # 原始仓库使用 Adam with lr=0.001, 这里添加 weight_decay 作为 L2 正则化
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # 学习率调度器 - 参考原始仓库的 ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.2,  # factor=0.2
        patience=3,  # patience=3
        min_lr=1e-6  # min_lr=1e-6
    )
    
    # Early Stopping 设置
    patience = 10
    patience_counter = 0
    
    # 训练
    best_f1 = 0.0
    train_start_time = time.time()
    
    for epoch in range(config['epochs']):
        # === 训练阶段 ===
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        
        # === 验证阶段 ===
        model.eval()
        val_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_probs.append(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        
        # 计算指标
        macro_f1, auc, macro_recall, macro_spec, macro_fpr = compute_metrics(
            all_labels, all_preds, all_probs, config['num_classes']
        )
        acc = (all_preds == all_labels).mean()
        
        # 更新学习率
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录到 TensorBoard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        writer.add_scalar('Metrics/Accuracy', acc, epoch)
        writer.add_scalar('Metrics/Macro_F1', macro_f1, epoch)
        writer.add_scalar('Metrics/AUC', auc, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        
        print(f"Epoch [{epoch+1}/{config['epochs']}] "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Acc: {acc:.4f} | F1: {macro_f1:.4f} | AUC: {auc:.4f} | LR: {current_lr:.2e}")
        
        # 保存最佳模型
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            patience_counter = 0
            
            save_path_model = os.path.join(current_save_dir, f"{exp_name}_best.pth")
            torch.save(model.state_dict(), save_path_model)
            
            # 计算训练时长
            total_time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - train_start_time))
            
            # 模型文件大小
            model_size_mb = os.path.getsize(save_path_model) / (1024 * 1024)
            
            # 显存占用
            max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
            
            # 重新测试推理速度
            _, _, fps, latency = measure_efficiency(model, device)
            
            # 写入结果文件 (与 train_album.py 格式一致)
            txt_filename = f"{exp_name}_best_metrics.txt"
            result_path = os.path.join(config['result_dir'], txt_filename)
            
            with open(result_path, "w", encoding="utf-8") as f:
                f.write(f"Experiment:       {exp_name}\n")
                f.write(f"Best Epoch:       {epoch+1}\n")
                f.write("=" * 40 + "\n")
                f.write("--- Clinical Metrics ---\n")
                f.write(f"Accuracy:         {acc:.4f}\n")
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
            
            print(f"✅ [新纪录] F1={macro_f1:.4f} 已保存: {txt_filename}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break
    
    writer.close()
    del model, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()
    print(f"🏁 实验结束: {exp_name}\n")


# ================= 4. 实验队列 =================
def run_all_experiments():
    """
    运行所有 NeuroTumorNet 实验
    
    变体:
    - baseline: 原始3层CNN
    - enhanced: 带增强正则化的CNN
    - deep: 5层深度CNN
    - vgg_style: VGG风格的CNN
    - tiny: 轻量级CNN
    """
    
    experiment_queue = [
        # 1. Baseline (无数据增强)
        {'variant': 'baseline', 'use_augmentation': False, 'dropout_rate': 0.5},
        
        # 2. Baseline (有数据增强) - 模拟原始仓库的 ImageDataGenerator
        {'variant': 'baseline', 'use_augmentation': True, 'dropout_rate': 0.5},
        
        # 3. Enhanced (带增强正则化)
        {'variant': 'enhanced', 'use_augmentation': True, 'dropout_rate': 0.5},
        
        # 4. Deep (5层CNN)
        {'variant': 'deep', 'use_augmentation': True, 'dropout_rate': 0.5},
        
        # 5. VGG-style
        {'variant': 'vgg_style', 'use_augmentation': True, 'dropout_rate': 0.5},
        
        # 6. Tiny (轻量级)
        {'variant': 'tiny', 'use_augmentation': True, 'dropout_rate': 0.3},
    ]
    
    print(f"📋 任务队列: {len(experiment_queue)} 个实验")
    print("=" * 60)
    
    for idx, params in enumerate(experiment_queue):
        cfg = BASE_CONFIG.copy()
        cfg.update(params)
        
        print(f"\n🚀 执行第 {idx+1}/{len(experiment_queue)} 个实验: {params['variant']}")
        
        try:
            train_one_experiment(cfg)
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    run_all_experiments()
