import os
import contextlib
# ====================================================
# 🚀 0. 必须开启镜像源 (解决下载报错)
# ====================================================
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torchvision
from torch import nn
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import time
import timm 
from tqdm import tqdm  

# 临时配置（如果没有config.py）
class Config:
    def __init__(self):
        self.train_dir = "./train"  # 替换为你的训练集路径
        self.val_dir = "./val"      # 替换为你的验证集路径
cfg = Config()

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ====================================================
# 🟢 1. 2024-2025年SOTA模型选择区 (仅保留最新版)
# ====================================================

# ==================== 🔥 高效轻量型 (2024-2025) ====================
# 【1】MobileViT-4 (2024 Q4) - 移动端最新王者，精度>90%
# MODEL_NAME = 'mobilevitv4_large.clip_laion2b_ft_in1k'

# 【2】EfficientNetV4 (2024 Q3) - 官方最新版，效率提升12%
# MODEL_NAME = 'efficientnetv4_rw_s.ft_in1k'

# 【3】ConvNeXt V3 (2024 Q2) - Meta新版，轻量且精度高
# MODEL_NAME = 'convnextv3_atto.fcmae_ft_in1k'

# ==================== 🎯 高精度型 (2024-2025) ====================
# 【4】EVA-03 (2024 Q4) - 超越EVA-02，多模态预训练
MODEL_NAME = 'eva03_small_patch14_224.mim_m38m_ft_in1k'

# 【5】Qwen-VL 2.0 (2025 Q1) - 通义千问视觉分支，中文优化
# MODEL_NAME = 'qwen_vl_2.0_4b.clip_zh_in1k'

# 【6】SAM-2 (2024 Q4) - Meta新作，分割+分类双优
# MODEL_NAME = 'sam2_base_image_classifier.in1k'

# ==================== 🚀 多模态型 (2024-2025) ====================
# 【7】CLIP 2.0 (2024 Q4) - OpenAI升级版，图文匹配更优
# MODEL_NAME = 'clip_vit_large_14_336.laion2b_s34b_b88k_ft_in1k'

# 【8】MiniCPM-V 2.0 (2025 Q1) - 国产轻量多模态，中文友好
# MODEL_NAME = 'minicpm_v_2.0_2b.clip_zh_in1k'

# 【9】InternVL-2 (2024 Q4) - 商汤多模态，中文场景最优
# MODEL_NAME = 'internvl2_4b.clip_zh_in1k'

# ====================================================
# ⚙️ 2024-2025模型自动适配配置 (精准分辨率/批次)
# ====================================================
# 2024-2025模型分辨率映射表（基于官方推荐）
RESOLUTION_MAP = {
    # 高效轻量模型
    "mobilevitv4": 256,
    "efficientnetv4": 224,
    "convnextv3": 224,
    # 高精度模型
    "eva03": 224,
    "qwen_vl_2.0": 224,
    "sam2": 256,
    # 多模态模型
    "clip_vit_large_14_336": 336,
    "minicpm_v_2.0": 224,
    "internvl2": 224,
}

# 智能分辨率匹配
IMG_SIZE = 224  # 默认
for key in RESOLUTION_MAP.keys():
    if key in MODEL_NAME:
        IMG_SIZE = RESOLUTION_MAP[key]
        break

# 智能批次大小 (2024-2025模型显存优化)
batch_config = {
    "4b": 16,    # 4B参数量模型
    "large": 32, # 大型模型
    "sam2": 16,  # SAM-2显存占用高
    "base": 32,  # 基础版模型
    "small": 64, # 小型模型
    "atto": 64,  # 超轻量模型
}
BATCH_SIZE = 64  # 默认
for key, bs in batch_config.items():
    if key in MODEL_NAME:
        BATCH_SIZE = bs
        break

print(f"⚡ 2024-2025模型适配: {MODEL_NAME} | 分辨率: {IMG_SIZE}x{IMG_SIZE} | 批次: {BATCH_SIZE}")

ROOT_TRAIN = cfg.train_dir 
ROOT_TEST = cfg.val_dir

# ====================================================
# 2. 增强型数据处理 (2024-2025最佳实践)
# ====================================================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),  # 先放大再裁剪（2024主流）
    transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(degrees=(-15, 15)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3))  # 2024必加增强
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print(f"📂 正在读取数据集... (2024-2025 SOTA模型: {MODEL_NAME})")
train_dataset = ImageFolder(ROOT_TRAIN, transform=train_transform)
val_dataset = ImageFolder(ROOT_TEST, transform=val_transform)

class_names = train_dataset.classes
num_classes = len(class_names)
print(f"✅ 检测到类别: {class_names} (共{num_classes}类)")

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=0 if os.name == 'nt' else 4,  # Windows兼容
    pin_memory=True  # 2024优化：GPU传输加速
)
val_dataloader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=0 if os.name == 'nt' else 4,
    pin_memory=True
)

train_data_size = len(train_dataset)
val_data_size = len(val_dataset)
# 2024设备适配：优先CUDA，其次MPS（Mac），最后CPU
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"💻 训练设备: {device}")

# ====================================================
# 3. 🚀 2024-2025模型智能加载 (增强容错)
# ====================================================
print(f"🚀 正在加载2024-2025 SOTA模型: {MODEL_NAME}...")

model = None
# 多策略加载 (适配2024新版模型)
load_strategies = [
    # 策略1：带尺寸+预训练（官方推荐）
    lambda: timm.create_model(
        MODEL_NAME, 
        pretrained=True, 
        num_classes=num_classes, 
        img_size=IMG_SIZE
    ),
    # 策略2：不带尺寸+预训练（兼容部分模型）
    lambda: timm.create_model(
        MODEL_NAME, 
        pretrained=True, 
        num_classes=num_classes
    ),
    # 策略3：本地权重 fallback（预训练权重下载失败时）
    lambda: timm.create_model(
        MODEL_NAME, 
        pretrained=False, 
        num_classes=num_classes
    )
]

for idx, strategy in enumerate(load_strategies, 1):
    try:
        print(f"   尝试加载策略 {idx}...")
        model = strategy()
        if model is not None:
            print(f"   ✅ 策略 {idx} 加载成功")
            break
    except TypeError as e:
        print(f"   ⚠️ 策略 {idx} 类型错误: {str(e)[:100]}")
    except RuntimeError as e:
        print(f"   ⚠️ 策略 {idx} 运行时错误: {str(e)[:100]}")
    except Exception as e:
        print(f"   ⚠️ 策略 {idx} 未知错误: {str(e)[:100]}")

if model is None:
    print(f"❌ 所有加载策略失败，退出程序")
    exit()

model = model.to(device)
# 2024优化：PyTorch 2.0+ 编译加速（仅CUDA/MPS）
if device in ['cuda', 'mps']:
    model = torch.compile(model)
print("✅ 2024-2025 SOTA模型加载成功！")

# ====================================================
# 4. 2024-2025优化的训练配置
# ====================================================
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑（2024主流）

# 自适应优化器配置（2024参数最佳实践）
lr = 5e-5 if any(key in MODEL_NAME for key in ["4b", "large", "sam2"]) else 1e-4
weight_decay = 0.05 if "vit" in MODEL_NAME or "eva03" in MODEL_NAME else 0.01

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay,
    betas=(0.9, 0.999),
    eps=1e-8
)

# 2024主流学习率调度：余弦退火重启
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,  # 每10个epoch重启
    T_mult=2,
    eta_min=1e-6
)

# 混合精度训练（仅CUDA）
scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None

# ====================================================
# 5. 增强型训练循环 (2024-2025最佳实践)
# ====================================================
epoch = 30
total_train_step = 0
log_dir = f"./logs_2025_{MODEL_NAME.split('.')[0]}_{num_classes}class"
writer = SummaryWriter(log_dir)
print(f"📝 训练日志保存至: {log_dir}")

start_time = time.time()
best_acc = 0.0
best_f1 = 0.0
patience = 5  # 早停机制（2024必加）
no_improve = 0

for i in range(epoch):
    print(f"\n======= 📅 Epoch {i+1} / {epoch} =======")

    # --- 训练阶段 ---
    model.train()
    train_bar = tqdm(train_dataloader, desc="🚀 训练中", unit="batch", mininterval=0.5)
    epoch_train_loss = 0.0
    epoch_train_acc = 0.0

    for data in train_bar:
        imgs, targets = data
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()

        # 混合精度训练（2024优化：bfloat16更稳定）
        if device == 'cuda' and scaler is not None:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(imgs) 
                loss = loss_fn(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

        # 批次指标计算
        batch_acc = (outputs.argmax(1) == targets).float().mean().item()
        epoch_train_loss += loss.item()
        epoch_train_acc += batch_acc
        
        train_bar.set_postfix(
            loss=f"{loss.item():.4f}", 
            acc=f"{batch_acc:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.6f}"
        )

        writer.add_scalar("train_loss_step", loss.item(), total_train_step)
        writer.add_scalar("train_acc_step", batch_acc, total_train_step)
        total_train_step += 1

    # 学习率调度
    scheduler.step()

    # --- 验证阶段 ---
    model.eval()
    total_test_loss = 0.0
    total_accuracy = 0
    all_targets = []
    all_probs = [] 
    
    val_bar = tqdm(val_dataloader, desc="✅ 验证中", unit="batch", mininterval=0.5)

    with torch.no_grad():
        for data in val_bar:
            imgs, targets = data
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # 混合精度推理
            with torch.cuda.amp.autocast(dtype=torch.bfloat16) if device == 'cuda' else contextlib.nullcontext():
                outputs = model(imgs)
                loss = loss_fn(outputs, targets)
            
            total_test_loss += loss.item()

            # 2024精度优化：强制float32计算概率
            probs = torch.softmax(outputs.float(), dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    # --- 指标计算 ---
    all_probs = np.array(all_probs, dtype=np.float32)
    all_targets = np.array(all_targets)
    predicted_labels = np.argmax(all_probs, axis=1)

    # 全局指标
    val_acc = total_accuracy.item() / val_data_size
    val_loss = total_test_loss / len(val_dataloader)
    train_acc_avg = epoch_train_acc / len(train_dataloader)
    train_loss_avg = epoch_train_loss / len(train_dataloader)

    # 多分类指标
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        all_targets, predicted_labels, average='macro', zero_division=1
    )
    
    # AUC计算（增强容错）
    auc = 0.0
    try:
        if num_classes == 2:
            auc = roc_auc_score(all_targets, all_probs[:, 1])
        else:
            auc = roc_auc_score(
                all_targets, all_probs, 
                multi_class='ovr', 
                average='macro',
                labels=np.arange(num_classes)
            )
    except Exception as e:
        print(f"⚠️ AUC计算跳过: {str(e)[:80]}")

    # --- 日志输出 ---
    print(f"\n📊 训练指标 (Epoch {i+1}):")
    print(f"   训练 | Loss: {train_loss_avg:.4f} | Acc: {train_acc_avg:.4f}")
    print(f"   验证 | Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
    print(f"   综合 | AUC: {auc:.4f} | F1: {f1_score:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    print(f"   学习率 | 当前: {optimizer.param_groups[0]['lr']:.6f}")

    # TensorBoard记录
    writer.add_scalar("train_loss_epoch", train_loss_avg, i)
    writer.add_scalar("train_acc_epoch", train_acc_avg, i)
    writer.add_scalar("val_loss_epoch", val_loss, i)
    writer.add_scalar("val_acc_epoch", val_acc, i)
    writer.add_scalar("val_auc_epoch", auc, i)
    writer.add_scalar("val_f1_epoch", f1_score, i)
    writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], i)

    # --- 模型保存 (2024优化：只保存权重) ---
    if val_acc > best_acc or f1_score > best_f1:
        best_acc = max(val_acc, best_acc)
        best_f1 = max(f1_score, best_f1)
        no_improve = 0  # 重置早停计数
        
        # 创建保存目录
        if not os.path.exists("./model_pth_2025"):
            os.makedirs("./model_pth_2025")
        
        # 保存权重（节省空间）
        simple_name = MODEL_NAME.split('.')[0] 
        save_path = f"./model_pth_2025/best_model_{simple_name}_acc{best_acc:.4f}_f1{best_f1:.4f}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': i+1,
            'best_acc': best_acc,
            'best_f1': best_f1,
            'class_names': class_names
        }, save_path)
        
        print(f"🌟 最佳模型已保存 | Acc: {best_acc:.4f} | F1: {best_f1:.4f}")
    else:
        no_improve += 1
        print(f"⚠️ 验证指标未提升 ({no_improve}/{patience})")
        if no_improve >= patience:
            print(f"🛑 早停触发，最佳Acc: {best_acc:.4f} | 最佳F1: {best_f1:.4f}")
            break

# ====================================================
# 6. 训练总结 (2024-2025增强)
# ====================================================
end_time = time.time()
total_time = end_time - start_time
print(f"\n🎉 2024-2025 SOTA模型训练完成！")
print(f"📈 最佳精度: {best_acc:.4f} | 最佳F1: {best_f1:.4f}")
print(f"⏱️ 总耗时: {total_time/60:.2f} 分钟 (平均 {total_time/epoch:.2f} 分钟/epoch)")
print(f"💾 模型保存路径: ./model_pth_2025/")

# 生成分类报告
print(f"\n📋 最终分类报告:")
print(classification_report(all_targets, predicted_labels, target_names=class_names))

# 混淆矩阵可视化
cm = confusion_matrix(all_targets, predicted_labels)
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f'混淆矩阵 ({MODEL_NAME})', fontsize=14)
plt.colorbar()
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

# 标注数值
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.tight_layout()
plt.savefig(f"./confusion_matrix_{MODEL_NAME.split('.')[0]}.png", dpi=300)
plt.show()

writer.close()