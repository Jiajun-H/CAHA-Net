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
import time  # 新增：用于计时
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from albumentations import Compose, CLAHE, HorizontalFlip, Rotate
from PIL import Image

# *** 引入模型 ***
from model_utils import BrainTumorFinalNet

# *** 引入计算FLOPs的库 (请确保 pip install thop) ***
try:
    from thop import profile
except ImportError:
    print("❌ 警告: 未安装 'thop' 库。无法计算 GFLOPs。请运行 pip install thop")
    profile = None

# ================= 1. 基础配置 =================
BASE_CONFIG = {
    'train_dir': './data/train',      
    'val_dir': './data/val',          
    'log_dir': './logs_ablation',    
    'save_dir': './checkpoints_ablation',
    'num_classes': 4,
    'batch_size': 64,       
    'lr': 1e-3,             
    'epochs': 60,          
    'num_workers': 8,      
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
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
        if use_clahe: aug_list.append(CLAHE(clip_limit=4.0, p=1.0))
        if is_train:
            aug_list.extend([Rotate(limit=15, p=0.5), HorizontalFlip(p=0.5)])
        self.aug = Compose(aug_list)
    def __call__(self, img):
        img_np = np.array(img)
        if self.aug: return Image.fromarray(self.aug(image=img_np)['image'])
        return Image.fromarray(img_np)

# --- FMix 逻辑 (保持不变) ---
def fftfreqnd(h, w=None, z=None):
    fz = fx = 0
    fy = np.fft.fftfreq(h)
    if w is not None: fx = np.fft.fftfreq(w)
    if z is not None: fz = np.fft.fftfreq(z)
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
    if mask.ndim > 2: mask = mask[0]
    mask = mask - mask.min()
    return mask / mask.max()

def fmix_data(data, targets, alpha=1.0, decay_power=3.0, shape=(224,224), device='cuda'):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    index = torch.randperm(data.size(0)).to(device)
    soft_mask = make_low_freq_image(decay_power, shape)
    mask_flat = soft_mask.flatten()
    idx = int((1 - lam) * len(mask_flat))
    threshold = np.partition(mask_flat, idx)[idx]
    binary_mask = torch.from_numpy((soft_mask > threshold).astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)
    mixed_x = data * binary_mask + data[index] * (1 - binary_mask)
    return mixed_x, targets, targets[index], binary_mask.mean().item()

# ================= 3. 核心功能：效率评估模块 (NEW) =================
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
            flops, params = profile(model, inputs=(dummy_input, ), verbose=False)
            flops_g = flops / 1e9      # 转为 GFLOPs
            params_m = params / 1e6    # 转为 Million
        except Exception as e:
            print(f"GFLOPs calculation failed: {e}")

    # 2. 计算推理速度 (FPS & Latency)
    # 预热 (Warm up)
    for _ in range(20):
        with torch.no_grad():
            _ = model(dummy_input)
            
    # 正式测量
    iterations = 100
    torch.cuda.synchronize() if device == 'cuda' else None
    t_start = time.time()
    
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
            
    torch.cuda.synchronize() if device == 'cuda' else None
    t_end = time.time()
    
    avg_time = (t_end - t_start) / iterations
    fps = 1.0 / avg_time
    latency_ms = avg_time * 1000 # 毫秒
    
    return flops_g, params_m, fps, latency_ms

# ================= 4. 单次实验训练函数 =================
def train_one_experiment(config):
    device = config['device']
    
    # --- 生成实验名称 ---
    name_parts = []
    if config['use_symmetry']: name_parts.append("Sym")
    else: name_parts.append("PureBase")
    
    if config['use_clahe']: name_parts.append("CLAHE")
    if config['use_fmix']:  name_parts.append("FMix")
    if config['use_dcn']:   name_parts.append("DCN")
    if config['use_ca']:    name_parts.append("CA")
    
    exp_name = "_".join(name_parts)
    current_save_dir = os.path.join(config['save_dir'], exp_name)
    current_log_dir = os.path.join(config['log_dir'], exp_name)
    os.makedirs(current_save_dir, exist_ok=True)
    if os.path.exists(current_log_dir): shutil.rmtree(current_log_dir)
    writer = SummaryWriter(current_log_dir)
    
    print("\n" + "#"*60)
    print(f"🔥 开始执行: {exp_name}")
    print("#"*60 + "\n")

    # 数据集
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        AlbumentationsTransform(is_train=True, use_clahe=config['use_clahe']),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        AlbumentationsTransform(is_train=False, use_clahe=config['use_clahe']),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_loader = DataLoader(ImageFolder(config['train_dir'], transform=train_transform), 
                              batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=True)
    val_loader = DataLoader(ImageFolder(config['val_dir'], transform=val_transform), 
                            batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True)

    # 模型与硬件信息
    model = BrainTumorFinalNet(
        num_classes=config['num_classes'],
        use_dcn=config['use_dcn'],
        use_ca=config['use_ca'],
        use_symmetry=config['use_symmetry']
    ).to(device)
    
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    
    # --- [NEW] 在训练前评估模型计算复杂度 (GFLOPs & Params) ---
    flops_g, params_m, _, _ = measure_efficiency(model, device)
    print(f"📊 模型分析: Params={params_m:.2f}M, GFLOPs={flops_g:.2f}G")

    criterion = FocalLoss(gamma=2.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_macro_f1 = 0.0
    
    # --- [NEW] 记录训练总开始时间 ---
    train_start_time = time.time()

    # 训练循环
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
            
            if (i+1) % 20 == 0:
                 print(f"\r[Exp: {exp_name}] Epoch {epoch+1} Step {i+1}/{len(train_loader)} Loss: {loss.item():.4f}", end="")

        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Train/Loss', avg_train_loss, epoch)

        # --- Val ---
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, targets).item()
                probs = torch.softmax(outputs, dim=1)
                all_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        # 指标计算
        report = classification_report(all_targets, all_preds, output_dict=True, zero_division=0)
        macro_f1 = report['macro avg']['f1-score']
        macro_recall = report['macro avg']['recall']
        
        try:
            auc = roc_auc_score(np.eye(config['num_classes'])[all_targets], all_probs, multi_class='ovr', average='macro')
        except: auc = 0.0

        cm = confusion_matrix(all_targets, all_preds)
        FP = cm.sum(axis=0) - np.diag(cm)
        TN = cm.sum() - (FP + (cm.sum(axis=1) - np.diag(cm)) + np.diag(cm))
        macro_spec = (TN / (TN + FP + 1e-6)).mean()
        macro_fpr = (FP / (FP + TN + 1e-6)).mean()

        print(f"\nVal ({exp_name}) -> F1: {macro_f1:.4f} | AUC: {auc:.4f}")
        
        # 保存逻辑
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            
            model_filename = f"{exp_name}_best_model.pth"
            txt_filename   = f"{exp_name}_best_metrics.txt"
            save_path_model = os.path.join(current_save_dir, model_filename)
            torch.save(model.state_dict(), save_path_model)
            
            # --- [NEW] 获取所有最终指标 ---
            # 1. 训练时长
            total_time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - train_start_time))
            
            # 2. 模型文件大小 (MB)
            model_size_mb = os.path.getsize(save_path_model) / (1024 * 1024)
            
            # 3. 显存占用 (Max Allocated)
            max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
            
            # 4. 重新测试推理速度 (此时模型已经是最佳权重)
            # 注意：FPS 和 GFLOPs 之前算过，但 FPS 可以在这里再测一次更准确
            _, _, fps, latency = measure_efficiency(model, device)
            
            # 5. 写入 TXT (包含审稿人要求的所有 Efficiency 指标)
            with open(os.path.join(current_save_dir, txt_filename), "w", encoding="utf-8") as f:
                f.write(f"Experiment:       {exp_name}\n")
                f.write(f"Best Epoch:       {epoch+1}\n")
                f.write("="*40 + "\n")
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
                f.write("="*40 + "\n")
                f.write("Configuration:\n" + str(config))
                
            print(f"✅ [新纪录] 指标与效率数据已保存: {txt_filename}")
            
    writer.close()
    del model, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()
    print(f"🏁 实验结束: {exp_name}\n")

# ================= 5. 自动化队列 =================
def run_all_experiments():
    # 现在的队列是空的，或者你可以在这里写新的实验
    experiment_queue = [
        {'use_symmetry': False, 'use_clahe': True, 'use_fmix': True, 'use_dcn': False, 'use_ca': True}
        
    ] 

    # === 下面是被注释掉的备份代码（放在列表外面就安全了）===
    '''
    # 1. 纯基线
    {'use_symmetry': False, 'use_clahe': False, 'use_fmix': False, 'use_dcn': False, 'use_ca': False},
    # 2. 最终完整模型
    {'use_symmetry': False, 'use_clahe': False, 'use_fmix': False, 'use_dcn': False, 'use_ca': False},

    {'use_symmetry': False, 'use_clahe': True, 'use_fmix': True, 'use_dcn': False, 'use_ca': False},
    {'use_symmetry': False, 'use_clahe': False, 'use_fmix': False, 'use_dcn': True, 'use_ca': False},
    {'use_symmetry': False, 'use_clahe': False, 'use_fmix': False, 'use_dcn': False, 'use_ca': True},
    {'use_symmetry': False,  'use_clahe': True,  'use_fmix': True,  'use_dcn': True,  'use_ca': True},
    '''
    
    # ... 后面的代码 ...


    print(f"📋 任务队列: {len(experiment_queue)} 个实验")
    for idx, params in enumerate(experiment_queue):
        cfg = BASE_CONFIG.copy()
        cfg.update(params)
        print(f"\n🚀 执行第 {idx+1}/{len(experiment_queue)} 个实验")
        try:
            train_one_experiment(cfg)
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    run_all_experiments()
