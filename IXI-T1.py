import os
import glob
import numpy as np
import nibabel as nib
from PIL import Image
import random

# ================= 配置路径 =================
input_nii_folder = 'C:/Users/Public/IXI' 
output_folder = 'C:/Users/Public/CA-densenet-main/output_images/no_tumor'

# ================= 采样配置 =================
PATIENT_LIMIT = 100        
SLICES_PER_VIEW = 4        # 每个视角取 4 张
# ===========================================

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 1. 扫描文件
all_files = glob.glob(os.path.join(input_nii_folder, '*.nii.gz'))
if not all_files:
    print("错误：未找到文件。")
    exit()

# 2. 随机打乱并取前 100
random.seed(42)
random.shuffle(all_files)
selected_files = all_files[:PATIENT_LIMIT]

print(f"开始极速处理 {len(selected_files)} 个病人...")
total_images_generated = 0

for index, file_path in enumerate(selected_files):
    try:
        file_name = os.path.basename(file_path)
        pid = file_name.split('-')[0]
        
        # 加载数据 (这一步是主要耗时，大概0.5秒)
        img = nib.load(file_path)
        data = img.get_fdata()
        
        # 归一化 (利用 Numpy 整体运算，极快)
        data_min, data_max = np.min(data), np.max(data)
        if data_max > data_min: # 防止除以0
            data = (data - data_min) / (data_max - data_min) * 255
        data = data.astype(np.uint8)
        
        # 定义视角 (名称, 轴索引, 旋转次数)
        # 0:侧面, 1:正面, 2:顶部
        views_config = [('Sagittal', 0, 1), ('Coronal', 1, 1), ('Axial', 2, 1)]
        
        for view_name, axis, rot_k in views_config:
            
            # 获取该方向的总层数
            total_slices = data.shape[axis]
            
            # 【核心优化】直接计算中间区域的 4 个坐标点
            # 我们只在 35% 到 65% 的范围内取样，这里肯定是大脑核心，不用检查是否全黑
            start_pos = total_slices * 0.35
            end_pos = total_slices * 0.65
            
            # 直接生成 4 个整数索引
            target_indices = np.linspace(start_pos, end_pos, SLICES_PER_VIEW, dtype=int)
            
            for slice_idx in target_indices:
                # 提取切片
                slice_data = np.take(data, slice_idx, axis=axis)
                
                # 简单旋转
                slice_data = np.rot90(slice_data, k=rot_k)
                
                # 图片处理
                img_pil = Image.fromarray(slice_data).resize((512, 512))
                
                # 保存
                save_name = f"{pid}_{view_name}_{slice_idx}_no_tumor.jpg"
                img_pil.save(os.path.join(output_folder, save_name))
                
                total_images_generated += 1
        
        # 打印进度 (每10个病人打印一次，或者你可以每1个打印一次)
        print(f"[{index+1}/{len(selected_files)}] 完成病人 {pid} (累计生成 {total_images_generated} 张)")

    except Exception as e:
        print(f"Skip {file_name}: {e}")

print(f"\n处理完成！总共生成: {total_images_generated} 张图片。")
