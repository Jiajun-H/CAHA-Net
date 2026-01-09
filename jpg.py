import os
import mat73
import numpy as np
from PIL import Image

# ================= 配置路径 =================
input_folder = 'brainTumorDataPublic' 
output_folder = 'C:/Users/Public/CA-densenet-main/output_images'
# ===========================================

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 1. 读取所有文件名
files = [f for f in os.listdir(input_folder) if f.endswith('.mat')]

# 关键步骤：按文件名的数字大小排序 (1.mat, 2.mat, ..., 100.mat)
# 这样能保证同一个病人的切片大致是按顺序读取的
files.sort(key=lambda x: int(os.path.splitext(x)[0]))

print(f"找到 {len(files)} 个文件，准备开始...")

# 2. 建立一个字典，用来记录每个病人的切片计数
# 格式: { '病人ID': 当前是第几张 }
patient_slice_counter = {}

success_count = 0

for file_name in files:
    file_path = os.path.join(input_folder, file_name)
    
    try:
        # 读取数据
        data_dict = mat73.loadmat(file_path)
        
        if 'cjdata' in data_dict:
            cjdata = data_dict['cjdata']
            image_data = cjdata['image'] 
            label = int(cjdata['label']) 
            
            # 获取病人ID并转为字符串，去除可能存在的空格
            pid = str(cjdata['PID']).strip()
            
            # --- 计数逻辑 ---
            if pid not in patient_slice_counter:
                patient_slice_counter[pid] = 0
            patient_slice_counter[pid] += 1
            
            # 获取当前是该病人的第几张图
            current_slice_idx = patient_slice_counter[pid]
            
            # --- 归一化处理 ---
            min_val, max_val = np.min(image_data), np.max(image_data)
            if max_val - min_val != 0:
                image_norm = (image_data - min_val) / (max_val - min_val) * 255.0
            else:
                image_norm = image_data * 0
            
            # --- 保存图片 ---
            # 对应文件夹
            label_dir = os.path.join(output_folder, str(label))
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)
            
            # 原始文件名去掉 .mat 后缀
            original_name_base = os.path.splitext(file_name)[0]
            
            # 生成你要求的文件名: 病人ID_第几张图_原始文件名.jpg
            # 例如: 100234_1_245.jpg
            new_name = f"{pid}_{current_slice_idx}_{original_name_base}.jpg"
            save_path = os.path.join(label_dir, new_name)
            
            Image.fromarray(image_norm.astype(np.uint8)).save(save_path)
            success_count += 1

            if success_count % 500 == 0:
                print(f"已处理 {success_count} 张...")

    except Exception as e:
        print(f"文件 {file_name} 出错: {e}")

print(f"处理完成！共生成 {success_count} 张图片。")
print(f"文件命名格式示例: 病人ID_序号_原始名.jpg (例如: 10243_3_150.jpg)")
