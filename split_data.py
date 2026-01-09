import os
import random
import shutil
import sys

# ================= 用户配置区域 =================
# 1. 你的源数据文件夹路径 (里面包含 no_tumor, glioma 等子文件夹)
# 注意：请确保不要把这个文件夹放在下面的 output_root 里面，否则会无限递归
source_path = r'output_images'  # 修改为你的源数据路径

# 2. 输出路径 (保持和你原程序一致，生成 data 文件夹)
# 如果你想放在当前脚本运行的目录下，就保持为 './data'
output_root = './data'

# 3. 验证集比例 (20% 的病人进入验证集)
split_rate = 0.2
# ===============================================

def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)

def main():
    # 1. 检查源路径
    if not os.path.exists(source_path):
        print(f"错误：找不到源文件夹: {source_path}")
        return

    # 2. 获取类别 (自动扫描子文件夹)
    classes = [cla for cla in os.listdir(source_path) 
               if os.path.isdir(os.path.join(source_path, cla))]
    
    if not classes:
        print("错误：源文件夹下没有找到分类子文件夹！")
        return

    print(f"检测到类别: {classes}")
    print(f"划分模式: 按病人ID划分 (防止数据泄露)")

    # 3. 创建目录结构 (data/train/..., data/val/...)
    mkfile(os.path.join(output_root, 'train'))
    mkfile(os.path.join(output_root, 'val'))
    for cla in classes:
        mkfile(os.path.join(output_root, 'train', cla))
        mkfile(os.path.join(output_root, 'val', cla))

    # 4. 开始处理每个类别
    for cla in classes:
        cla_path = os.path.join(source_path, cla)
        images = os.listdir(cla_path)
        
        # --- 核心修改：按病人ID分组 ---
        patient_dict = {} # 格式: {'ID001': ['img1.jpg', 'img2.jpg'], ...}
        
        for img_name in images:
            # 获取文件名中第一个 '_' 之前的部分作为 ID
            # 例如: "88670_1_792.png" -> id="88670"
            # 例如: "IXI015_Axial_52.jpg" -> id="IXI015"
            try:
                pid = img_name.split('_')[0]
                if pid not in patient_dict:
                    patient_dict[pid] = []
                patient_dict[pid].append(img_name)
            except Exception:
                print(f"警告: 文件名格式异常，跳过: {img_name}")
                continue

        # 获取所有唯一的病人ID
        all_patients = list(patient_dict.keys())
        num_patients = len(all_patients)
        
        # 随机打乱病人ID
        random.seed(42) # 固定种子，保证复现
        random.shuffle(all_patients)
        
        # 计算验证集病人数量
        val_num = int(num_patients * split_rate)
        
        # 划分病人ID
        val_patients = all_patients[:val_num]      # 前20%的病人
        train_patients = all_patients[val_num:]    # 后80%的病人
        
        print(f"\n正在处理类别 [{cla}]:")
        print(f"  - 总病人数: {num_patients}")
        print(f"  - 训练集病人数: {len(train_patients)}")
        print(f"  - 验证集病人数: {len(val_patients)}")

        # --- 开始复制文件 ---
        count = 0
        total_images = len(images)
        
        # 复制验证集图片
        for pid in val_patients:
            img_list = patient_dict[pid]
            for img in img_list:
                src = os.path.join(cla_path, img)
                dst = os.path.join(output_root, 'val', cla, img)
                shutil.copy(src, dst)
                count += 1
                sys.stdout.write(f"\r  -> Copying... [{count}/{total_images}]")

        # 复制训练集图片
        for pid in train_patients:
            img_list = patient_dict[pid]
            for img in img_list:
                src = os.path.join(cla_path, img)
                dst = os.path.join(output_root, 'train', cla, img)
                shutil.copy(src, dst)
                count += 1
                sys.stdout.write(f"\r  -> Copying... [{count}/{total_images}]")
        
        print() # 换行

    print("\n========================================")
    print("划分完成！")
    print(f"文件结构已生成在: {os.path.abspath(output_root)}")
    print("包含了 'train' 和 'val' 两个文件夹。")
    print("========================================")

if __name__ == '__main__':
    main()
