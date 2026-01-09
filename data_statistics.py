"""
数据集统计脚本
统计每种类别的病人个数、图片总数、平均每人切片数
"""

import os
from collections import defaultdict
from pathlib import Path

def get_patient_id(filename):
    """
    从文件名提取病人ID
    文件名格式: 病人ID_切片序号_其他编号.jpg
    例如: 100416_1_818.jpg -> 100416
          MR017260F_10_2393.jpg -> MR017260F
    """
    # 去掉扩展名
    name = os.path.splitext(filename)[0]
    # 分割并获取第一个部分（病人ID）
    parts = name.split('_')
    if len(parts) >= 2:
        return parts[0]
    return name

def analyze_data_folder(data_path):
    """
    分析data文件夹中的数据分布
    """
    results = {}
    
    # 遍历train和val文件夹
    for split in ['train', 'val']:
        split_path = os.path.join(data_path, split)
        if not os.path.exists(split_path):
            continue
            
        # 遍历每个类别
        for category in os.listdir(split_path):
            category_path = os.path.join(split_path, category)
            if not os.path.isdir(category_path):
                continue
            
            if category not in results:
                results[category] = {
                    'patients': set(),
                    'images': 0,
                    'patient_images': defaultdict(int)
                }
            
            # 统计该类别下的所有图片
            for filename in os.listdir(category_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    patient_id = get_patient_id(filename)
                    results[category]['patients'].add(patient_id)
                    results[category]['images'] += 1
                    results[category]['patient_images'][patient_id] += 1
    
    return results

def print_statistics(results):
    """
    打印统计结果
    """
    print("\n" + "=" * 80)
    print("数据集统计结果")
    print("=" * 80)
    
    # 表头
    print(f"\n{'病因类别(Class)':<20} {'患病人数(Patients)':>18} {'图片总数(Images)':>18} {'平均每人切片数':>15}")
    print("-" * 80)
    
    total_patients = 0
    total_images = 0
    all_patients = set()
    
    # 按类别名排序输出
    for category in sorted(results.keys()):
        data = results[category]
        num_patients = len(data['patients'])
        num_images = data['images']
        avg_per_patient = num_images / num_patients if num_patients > 0 else 0
        
        print(f"{category:<20} {num_patients:>18} {num_images:>18} {avg_per_patient:>15.2f}")
        
        total_patients += num_patients
        total_images += num_images
        all_patients.update(data['patients'])
    
    # 总计
    print("-" * 80)
    unique_patients = len(all_patients)
    total_avg = total_images / unique_patients if unique_patients > 0 else 0
    print(f"{'总计(Total)':<20} {unique_patients:>18} {total_images:>18} {total_avg:>15.2f}")
    print(f"{'(按类别累加)':<20} {total_patients:>18}")
    print("=" * 80)
    
    # 详细统计每个类别的切片数分布
    print("\n\n各类别切片数分布详情:")
    print("-" * 80)
    
    for category in sorted(results.keys()):
        data = results[category]
        patient_images = list(data['patient_images'].values())
        
        if patient_images:
            min_slices = min(patient_images)
            max_slices = max(patient_images)
            avg_slices = sum(patient_images) / len(patient_images)
            
            print(f"\n{category}:")
            print(f"  - 最少切片数: {min_slices}")
            print(f"  - 最多切片数: {max_slices}")
            print(f"  - 平均切片数: {avg_slices:.2f}")
            print(f"  - 中位数切片数: {sorted(patient_images)[len(patient_images)//2]}")

def save_to_file(results, output_path):
    """
    将统计结果保存到文件
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("数据集统计结果\n")
        f.write("=" * 80 + "\n\n")
        
        # 表头
        f.write(f"{'病因类别(Class)':<20} {'患病人数(Patients)':>18} {'图片总数(Images)':>18} {'平均每人切片数':>15}\n")
        f.write("-" * 80 + "\n")
        
        total_patients = 0
        total_images = 0
        all_patients = set()
        
        for category in sorted(results.keys()):
            data = results[category]
            num_patients = len(data['patients'])
            num_images = data['images']
            avg_per_patient = num_images / num_patients if num_patients > 0 else 0
            
            f.write(f"{category:<20} {num_patients:>18} {num_images:>18} {avg_per_patient:>15.2f}\n")
            
            total_patients += num_patients
            total_images += num_images
            all_patients.update(data['patients'])
        
        f.write("-" * 80 + "\n")
        unique_patients = len(all_patients)
        total_avg = total_images / unique_patients if unique_patients > 0 else 0
        f.write(f"{'总计(Total)':<20} {unique_patients:>18} {total_images:>18} {total_avg:>15.2f}\n")
        f.write(f"{'(按类别累加)':<20} {total_patients:>18}\n")
        f.write("=" * 80 + "\n")
        
        # 详细统计
        f.write("\n\n各类别切片数分布详情:\n")
        f.write("-" * 80 + "\n")
        
        for category in sorted(results.keys()):
            data = results[category]
            patient_images = list(data['patient_images'].values())
            
            if patient_images:
                min_slices = min(patient_images)
                max_slices = max(patient_images)
                avg_slices = sum(patient_images) / len(patient_images)
                
                f.write(f"\n{category}:\n")
                f.write(f"  - 最少切片数: {min_slices}\n")
                f.write(f"  - 最多切片数: {max_slices}\n")
                f.write(f"  - 平均切片数: {avg_slices:.2f}\n")
                f.write(f"  - 中位数切片数: {sorted(patient_images)[len(patient_images)//2]}\n")
    
    print(f"\n统计结果已保存到: {output_path}")

def main():
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'data')
    
    print(f"正在分析数据目录: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"错误: 数据目录不存在: {data_path}")
        return
    
    # 分析数据
    results = analyze_data_folder(data_path)
    
    if not results:
        print("未找到任何数据")
        return
    
    # 打印统计结果
    print_statistics(results)
    
    # 保存结果到文件
    output_path = os.path.join(script_dir, 'result', 'data_statistics.txt')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_to_file(results, output_path)

if __name__ == '__main__':
    main()
