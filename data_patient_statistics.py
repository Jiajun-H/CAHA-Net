#!/usr/bin/env python3
"""
统计 data 目录下每个数据集->每个类别的图片数、病人数、平均每人图片数。
用法：
  python data_patient_statistics.py --data_root ./data --out stats.csv

算法：
- 支持 data 根目录下直接是 dataset（每个 dataset 下可能有 train/val/test），
  或者 data 本身就是 dataset（包含 train/val）。
- 对每个类别递归收集图片文件，然后尝试从文件名提取病人 ID：
  优先取下划线或连字符前缀；否则取开头连续字母/数字序列；若仍无，则每张图片当一个唯一患者。
- 输出 CSV：dataset,class,total_images,unique_patients,avg_imgs_per_patient,median,stdev
"""
from pathlib import Path
import argparse
import csv
import re
from collections import defaultdict, Counter
import statistics
from typing import Dict, List

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def extract_pid(fname: str) -> str:
    """尝试从文件名提取病人ID的启发式规则。
    规则优先级：
    1) 文件名中第一个'_'之前的部分（如 88670_1_792.png -> 88670）
    2) 文件名中第一个'-'之前的部分
    3) 开头连续字母数字序列
    4) 否则返回完整 stem
    """
    stem = Path(fname).stem
    # 1. 下划线
    if "_" in stem:
        return stem.split("_")[0]
    # 2. 连字符
    if "-" in stem:
        return stem.split("-")[0]
    # 3. 开头字母数字
    m = re.match(r'([A-Za-z0-9]+)', stem)
    if m:
        return m.group(1)
    return stem


def gather_images(root: Path) -> Dict[str, List[Path]]:
    """返回类名 -> 列表图片路径。支持两种布局：
    - root/train/<class> and root/val/<class>
    - root/<class> (直接类文件夹)
    当检测 train 子目录存在时优先使用 train+val+test 的类集合。
    """
    res = defaultdict(list)
    if (root / 'train').exists():
        # use train/val/test subfolders
        for split in ['train', 'val', 'test']:
            sp = root / split
            if not sp.exists():
                continue
            for cls in sp.iterdir():
                if not cls.is_dir():
                    continue
                for p in cls.rglob('*'):
                    if is_image(p):
                        res[cls.name].append(p)
    else:
        # treat immediate subfolders as classes
        for cls in root.iterdir():
            if not cls.is_dir():
                continue
            for p in cls.rglob('*'):
                if is_image(p):
                    res[cls.name].append(p)
    return res


def analyze_class(img_paths: List[Path]) -> Dict[str, object]:
    total = len(img_paths)
    if total == 0:
        return {'total': 0, 'unique_patients': 0, 'avg_per_patient': 0.0, 'median': 0.0, 'stdev': 0.0}
    pid_map = defaultdict(list)
    for p in img_paths:
        pid = extract_pid(p.name)
        pid_map[pid].append(p)
    counts = [len(v) for v in pid_map.values()]
    unique_patients = len(counts)
    avg = sum(counts) / unique_patients if unique_patients else 0.0
    median = statistics.median(counts) if unique_patients else 0.0
    stdev = statistics.pstdev(counts) if unique_patients else 0.0
    return {
        'total': total,
        'unique_patients': unique_patients,
        'avg_per_patient': round(avg, 3),
        'median': median,
        'stdev': round(stdev, 3),
    }


def find_datasets(data_root: Path) -> List[Path]:
    # if data_root contains train/val -> treat as single dataset
    if (data_root / 'train').exists() or (data_root / 'val').exists():
        return [data_root]
    # otherwise each subdir is a dataset
    datasets = [p for p in data_root.iterdir() if p.is_dir()]
    return datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='./data', help='data 根目录')
    parser.add_argument('--out', default='data_patient_stats.csv', help='输出 CSV 文件')
    parser.add_argument('--top_patients', type=int, default=5, help='在控制台显示每个类别最常见的前 N 个病人')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise SystemExit(f'data_root 不存在: {data_root}')

    datasets = find_datasets(data_root)

    rows = []
    for ds in datasets:
        class_map = gather_images(ds)
        print(f'\nDataset: {ds.name}  Classes: {len(class_map)}')
        for cls, imgs in sorted(class_map.items()):
            stats = analyze_class(imgs)
            rows.append({
                'dataset': ds.name,
                'class': cls,
                'total_images': stats['total'],
                'unique_patients': stats['unique_patients'],
                'avg_per_patient': stats['avg_per_patient'],
                'median_per_patient': stats['median'],
                'stdev_per_patient': stats['stdev'],
            })

            print(f"  - Class '{cls}': images={stats['total']} patients={stats['unique_patients']} avg_per_patient={stats['avg_per_patient']} median={stats['median']} stdev={stats['stdev']}")

            # show top patients
            pid_counter = Counter()
            for p in imgs:
                pid_counter[extract_pid(p.name)] += 1
            top = pid_counter.most_common(args.top_patients)
            if top:
                s = ", ".join([f"{pid}:{cnt}" for pid, cnt in top])
                print(f"    top {args.top_patients}: {s}")

    # write csv
    out_path = Path(args.out)
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset','class','total_images','unique_patients','avg_per_patient','median_per_patient','stdev_per_patient'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f'\nSaved CSV: {out_path.resolve()}')


if __name__ == '__main__':
    main()
