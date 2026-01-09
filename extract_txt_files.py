import os
import shutil
from pathlib import Path

def extract_txt_files():
    # 定义目标目录
    dest_dir = Path('result')
    if not dest_dir.exists():
        dest_dir.mkdir(parents=True)
    
    # 查找所有以 checkpoints 开头的目录
    current_dir = Path('.')
    checkpoint_dirs = [d for d in current_dir.iterdir() if d.is_dir() and d.name.startswith('checkpoints')]
    
    count = 0
    for cp_dir in checkpoint_dirs:
        print(f"正在扫描目录: {cp_dir}")
        # 递归查找所有 .txt 文件
        for txt_file in cp_dir.rglob('*.txt'):
            # 保持原始文件名
            new_name = txt_file.name
            dest_path = dest_dir / new_name
            
            # 如果文件名冲突，可以在这里处理，但用户要求保持命名不变
            shutil.copy2(txt_file, dest_path)
            print(f"已复制: {txt_file} -> {dest_path}")
            count += 1
            
    print(f"\n完成！共提取了 {count} 个文件到 {dest_dir} 目录。")

if __name__ == "__main__":
    extract_txt_files()
