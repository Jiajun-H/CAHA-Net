"""Run training sequentially for the three prepared Kaggle datasets.

Usage:
  python run_kaggle_all.py --python /path/to/python --train_script train_kaggle_album.py

By default this script calls `python train_kaggle_album.py --data_dir <dataset>`
without `--mode` so the training script uses its internal `ENABLE_TRANSFER_LEARNING`
and `PRETRAINED_CKPT_PATH`. Logs are saved to `runs/<dataset_name>.log`.
"""
import argparse
import subprocess
from pathlib import Path
import sys

DATA_ROOT = Path("./data_kaggle_prepared")
DATASETS = [
    "Brain_MRI_Images_for_Brain_Tumor_Detection",
    "Brain_Tumor_Image_Dataset",
    "Medical_Image_DataSet_Brain_Tumor_Detection",
]


def run_one(python_exe: str, train_script: str, dataset_path: Path, extra_args=None):
    out_log = Path("runs") / f"{dataset_path.name}.log"
    out_log.parent.mkdir(parents=True, exist_ok=True)

    cmd = [python_exe, train_script, "--data_dir", str(dataset_path)]
    if extra_args:
        cmd += extra_args

    print(f"Running: {' '.join(cmd)}")
    with out_log.open("wb") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        ret = proc.wait()
    print(f"Finished {dataset_path.name} -> returncode={ret}. Log: {out_log}")
    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable, help="Python executable to run the training script")
    parser.add_argument("--train_script", default="train_kaggle_album.py", help="Training script to call")
    parser.add_argument("--data_root", default=str(DATA_ROOT), help="Prepared datasets root")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional list of dataset folder names to run")
    parser.add_argument("--extra_args", nargs=argparse.REMAINDER, help="Extra args appended to each training invocation")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if args.datasets:
        ds_list = args.datasets
    else:
        ds_list = DATASETS

    failures = []
    for ds in ds_list:
        ds_path = data_root / ds
        if not ds_path.exists():
            print(f"Skipping missing dataset: {ds_path}")
            continue
        ret = run_one(args.python, args.train_script, ds_path, extra_args=args.extra_args)
        if ret != 0:
            failures.append((ds, ret))

    if failures:
        print("Some runs failed:")
        for ds, code in failures:
            print(f" - {ds}: return code {code}")
        raise SystemExit(1)
    print("All done.")


if __name__ == '__main__':
    main()
