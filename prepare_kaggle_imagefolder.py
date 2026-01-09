import argparse
import ast
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class SplitRatios:
    train: float
    val: float
    test: float

    def normalized(self) -> "SplitRatios":
        s = self.train + self.val + self.test
        if s <= 0:
            raise ValueError("split ratios sum must be > 0")
        return SplitRatios(self.train / s, self.val / s, self.test / s)


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_rmtree(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    """mode: copy | hardlink

    hardlink on Windows may fail across drives or without permission; falls back to copy.
    """
    ensure_dir(dst.parent)
    if mode == "hardlink":
        try:
            if dst.exists():
                return
            os.link(src, dst)
            return
        except Exception:
            # fall back
            pass
    if dst.exists():
        return
    shutil.copy2(src, dst)


def _split_counts(n: int, ratios: SplitRatios) -> Tuple[int, int, int]:
    r = ratios.normalized()
    n_train = int(round(n * r.train))
    n_val = int(round(n * r.val))
    n_test = n - n_train - n_val

    # fix rounding drift
    if n_test < 0:
        n_test = 0
    while (n_train + n_val + n_test) < n:
        n_train += 1
    while (n_train + n_val + n_test) > n:
        if n_train >= n_val and n_train >= n_test and n_train > 0:
            n_train -= 1
        elif n_val >= n_test and n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1
        else:
            break

    return n_train, n_val, n_test


def stratified_split(
    items_by_class: Dict[str, List[Path]],
    ratios: SplitRatios,
    seed: int,
) -> Dict[str, Dict[str, List[Path]]]:
    """Return {split: {class: [paths...]}}"""
    rng = random.Random(seed)
    out: Dict[str, Dict[str, List[Path]]] = {"train": {}, "val": {}, "test": {}}

    for cls, items in items_by_class.items():
        items = list(items)
        rng.shuffle(items)
        n_train, n_val, n_test = _split_counts(len(items), ratios)
        out["train"][cls] = items[:n_train]
        out["val"][cls] = items[n_train : n_train + n_val]
        out["test"][cls] = items[n_train + n_val : n_train + n_val + n_test]

    return out


def write_summary(summary_path: Path, rows: Sequence[str]) -> None:
    ensure_dir(summary_path.parent)
    summary_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def prepare_imagefolder_from_class_dirs(
    src_root: Path,
    class_dirs: Sequence[str],
    out_root: Path,
    ratios: SplitRatios,
    seed: int,
    mode: str,
    overwrite: bool,
) -> None:
    if overwrite:
        safe_rmtree(out_root)

    items_by_class: Dict[str, List[Path]] = {}
    for cls in class_dirs:
        cls_dir = src_root / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"class dir not found: {cls_dir}")
        items = [p for p in cls_dir.iterdir() if is_image_file(p)]
        if not items:
            raise RuntimeError(f"no images found in: {cls_dir}")
        items_by_class[cls] = items

    splits = stratified_split(items_by_class, ratios=ratios, seed=seed)

    rows = [f"Source: {src_root}", f"Output: {out_root}", f"Mode: {mode}", f"Seed: {seed}", ""]
    for split_name, by_cls in splits.items():
        for cls, items in by_cls.items():
            rows.append(f"{split_name}\t{cls}\t{len(items)}")
            for src in items:
                dst = out_root / split_name / cls / src.name
                copy_or_link(src, dst, mode=mode)

    write_summary(out_root / "prepare_summary.tsv", rows)


def parse_yolo_names_from_yaml(yaml_path: Path) -> List[str]:
    text = yaml_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        line_stripped = line.strip()
        if line_stripped.startswith("names:"):
            # expects: names: ['a','b',...]
            rhs = line_stripped.split("names:", 1)[1].strip()
            try:
                names = ast.literal_eval(rhs)
            except Exception as e:
                raise ValueError(f"failed to parse names from {yaml_path}: {e}")
            if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
                raise ValueError(f"invalid names format in {yaml_path}")
            return names
    raise ValueError(f"names not found in {yaml_path}")


def yolo_label_to_class_id(label_path: Path) -> Optional[int]:
    """Return a single class id for a YOLO label file.

    If multiple classes exist in one image, choose the most frequent.
    """
    if not label_path.exists():
        return None

    counts: Dict[int, int] = {}
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            cls_id = int(float(parts[0]))
        except Exception:
            continue
        counts[cls_id] = counts.get(cls_id, 0) + 1

    if not counts:
        return None

    # pick most frequent; tie -> smallest id
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def prepare_imagefolder_from_yolo(
    yolo_root: Path,
    out_root: Path,
    mode: str,
    overwrite: bool,
) -> None:
    """Convert a YOLO classification-by-labels dataset into ImageFolder.

    Expected layout:
      yolo_root/
        data.yaml
        train/images, train/labels
        valid/images, valid/labels
        test/images,  test/labels
    """
    if overwrite:
        safe_rmtree(out_root)

    yaml_path = yolo_root / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {yaml_path}")

    names = parse_yolo_names_from_yaml(yaml_path)

    split_map = {
        "train": "train",
        "val": "valid",
        "test": "test",
    }

    rows = [f"Source: {yolo_root}", f"Output: {out_root}", f"Mode: {mode}", "", "split\tclass\tcount\tskipped_missing_label\tskipped_multi_class"]

    for out_split, in_split in split_map.items():
        img_dir = yolo_root / in_split / "images"
        lbl_dir = yolo_root / in_split / "labels"
        if not img_dir.exists():
            raise FileNotFoundError(f"images dir not found: {img_dir}")
        if not lbl_dir.exists():
            raise FileNotFoundError(f"labels dir not found: {lbl_dir}")

        images = [p for p in img_dir.iterdir() if is_image_file(p)]
        missing_label = 0
        multi_class = 0
        counts: Dict[str, int] = {name: 0 for name in names}

        for img_path in images:
            label_path = lbl_dir / (img_path.stem + ".txt")
            cls_id = yolo_label_to_class_id(label_path)
            if cls_id is None:
                missing_label += 1
                continue
            if cls_id < 0 or cls_id >= len(names):
                missing_label += 1
                continue

            # detect multi-class (best-effort)
            if label_path.exists():
                unique_ids = set()
                for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        unique_ids.add(int(float(parts[0])))
                    except Exception:
                        continue
                if len(unique_ids) > 1:
                    multi_class += 1

            cls_name = names[cls_id]
            counts[cls_name] = counts.get(cls_name, 0) + 1
            dst = out_root / out_split / cls_name / img_path.name
            copy_or_link(img_path, dst, mode=mode)

        for cls_name in names:
            rows.append(f"{out_split}\t{cls_name}\t{counts.get(cls_name, 0)}\t{missing_label}\t{multi_class}")

    write_summary(out_root / "prepare_summary.tsv", rows)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Prepare 3 Kaggle brain tumor datasets into ImageFolder train/val/test structure (no changes to existing training code)."
    )

    p.add_argument(
        "--data_root",
        default="./data",
        help="Workspace data root (default: ./data)",
    )
    p.add_argument(
        "--out_root",
        default="./data_kaggle_prepared",
        help="Output root (default: ./data_kaggle_prepared)",
    )
    p.add_argument(
        "--dataset",
        default="all",
        choices=["all", "brain-mri", "tumor-123", "medical-yolo"],
        help="Which dataset to prepare",
    )

    p.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    p.add_argument(
        "--split",
        default="0.8,0.1,0.1",
        help="Split ratios for datasets without predefined splits: train,val,test (default: 0.8,0.1,0.1)",
    )
    p.add_argument(
        "--mode",
        default="copy",
        choices=["copy", "hardlink"],
        help="How to materialize output files (default: copy)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing prepared output before writing",
    )

    return p


def parse_split(split_str: str) -> SplitRatios:
    parts = [x.strip() for x in split_str.split(",") if x.strip()]
    if len(parts) != 3:
        raise ValueError("--split must have 3 comma-separated numbers: train,val,test")
    try:
        vals = [float(x) for x in parts]
    except Exception:
        raise ValueError("--split values must be numbers")
    return SplitRatios(train=vals[0], val=vals[1], test=vals[2])


def main() -> None:
    args = build_arg_parser().parse_args()
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    ratios = parse_split(args.split)

    # 1) Brain MRI Images for Brain Tumor Detection (yes/no)
    if args.dataset in {"all", "brain-mri"}:
        src = data_root / "Brain MRI Images for Brain Tumor Detection"
        dst = out_root / "Brain_MRI_Images_for_Brain_Tumor_Detection"
        prepare_imagefolder_from_class_dirs(
            src_root=src,
            class_dirs=["yes", "no"],
            out_root=dst,
            ratios=ratios,
            seed=args.seed,
            mode=args.mode,
            overwrite=args.overwrite,
        )

    # 2) Brain Tumor Image Dataset (1/2/3)
    if args.dataset in {"all", "tumor-123"}:
        src = data_root / "Brain Tumor Image Dataset"
        dst = out_root / "Brain_Tumor_Image_Dataset"
        prepare_imagefolder_from_class_dirs(
            src_root=src,
            class_dirs=["1", "2", "3"],
            out_root=dst,
            ratios=ratios,
            seed=args.seed,
            mode=args.mode,
            overwrite=args.overwrite,
        )

    # 3) Medical Image DataSet Brain Tumor Detection (YOLO)
    if args.dataset in {"all", "medical-yolo"}:
        yolo_root = (
            data_root
            / "Medical Image DataSet Brain Tumor Detection"
            / "BrainTumor"
            / "BrainTumorYolov11"
        )
        dst = out_root / "Medical_Image_DataSet_Brain_Tumor_Detection"
        prepare_imagefolder_from_yolo(
            yolo_root=yolo_root,
            out_root=dst,
            mode=args.mode,
            overwrite=args.overwrite,
        )

    print("✅ 完成：已生成 ImageFolder 结构数据集。")
    print(f"输出目录: {out_root.resolve()}")


if __name__ == "__main__":
    main()
