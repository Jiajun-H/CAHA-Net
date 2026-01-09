import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent


def _configure_stdio() -> None:
    # Windows 控制台/conda run 下可能出现编码不一致，导致中文输出异常。
    # 这里尽量统一为 UTF-8，并在无法编码时用替换策略保证不崩。
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


@dataclass(frozen=True)
class RunResult:
    script: str
    returncode: int
    seconds: float


def discover_train_scripts(root: Path) -> List[str]:
    scripts = []
    for p in sorted(root.glob("train*.py")):
        name = p.name
        if name in {"run_train_suite.py"}:
            continue
        if name.startswith("train_") or name == "train.py" or name.startswith("train"):
            scripts.append(name)
    return scripts


def parse_csv_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    items = [x.strip() for x in value.split(",")]
    return [x for x in items if x]


def parse_selection_expr(expr: str, max_index: int) -> List[int]:
    """Parse selection like: 1,3-5,8"""
    expr = expr.strip().lower()
    if not expr:
        return []
    parts = [p.strip() for p in expr.split(",") if p.strip()]
    indices: List[int] = []
    for part in parts:
        m = re.fullmatch(r"(\d+)(?:\s*-\s*(\d+))?", part)
        if not m:
            raise ValueError(f"无法解析选择表达式: {part}")
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        if a < 1 or b < 1 or a > max_index or b > max_index:
            raise ValueError(f"索引超出范围: {part} (有效范围 1-{max_index})")
        if a <= b:
            indices.extend(range(a, b + 1))
        else:
            indices.extend(range(a, b - 1, -1))
    # 去重保持顺序
    seen = set()
    out = []
    for i in indices:
        if i not in seen:
            out.append(i)
            seen.add(i)
    return out


def interactive_pick(scripts: Sequence[str]) -> List[str]:
    print("\n可用训练脚本:")
    for i, s in enumerate(scripts, start=1):
        print(f"  {i:>2}. {s}")

    print("\n选择要运行的脚本:")
    print("- 输入 all 运行全部")
    print("- 输入 1,3-5 这种格式选择")
    print("- 输入 q 退出")

    while True:
        raw = input("你的选择> ").strip().lower()
        if raw in {"q", "quit", "exit"}:
            return []
        if raw in {"all", "a"}:
            return list(scripts)
        try:
            idxs = parse_selection_expr(raw, max_index=len(scripts))
            return [scripts[i - 1] for i in idxs]
        except Exception as e:
            print(f"输入无效: {e}")


def apply_include_exclude(
    scripts: Sequence[str],
    include: Sequence[str],
    exclude: Sequence[str],
) -> List[str]:
    scripts_set = list(scripts)

    if include:
        wanted = []
        missing = []
        for s in include:
            s = s.strip()
            if not s:
                continue
            if s in scripts_set:
                wanted.append(s)
            else:
                # 允许用户只写不带 .py 的名字
                if not s.endswith(".py") and (s + ".py") in scripts_set:
                    wanted.append(s + ".py")
                else:
                    missing.append(s)
        if missing:
            raise RuntimeError(f"include 里有未找到的脚本: {missing}")
        scripts_set = wanted

    if exclude:
        exclude_norm = set()
        for s in exclude:
            s = s.strip()
            if not s:
                continue
            exclude_norm.add(s)
            if not s.endswith(".py"):
                exclude_norm.add(s + ".py")
        scripts_set = [s for s in scripts_set if s not in exclude_norm]

    return scripts_set


def run_one(script: str, *, python: str, cwd: Path) -> RunResult:
    start = time.time()
    proc = subprocess.run([python, script], cwd=str(cwd))
    end = time.time()
    return RunResult(script=script, returncode=proc.returncode, seconds=end - start)


def main(argv: Optional[Sequence[str]] = None) -> int:
    _configure_stdio()
    parser = argparse.ArgumentParser(description="一键连续运行多个 train*.py 脚本")
    parser.add_argument("--list", action="store_true", help="列出可用脚本并退出")
    parser.add_argument("--all", action="store_true", help="运行全部 train*.py")
    parser.add_argument(
        "--select",
        default=None,
        help="按序号选择，例如: 1,3-5。配合 --list 查看序号。",
    )
    parser.add_argument(
        "--include",
        default=None,
        help="仅运行指定脚本，逗号分隔，例如: train_resnet50.py,train_vmamba.py",
    )
    parser.add_argument(
        "--exclude",
        default=None,
        help="排除指定脚本，逗号分隔，例如: train_album.py,train5_inceptionv3.py",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="某个脚本失败后继续跑后面的脚本",
    )
    parser.add_argument(
        "--skip-on-error",
        action="store_true",
        help="遇到脚本报错时跳过该脚本并继续（在最终报告中标记为已跳过）。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将要运行的脚本，不实际执行",
    )
    parser.add_argument(
        "--python",
        default=None,
        help=(
            "指定用于执行各个 train 脚本的 Python 解释器路径。"
            "默认使用当前解释器(sys.executable)。"
            "例如: C:/miniforge3/envs/densenet/python.exe"
        ),
    )

    args = parser.parse_args(argv)

    scripts = discover_train_scripts(ROOT)
    if not scripts:
        print("未发现任何 train*.py")
        return 1

    if args.list:
        print("可用训练脚本:")
        for i, s in enumerate(scripts, start=1):
            print(f"{i:>2}. {s}")
        return 0

    include = parse_csv_list(args.include)
    exclude = parse_csv_list(args.exclude)

    selected: List[str]
    if args.all:
        selected = list(scripts)
    elif args.select:
        idxs = parse_selection_expr(args.select, max_index=len(scripts))
        selected = [scripts[i - 1] for i in idxs]
    elif include or exclude:
        selected = apply_include_exclude(scripts, include, exclude)
    else:
        selected = interactive_pick(scripts)

    if not selected:
        print("未选择任何脚本，退出。")
        return 0

    print("\n将要运行的脚本(按顺序):")
    for s in selected:
        print(f"- {s}")

    if args.dry_run:
        print("\n(dry-run) 未执行任何脚本。")
        return 0

    python = args.python or sys.executable
    print(f"\n使用 Python: {python}")
    print(f"工作目录: {ROOT}")

    results: List[RunResult] = []
    skipped: List[str] = []
    failed: List[RunResult] = []
    started = time.time()

    for idx, script in enumerate(selected, start=1):
        print("\n" + "=" * 70)
        print(f"[{idx}/{len(selected)}] 开始: {script}")
        print("=" * 70)

        r = run_one(script, python=python, cwd=ROOT)
        if r.returncode == 0:
            results.append(r)
            status = "✅ 成功"
            print(f"[{idx}/{len(selected)}] 结束: {script} | {status} | 用时 {r.seconds:.1f}s")
        else:
            # 处理失败
            if args.skip_on_error:
                skipped.append(script)
                status = f"❌ 跳过(rc={r.returncode})"
                print(f"[{idx}/{len(selected)}] 结束: {script} | {status} | 用时 {r.seconds:.1f}s")
                # continue to next script
                continue
            elif args.continue_on_error:
                failed.append(r)
                status = f"❌ 失败(rc={r.returncode})"
                print(f"[{idx}/{len(selected)}] 结束: {script} | {status} | 用时 {r.seconds:.1f}s")
                continue
            else:
                results.append(r)
                status = f"❌ 失败(rc={r.returncode})"
                print(f"[{idx}/{len(selected)}] 结束: {script} | {status} | 用时 {r.seconds:.1f}s")
                print("检测到失败，已停止。可使用 --continue-on-error 或 --skip-on-error 继续跑后续脚本。")
                break

    total = time.time() - started

    print("\n" + "#" * 70)
    print("运行汇总")
    print("#" * 70)
    succeeded = [r for r in results if r.returncode == 0]
    failed_final = failed + [r for r in results if r.returncode != 0]

    print("\n运行成功:")
    for r in succeeded:
        print(f"- {r.script}: {r.seconds:.1f}s")

    if failed_final:
        print("\n运行失败 (已记录):")
        for r in failed_final:
            print(f"- {r.script}: rc={r.returncode}, {r.seconds:.1f}s")

    if skipped:
        print("\n已跳过的脚本:")
        for s in skipped:
            print(f"- {s}")

    total_ok = len(succeeded)
    total_attempted = len(succeeded) + len(failed_final) + len(skipped)
    print(f"\n完成: {total_ok}/{total_attempted} 成功 | 总用时 {total/60:.1f} min")

    return 0 if total_ok == total_attempted else 1


if __name__ == "__main__":
    raise SystemExit(main())
