#!/usr/bin/env python3
"""
rename_summary_cols_runner.py

在 Jupyter 或命令行中运行：
  # dry-run（只预览）：
  %run pythonProject/src/Structure_first/exp/rename_summary_cols_runner.py --path /home/wangshuo/resource/datasets/parler_data/dataset_one/results/result_summarys/proxy_4b1 --dry-run

  # 真正执行（会备份原文件）：
  %run pythonProject/src/Structure_first/exp/rename_summary_cols_runner.py --path /home/wangshuo/resource/datasets/parler_data/dataset_one/results/result_summarys/proxy_4b1

功能：
 - 在指定目录下查找所有 .csv 文件（可选递归）
 - 将列名按映射重命名
 - 在修改前备份原文件到同目录的备份文件夹
 - 支持 dry-run 模式，仅打印将要修改的文件/映射

注意：在 Jupyter 中，使用 `%run <script>` 或在 notebook code cell 中用 `!python3 <script>` 运行。
"""

import argparse
import time
from pathlib import Path
import shutil
import pandas as pd
import sys

COL_MAP = {
    "T_hat_mean": "T_hat",
    "T_hat_std": "n_post",
    "Qerror_mean": "Qerror",
    "Qerror_std": "n_comment",
}


def process_csv_file(path: Path, backup_dir: Path, dry_run: bool):
    """处理单个 CSV 文件：备份并按 COL_MAP 重命名（如果需要）。"""
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return False, f"read_error:{e}"

    cols_before = list(df.columns)
    rename_map = {k: v for k, v in COL_MAP.items() if k in df.columns}
    if not rename_map:
        return False, "no_change"

    if dry_run:
        return True, f"dry_run:{rename_map}"

    # 备份
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_target = backup_dir / path.name
    try:
        shutil.copy2(path, backup_target)
    except Exception as e:
        return False, f"backup_error:{e}"

    # 重命名并写回
    try:
        df = df.rename(columns=rename_map)
        # 尝试保持原列顺序（用备份文件的列顺序为基准）
        try:
            orig_cols = list(pd.read_csv(backup_target).columns)
            new_cols = [COL_MAP.get(c, c) for c in orig_cols]
            if set(new_cols) == set(df.columns):
                df = df.reindex(columns=new_cols)
        except Exception:
            pass
        df.to_csv(path, index=False)
        return True, "modified"
    except Exception as e:
        return False, f"write_error:{e}"


def run_for_path(target_path: str, recursive: bool = False, dry_run: bool = False, make_backup: bool = True):
    target = Path(target_path)
    if not target.exists():
        print(f"[FATAL] 路径不存在: {target}")
        return

    if target.is_file() and target.suffix.lower() == ".csv":
        files = [target]
    else:
        if recursive:
            files = list(target.rglob("*.csv"))
        else:
            files = list(target.glob("*.csv"))

    if not files:
        print(f"[WARN] 未找到 CSV 文件: {target}")
        return

    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = None
    if make_backup and not dry_run:
        backup_dir = target.parent / f"backup_result_summarys_{ts}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] 备份目录: {backup_dir}")

    summary = {"modified": 0, "skipped": 0, "errors": 0, "dry_run": 0}

    for f in files:
        ok, reason = process_csv_file(f, backup_dir if backup_dir else (target.parent / f"backup_result_summarys_{ts}"), dry_run)
        if reason == "modified":
            summary["modified"] += 1
            print(f"[OK] 修改: {f.name}")
        elif reason == "no_change":
            summary["skipped"] += 1
            print(f"[SKIP] 无需修改: {f.name}")
        elif isinstance(reason, str) and reason.startswith("dry_run"):
            summary["dry_run"] += 1
            print(f"[DRY] {f.name} -> {reason}")
        else:
            summary["errors"] += 1
            print(f"[ERROR] {f.name} -> {reason}")

    print("\n[SUMMARY]")
    print(f"  modified: {summary['modified']}")
    print(f"  skipped : {summary['skipped']}")
    print(f"  dry_run : {summary['dry_run']}")
    print(f"  errors  : {summary['errors']}")
    if backup_dir and summary['modified'] > 0:
        print(f"备份已保存到: {backup_dir}")


def build_arg_parser():
    p = argparse.ArgumentParser(description="批量重命名 results_summary CSV 列名 (支持在 Jupyter 中运行)")
    p.add_argument("--path", "-p", required=True, help="目标目录或单个 csv 文件")
    p.add_argument("--recursive", "-r", action="store_true", help="递归处理子目录中的 csv 文件")
    p.add_argument("--dry-run", dest="dry_run", action="store_true", help="仅打印将要修改的文件/列（不写入）")
    p.add_argument("--no-backup", dest="backup", action="store_false", help="不备份原文件（不推荐）")
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    run_for_path(args.path, recursive=args.recursive, dry_run=args.dry_run, make_backup=args.backup)
