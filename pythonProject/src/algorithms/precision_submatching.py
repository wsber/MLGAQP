#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
precision_submatching.py
封装精确子图匹配（SubgraphMatching.out）为类：
- 支持执行外部程序并自动解析结果
- 输出解析为结构化字典
"""
import os
import sys
import csv
import json
import subprocess
import shlex
import re
import time
import argparse
from typing import Dict, Any, Optional

# 动态定位项目根目录 (自动向上追溯 3 级到达 PROXY)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))

class ExactSubgraphMatcher:
    """
    精确子图匹配类（封装 SubgraphMatching.out 调用和结果解析）
    """
    def __init__(self, exe_path: str, default_args=None, timeout: int = 300):
        self.exe_path = exe_path
        self.default_args = default_args or [
            "-filter", "GQL",
            "-order", "GQL",
            "-engine", "LFTJ",
            "-num", "MAX"
        ]
        self.timeout = timeout

    @staticmethod
    def _to_number(s: str):
        try:
            if "." in s or "e" in s or "E" in s:
                return float(s)
            else:
                return int(s)
        except Exception:
            return s

    @classmethod
    def _parse_output(cls, text: str) -> Dict[str, Any]:
        out = {"raw": text}

        # 命令行参数
        cmdline_block = re.search(r"Command Line:(.*?)(?:\n-+|\n\n)", text, flags=re.S)
        if cmdline_block:
            params = {}
            for line in cmdline_block.group(1).splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    params[k.strip()] = v.strip()
            out["cmdline"] = params

        # Query graph meta
        m = re.search(
            r"Query Graph Meta Information\s*\n\|V\|:\s*(\d+),\s*\|E\|:\s*(\d+),\s*\|Σ\|:\s*(\d+).*?Max Degree:\s*([\d\.]+),\s*Max Label Frequency:\s*([\d\.]+)",
            text, flags=re.S,
        )
        if m:
            out["query_graph"] = {
                "V": int(m.group(1)), "E": int(m.group(2)), "Sigma": int(m.group(3)),
                "MaxDegree": cls._to_number(m.group(4)), "MaxLabelFrequency": cls._to_number(m.group(5)),
            }

        # Data graph meta
        m = re.search(
            r"\|V\|:\s*([\d,]+).*?\|E\|:\s*([\d,]+).*?\|Σ\|:\s*([\d,]+).*?Max Degree:\s*([\d\.]+),\s*Max Label Frequency:\s*([\d\.]+)",
            text, flags=re.S,
        )
        if m:
            out["data_graph"] = {
                "V": int(m.group(1).replace(",", "")), "E": int(m.group(2).replace(",", "")),
                "Sigma": int(m.group(3).replace(",", "")), "MaxDegree": cls._to_number(m.group(4)),
                "MaxLabelFrequency": cls._to_number(m.group(5)),
            }

        # CoreTables
        core_tables = {}
        for mm in re.finditer(r"CoreTable\s+(\d+-\d+):\s*([\d\.eE\-]+)", text):
            core_tables[mm.group(1)] = cls._to_number(mm.group(2))
        if core_tables:
            out["core_tables"] = core_tables

        # 其他关键指标
        for key, pattern in {
            "total_cardinality": r"Total Cardinality:\s*([\d\.eE\-]+)",
            "query_plan": r"Query Plan:\s*([\d\t ]+)",
            "memory_mb": r"Memory cost\s*\(MB\)\s*:\s*([\d\.eE\-]+)",
            "embeddings": r"#Embeddings:\s*([\d\.eE\-]+)",
            "call_count": r"Call Count:\s*([\d\.eE\-]+)",
            "per_call_ns": r"Per Call Count Time.*?:\s*([\d\.eE\-]+)",
        }.items():
            m = re.search(pattern, text)
            if m:
                val = m.group(1)
                out[key] = [int(x) for x in val.split()] if key == "query_plan" else cls._to_number(val)

        # 时间部分
        timings = {}
        for k in [
            "Load graphs time", "Filter vertices time", "Build table time",
            "Generate query plan time", "Enumerate time", "Preprocessing time", "Total time",
        ]:
            m = re.search(fr"{re.escape(k)}.*?:\s*([\d\.eE\-]+)", text)
            if m:
                timings[k] = float(m.group(1))
        out["timings"] = timings
        out["ended"] = "End." in text
        return out

    def run(
            self,
            data_graph: str,
            query_graph: str,
            result_file: str = None,
            save_results: bool = False,
            extra_args: Optional[list] = None,
            timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        args = extra_args or self.default_args
        timeout = timeout or self.timeout

        cmd = [self.exe_path, "-d", data_graph, "-q", query_graph] + args
        cmd_str = " ".join(shlex.quote(x) for x in cmd)

        env = os.environ.copy()
        if save_results:
            env["SAVE_RESULTS"] = "1"
            if result_file:
                env["RESULT_FILE"] = result_file

        start = time.time()
        try:
            proc = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=timeout, env=env
            )
            elapsed = time.time() - start
            parsed = self._parse_output(proc.stdout)
            return {
                "command": cmd_str, "returncode": proc.returncode,
                "stdout": proc.stdout, "stderr": proc.stderr,
                "elapsed": elapsed, "parsed": parsed,
            }
        except subprocess.TimeoutExpired as e:
            elapsed_time = time.time() - start
            return {
                "command": cmd_str, "returncode": None,
                "stdout": (e.stdout or b"").decode("utf-8", errors="ignore") if isinstance(e.stdout, bytes) else (e.stdout or ""),
                "stderr": (e.stderr or b"").decode("utf-8", errors="ignore") if isinstance(e.stderr, bytes) else (e.stderr or ""),
                "elapsed": elapsed_time, "parsed": {},
            }
        except Exception as e:
            elapsed_time = time.time() - start
            return {
                "command": cmd_str, "returncode": None,
                "stdout": "", "stderr": f"[Exception: {repr(e)}]",
                "elapsed": elapsed_time, "parsed": {},
            }

    def run_batch(
            self,
            data_graph: str,
            query_graph_dir: str,
            output_dir: str,
            ans_suffix: str = "",
            result_subdir: str = "structure_result",
    ):
        os.makedirs(output_dir, exist_ok=True)
        result_dir = os.path.join(output_dir, result_subdir)
        os.makedirs(result_dir, exist_ok=True)

        # 动态命名输出文件 (支持 _count, _sum 隔离)
        sfx = f"_{ans_suffix}" if ans_suffix else ""
        summary_csv = os.path.join(output_dir, f"parler_ans{sfx}.csv")
        summary_txt = os.path.join(output_dir, f"parler_ans{sfx}.txt")

        if not os.path.exists(query_graph_dir):
            print(f"[Error] 查询图目录不存在: {query_graph_dir}")
            return

        query_graph_files = sorted(
            [os.path.join(query_graph_dir, f) for f in os.listdir(query_graph_dir) if f.endswith(".graph")]
        )
        print(f"📂 共检测到 {len(query_graph_files)} 个查询文件在 {query_graph_dir}")

        with open(summary_csv, "w", newline="") as fout_csv, open(summary_txt, "w") as fout_txt:
            writer = csv.writer(fout_csv)
            writer.writerow(["QueryGraph", "Elapsed(ms)", "Cardinality"])

            for query_path in query_graph_files:
                query_name = os.path.basename(query_path)
                result_file = os.path.join(result_dir, f"{query_name}_matches.csv")

                print(f"▶ 正在精确枚举: {query_name} ...", end=" ", flush=True)
                result = self.run(
                    data_graph=data_graph,
                    query_graph=query_path,
                    result_file=result_file,
                    save_results=True,
                )

                parsed = result.get("parsed", {})
                total_time_s = parsed.get("timings", {}).get("Total time")
                elapsed_ms = round(
                    (total_time_s if total_time_s is not None else result.get("elapsed", 0)) * 1000, 3
                )
                cardinality = parsed.get("embeddings", -1)
                if isinstance(cardinality, float):
                    cardinality = int(cardinality)

                writer.writerow([query_name, elapsed_ms, cardinality])
                fout_txt.write(f"{query_name} {elapsed_ms}ms {cardinality}\n")
                print(f"✅ 完成! 耗时: {elapsed_ms} ms, 基数(真值): {cardinality}")

        print(f"\n📊 真值与枚举统计已生成：\n - {summary_csv}\n - {summary_txt}")

# ============================================================
# 命令行调度入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exact Subgraph Matching & Ground Truth Generator (ENUM)")
    parser.add_argument("--base_dir", default=DEFAULT_PROJECT_ROOT, help="项目根目录")
    parser.add_argument("--dataset", required=True, choices=["parler", "parler-E", "amazon"], help="数据集名称")
    parser.add_argument("--agg_mode", default="count", choices=["count", "sum"], help="聚合类型")
    parser.add_argument("--exe_path", default=None, help="SubgraphMatching.out 可执行文件路径")
    parser.add_argument("--timeout", type=int, default=300, help="单个查询超时时间 (秒)")
    args = parser.parse_args()

    # 动态匹配可执行文件
    if args.exe_path is None:
        cands = [
            os.path.join(args.base_dir, "cProject", "SubgraphMatching", "build", "matching", "SubgraphMatching.out"),
            os.path.join(args.base_dir, "SubgraphMatching", "build", "matching", "SubgraphMatching.out"),
            "/home/wangshuo/projects/SubgraphMatching/build/matching/SubgraphMatching.out"
        ]
        exe_path = next((p for p in cands if os.path.exists(p)), cands[0])
    else:
        exe_path = args.exe_path

    if not os.path.exists(exe_path):
        print(f"[Error] 找不到精确匹配程序: {exe_path}，请先编译 SubgraphMatching 工程！")
        sys.exit(1)

    # 路径拼装
    dataset_base = os.path.join(args.base_dir, "datasets", args.dataset)
    data_graph_path = os.path.join(dataset_base, "data_graph", "parler.graph")
    output_dir = os.path.join(dataset_base, "ground_truth")

    # 针对 Amazon 做 query_graph_count / query_graph_sum 区分
    if args.dataset == "amazon":
        query_dir = os.path.join(dataset_base, f"query_graph_{args.agg_mode}")
        ans_suffix = args.agg_mode
    else:
        query_dir = os.path.join(dataset_base, "query_graph")
        ans_suffix = ""

    print("=" * 70)
    print(f"🚀 启动精确子图匹配 (ENUM / Ground Truth 生产引擎)")
    print(f"   • 目标数据集 : {args.dataset} ({args.agg_mode.upper()})")
    print(f"   • 数据大图   : {data_graph_path}")
    print(f"   • 查询图目录 : {query_dir}")
    print(f"   • 输出目录   : {output_dir}")
    print(f"   • 可执行程序 : {exe_path}")
    print("=" * 70)

    matcher = ExactSubgraphMatcher(exe_path=exe_path, timeout=args.timeout)
    matcher.run_batch(
        data_graph=data_graph_path,
        query_graph_dir=query_dir,
        output_dir=output_dir,
        ans_suffix=ans_suffix
    )