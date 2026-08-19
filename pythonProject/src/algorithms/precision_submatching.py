#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import csv
import subprocess
import shlex
import re
import time
import argparse
from typing import Dict, Any, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))

class ExactSubgraphMatcher:
    def __init__(self, exe_path: str, default_args=None, timeout: int = 300):
        self.exe_path = exe_path
        self.default_args = default_args or ["-filter", "GQL", "-order", "GQL", "-engine", "LFTJ", "-num", "MAX"]
        self.timeout = timeout

    @staticmethod
    def _to_number(s: str):
        try:
            return float(s) if ("." in s or "e" in s or "E" in s) else int(s)
        except Exception:
            return s

    @classmethod
    def _parse_output(cls, text: str) -> Dict[str, Any]:
        out = {"raw": text}
        for key, pattern in {
            "total_cardinality": r"Total Cardinality:\s*([\d\.eE\-]+)",
            "embeddings": r"#Embeddings:\s*([\d\.eE\-]+)",
        }.items():
            m = re.search(pattern, text)
            if m: out[key] = cls._to_number(m.group(1))

        timings = {}
        for k in ["Load graphs time", "Enumerate time", "Total time"]:
            m = re.search(fr"{re.escape(k)}.*?:\s*([\d\.eE\-]+)", text)
            if m: timings[k] = float(m.group(1))
        out["timings"] = timings
        return out

    def run(self, data_graph: str, query_graph: str, result_file: str = None, save_results: bool = False) -> Dict[str, Any]:
        cmd = [self.exe_path, "-d", data_graph, "-q", query_graph] + self.default_args
        env = os.environ.copy()
        if save_results:
            env["SAVE_RESULTS"] = "1"
            if result_file: env["RESULT_FILE"] = result_file

        start = time.time()
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=self.timeout, env=env)
            elapsed = time.time() - start
            parsed = self._parse_output(proc.stdout)
            return {"elapsed": elapsed, "parsed": parsed, "returncode": proc.returncode}
        except Exception as e:
            return {"elapsed": time.time() - start, "parsed": {}, "returncode": None}

    def run_batch(self, data_graph: str, query_graph_dir: str, output_dir: str, result_subdir: str = "structure_result"):
        os.makedirs(output_dir, exist_ok=True)
        result_dir = os.path.join(output_dir, result_subdir)
        os.makedirs(result_dir, exist_ok=True)

        summary_csv = os.path.join(output_dir, "parler_ans.csv")
        summary_txt = os.path.join(output_dir, "parler_ans.txt")

        query_graph_files = sorted([os.path.join(query_graph_dir, f) for f in os.listdir(query_graph_dir) if f.endswith(".graph")])
        print(f"📂 共检测到 {len(query_graph_files)} 个查询文件在: {query_graph_dir}")

        with open(summary_csv, "w", newline="") as fout_csv, open(summary_txt, "w") as fout_txt:
            writer = csv.writer(fout_csv)
            writer.writerow(["QueryGraph", "Elapsed(ms)", "Cardinality"])

            for query_path in query_graph_files:
                query_name = os.path.basename(query_path)
                result_file = os.path.join(result_dir, f"{query_name}_matches.csv")

                print(f"▶ 正在精确匹配: {query_name:<20} ...", end=" ", flush=True)
                result = self.run(data_graph=data_graph, query_graph=query_path, result_file=result_file, save_results=True)

                parsed = result.get("parsed", {})
                total_time_s = parsed.get("timings", {}).get("Total time")
                elapsed_ms = round((total_time_s if total_time_s is not None else result.get("elapsed", 0)) * 1000, 3)
                cardinality = parsed.get("embeddings", -1)
                if isinstance(cardinality, float): cardinality = int(cardinality)

                writer.writerow([query_name, elapsed_ms, cardinality])
                fout_txt.write(f"{query_name} {elapsed_ms}ms {cardinality}\n")
                print(f"✅ 完成! 耗时: {elapsed_ms:>8.2f} ms | 匹配数: {cardinality}")

        print(f"\n📊 真值与结构匹配结果已生成至:\n - {summary_csv}\n - {summary_txt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exact Subgraph Matching (ENUM Baseline)")
    parser.add_argument("--base_dir", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--dataset", required=True, choices=["parler", "parler-E", "amazon"])
    parser.add_argument("--exe_path", default=None)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    if args.exe_path is None:
        cands = [
            os.path.join(args.base_dir, "cProject", "SubgraphMatching", "build", "matching", "SubgraphMatching.out"),
            os.path.join(args.base_dir, "SubgraphMatching", "build", "matching", "SubgraphMatching.out"),
            "/home/wangshuo/projects/SubgraphMatching/build/matching/SubgraphMatching.out"
        ]
        exe_path = next((p for p in cands if os.path.exists(p)), cands[0])
    else:
        exe_path = args.exe_path

    dataset_base = os.path.join(args.base_dir, "datasets", args.dataset)
    data_graph_path = os.path.join(dataset_base, "data_graph", "parler.graph")
    output_dir = os.path.join(dataset_base, "ground_truth")

    # Amazon 优先使用全量子图目录 query_graph_count，若不存在则回退至 query_graph
    if args.dataset == "amazon":
        q_count_dir = os.path.join(dataset_base, "query_graph_count")
        query_dir = q_count_dir if os.path.exists(q_count_dir) else os.path.join(dataset_base, "query_graph")
    else:
        query_dir = os.path.join(dataset_base, "query_graph")

    matcher = ExactSubgraphMatcher(exe_path=exe_path, timeout=args.timeout)
    matcher.run_batch(data_graph=data_graph_path, query_graph_dir=query_dir, output_dir=output_dir)