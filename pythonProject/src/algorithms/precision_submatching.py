#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exact_subgraph_matcher.py

封装精确子图匹配（SubgraphMatching.out）为类：
- 支持执行外部程序并自动解析结果
- 输出解析为结构化字典
"""
import csv
import json
import os
import subprocess
import shlex
import re
import time
from typing import Dict, Any, Optional


class ExactSubgraphMatcher:
    """
    精确子图匹配类（封装 SubgraphMatching.out 调用和结果解析）
    """

    def __init__(self, exe_path: str, default_args=None, timeout: int = 120):
        """
        初始化子图匹配器。

        :param exe_path: SubgraphMatching.out 可执行文件路径
        :param default_args: 默认参数列表（如 ["-filter","GQL","-order","GQL","-engine","LFTJ","-num","MAX"]）
        :param timeout: 默认超时时间（秒）
        """
        self.exe_path = exe_path
        self.default_args = default_args or [
            "-filter", "GQL",
            "-order", "GQL",
            "-engine", "LFTJ",
            "-num", "MAX"
        ]
        self.timeout = timeout

    # --------------------------
    # 工具函数
    # --------------------------
    @staticmethod
    def _to_number(s: str):
        try:
            if "." in s or "e" in s or "E" in s:
                return float(s)
            else:
                return int(s)
        except Exception:
            return s

    # --------------------------
    # 核心解析函数
    # --------------------------
    @classmethod
    def _parse_output(cls, text: str) -> Dict[str, Any]:
        """
        解析 SubgraphMatching 的输出文本为结构化字典。
        """
        out = {"raw": text}

        # ---- 命令行参数 ----
        cmdline_block = re.search(r"Command Line:(.*?)(?:\n-+|\n\n)", text, flags=re.S)
        if cmdline_block:
            params = {}
            for line in cmdline_block.group(1).splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    params[k.strip()] = v.strip()
            out["cmdline"] = params

        # ---- Query graph meta ----
        m = re.search(
            r"Query Graph Meta Information\s*\n\|V\|:\s*(\d+),\s*\|E\|:\s*(\d+),\s*\|Σ\|:\s*(\d+).*?Max Degree:\s*([\d\.]+),\s*Max Label Frequency:\s*([\d\.]+)",
            text,
            flags=re.S,
        )
        if m:
            out["query_graph"] = {
                "V": int(m.group(1)),
                "E": int(m.group(2)),
                "Sigma": int(m.group(3)),
                "MaxDegree": cls._to_number(m.group(4)),
                "MaxLabelFrequency": cls._to_number(m.group(5)),
            }

        # ---- Data graph meta ----
        m = re.search(
            r"\|V\|:\s*([\d,]+).*?\|E\|:\s*([\d,]+).*?\|Σ\|:\s*([\d,]+).*?Max Degree:\s*([\d\.]+),\s*Max Label Frequency:\s*([\d\.]+)",
            text,
            flags=re.S,
        )
        if m:
            out["data_graph"] = {
                "V": int(m.group(1).replace(",", "")),
                "E": int(m.group(2).replace(",", "")),
                "Sigma": int(m.group(3).replace(",", "")),
                "MaxDegree": cls._to_number(m.group(4)),
                "MaxLabelFrequency": cls._to_number(m.group(5)),
            }

        # ---- CoreTables ----
        core_tables = {}
        for mm in re.finditer(r"CoreTable\s+(\d+-\d+):\s*([\d\.eE\-]+)", text):
            core_tables[mm.group(1)] = cls._to_number(mm.group(2))
        if core_tables:
            out["core_tables"] = core_tables

        # ---- 其他关键指标 ----
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
                if key == "query_plan":
                    out[key] = [int(x) for x in val.split()]
                else:
                    out[key] = cls._to_number(val)

        # ---- 时间部分 ----
        timings = {}
        for k in [
            "Load graphs time",
            "Filter vertices time",
            "Build table time",
            "Generate query plan time",
            "Enumerate time",
            "Preprocessing time",
            "Total time",
        ]:
            m = re.search(fr"{re.escape(k)}.*?:\s*([\d\.eE\-]+)", text)
            if m:
                timings[k] = float(m.group(1))
        out["timings"] = timings

        out["ended"] = "End." in text
        return out

    # --------------------------
    # 主执行方法
    # --------------------------
    def run(
            self,
            data_graph: str,
            query_graph: str,
            result_file: str = None,  # 新增：指定输出文件
            save_results: bool = False,  # 新增：是否保存结果
            extra_args: Optional[list] = None,
            timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        执行精确子图匹配程序。
        """
        args = extra_args or self.default_args
        timeout = timeout or self.timeout

        cmd = [self.exe_path, "-d", data_graph, "-q", query_graph] + args
        cmd_str = " ".join(shlex.quote(x) for x in cmd)

        # 设置环境变量控制输出文件
        env = os.environ.copy()
        if save_results:
            env["SAVE_RESULTS"] = "1"
            if result_file:
                env["RESULT_FILE"] = result_file

        start = time.time()
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                env=env  # 使用带环境变量的子进程
            )
            elapsed = time.time() - start

            parsed = self._parse_output(proc.stdout)

            return {
                "command": cmd_str,
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "elapsed": elapsed,
                "parsed": parsed,
            }
        except subprocess.TimeoutExpired as e:
            elapsed_time = time.time() - start
            query_name = os.path.basename(query_graph)
            ans_path = os.path.join(os.path.dirname(result_file), "parler_ans.txt")

            # 写入固定占位结果
            with open(ans_path, "a") as fout:
                fout.write(f"{query_name} 9999.9ms -1\n")

            # 确保返回字典给 run_batch()
            return {
                "command": cmd_str,
                "returncode": None,
                "stdout": (e.stdout or b"").decode("utf-8", errors="ignore") if isinstance(e.stdout, bytes) else (
                            e.stdout or ""),
                "stderr": (e.stderr or b"").decode("utf-8", errors="ignore") if isinstance(e.stderr, bytes) else (
                            e.stderr or ""),
                "elapsed": elapsed_time,
                "parsed": {},  # 没有匹配结果
            }

        except Exception as e:
            elapsed_time = time.time() - start
            query_name = os.path.basename(query_graph)
            ans_path = os.path.join(os.path.dirname(result_file), "parler_ans.txt")

            # 写入固定占位结果
            with open(ans_path, "a") as fout:
                fout.write(f"{query_name} 9999.9ms -1\n")

            return {
                "command": cmd_str,
                "returncode": None,
                "stdout": "",
                "stderr": f"[Exception: {repr(e)}]",
                "elapsed": elapsed_time,
                "parsed": {},
            }

    def run_batch(
            self,
            data_graph: str,
            query_graph_dir: str,
            output_dir: str,
            result_subdir: str = "structure_result",
    ):
        """
        批量执行多个查询文件，并输出 summary_results.csv 和 summary_results.txt。

        :param data_graph: 数据图文件路径
        :param query_graph_dir: 查询图文件夹路径
        :param output_dir: 输出目录（包含summary结果）
        :param result_subdir: 匹配结果CSV子文件夹
        """
        import csv

        os.makedirs(output_dir, exist_ok=True)
        result_dir = os.path.join(output_dir, result_subdir)
        os.makedirs(result_dir, exist_ok=True)

        # 汇总文件路径
        summary_csv = os.path.join(output_dir, "parler_ans.csv")
        summary_txt = os.path.join(output_dir, "parler_ans.txt")

        # 获取所有查询文件
        query_graph_files = sorted(
            [os.path.join(query_graph_dir, f) for f in os.listdir(query_graph_dir) if f.endswith(".graph")]
        )

        print(f"📂 共检测到 {len(query_graph_files)} 个查询文件")

        # 打开两个汇总文件
        with open(summary_csv, "w", newline="") as fout_csv, open(summary_txt, "w") as fout_txt:
            writer = csv.writer(fout_csv)
            writer.writerow(["QueryGraph", "Elapsed(ms)", "Cardinality"])

            for query_path in query_graph_files:
                query_name = os.path.basename(query_path)
                result_file = os.path.join(result_dir, f"{query_name}_matches.csv")

                print(f"\n▶ 正在执行: {query_name}")
                result = self.run(
                    data_graph=data_graph,
                    query_graph=query_path,
                    result_file=result_file,
                    save_results=True,
                )

                # 1️⃣ 优先使用解析结果中的总耗时
                parsed = result.get("parsed", {})
                total_time_s = parsed.get("timings", {}).get("Total time")
                # 如果未找到，则回退到全局elapsed时间
                elapsed_ms = round(
                    (total_time_s if total_time_s is not None else result.get("elapsed", 0)) * 1000,
                    3,
                )

                # 2️⃣ 匹配结果数量
                cardinality = parsed.get("embeddings", "N/A")
                if isinstance(cardinality, float):
                    cardinality = int(cardinality)

                # 3️⃣ 写入 CSV
                writer.writerow([query_name, elapsed_ms, cardinality])

                # 4️⃣ 写入 TXT（按格式：query_name 0.06ms 110264）
                fout_txt.write(f"{query_name} {elapsed_ms}ms {cardinality}\n")

                print(f"✅ {query_name}: {elapsed_ms} ms, Cardinality={cardinality}")

        print(f"\n📊 已生成：\n - {summary_csv}\n - {summary_txt}")


# ============================================================
# ✅ 示例调用
# ============================================================
if __name__ == "__main__":

    # #v2.5 使用批量执行接口
    matcher = ExactSubgraphMatcher(
        exe_path="/home/wangshuo/projects/SubgraphMatching/build/matching/SubgraphMatching.out",
        default_args=["-filter", "GQL", "-order", "GQL", "-engine", "LFTJ", "-num", "MAX"],
        timeout=300,
    )

    matcher.run_batch(
        data_graph="/home/wangshuo/resource/datasets/parler_data/dataset_one/ground_truth/data_graph/parler.graph",
        query_graph_dir="/home/wangshuo/resource/datasets/parler_data/dataset_one/ground_truth/query_graph",
        output_dir="/home/wangshuo/resource/datasets/parler_data/dataset_one/ground_truth",
    )

    # # V2.0
    # matcher = ExactSubgraphMatcher(
    #     exe_path="/home/wangshuo/projects/SubgraphMatching/build/matching/SubgraphMatching.out",
    #     default_args=["-filter", "GQL", "-order", "GQL", "-engine", "LFTJ", "-num", "MAX"],
    #     timeout=300,
    # )
    #
    # # === 路径配置 ===
    # data_graph = "/home/wangshuo/resource/datasets/parler_data/dataset_one/ground_truth/data_graph/parler.graph"
    # query_graph_dir = "/home/wangshuo/resource/datasets/parler_data/dataset_one/ground_truth/query_graph"
    # output_dir = "/home/wangshuo/resource/datasets/parler_data/dataset_one/ground_truth"
    # summary_csv = os.path.join(output_dir, "summary_results.csv")
    #
    # # 获取所有查询文件
    # query_graph_files = sorted(
    #     [os.path.join(query_graph_dir, f) for f in os.listdir(query_graph_dir) if f.endswith(".graph")]
    # )
    #
    # os.makedirs(output_dir, exist_ok=True)
    #
    # with open(summary_csv, "w", newline="") as fout:
    #     writer = csv.writer(fout)
    #     writer.writerow(["QueryGraph", "Elapsed(ms)", "Cardinality"])
    #
    #     for query_path in query_graph_files:
    #         query_name = os.path.basename(query_path)
    #         result_file = os.path.join(output_dir, f"result/{query_name}_matches.csv")
    #
    #         print(f"\n▶ 正在执行: {query_name}")
    #         result = matcher.run(
    #             data_graph=data_graph,
    #             query_graph=query_path,
    #             result_file=result_file,
    #             save_results=True
    #         )
    #
    #         elapsed_ms = round(result.get("elapsed", 0) * 1000, 3)
    #         cardinality = result.get("total_cardinality", "N/A")
    #
    #         # 写入汇总结果
    #         writer.writerow([query_name, elapsed_ms, cardinality])
    #         print(f"✅ {query_name}: {elapsed_ms} ms, Cardinality={cardinality}")
    #
    # print(f"\n📄 已保存汇总文件: {summary_csv}")

    # V1.0
    # matcher = ExactSubgraphMatcher(
    #     exe_path="/home/wangshuo/projects/SubgraphMatching/build/matching/SubgraphMatching.out",
    #     default_args=["-filter", "GQL", "-order", "GQL", "-engine", "LFTJ", "-num", "MAX"],
    #     timeout=120,
    # )
    #
    # data_graph = "/home/wangshuo/projects/SubgraphMatching/test/parler/parler.graph"
    # query_graph = "/home/wangshuo/projects/SubgraphMatching/test/parler/query_dense_1_1.graph"
    #
    # result = matcher.run(data_graph, query_graph)
    #
    # print("\n=== ✅ 运行结果 ===")
    # print("命令行:", result["command"])
    # print("退出码:", result["returncode"])
    # print("耗时(秒):", result["elapsed"])
    #
    # parsed = result["parsed"]
    # print("\n=== 🧩 解析字段一览 ===")
    # for key in parsed.keys():
    #     print(f" - {key}")
    #
    # print("\n=== 🧠 详细解析内容 ===")
    # # 使用 JSON 美化格式打印
    # print(json.dumps(parsed, indent=2, ensure_ascii=False))
    #
    # print("\n=== 📊 关键指标 ===")
    # print("匹配数 (#Embeddings):", parsed.get("embeddings", "N/A"))
    # print("总基数 (Total Cardinality):", parsed.get("total_cardinality", "N/A"))
    # print("总耗时 (Total time):", parsed.get("timings", {}).get("Total time", "N/A"), "秒")
    # print("内存消耗 (Memory MB):", parsed.get("memory_mb", "N/A"))




    # V0.5
    # matcher = ExactSubgraphMatcher(
    #     exe_path="/home/wangshuo/projects/SubgraphMatching/build/matching/SubgraphMatching.out",
    #     default_args=["-filter", "GQL", "-order", "GQL", "-engine", "LFTJ", "-num", "MAX"],
    #     timeout=120,
    # )
    #
    # data_graph = "/home/wangshuo/projects/SubgraphMatching/test/parler/parler.graph"
    # query_graph = "/home/wangshuo/projects/SubgraphMatching/test/parler/query_dense_1_1.graph"
    # output_file = "/home/wangshuo/projects/SubgraphMatching/result/out.csv"
    #
    # result = matcher.run(data_graph, query_graph, save_results=True, result_file=output_file)
    #
    # print("匹配数 (#Embeddings):", result["parsed"].get("embeddings", "N/A"))
    # print("结果已保存到:", output_file)
    #