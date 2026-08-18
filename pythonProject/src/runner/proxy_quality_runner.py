import os
import sys
import json
import math
import argparse
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==============================================================================
# 1. 动态自动定位项目根目录 (彻底告别硬编码绝对路径)
# ==============================================================================
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "../../.."))

if DEFAULT_PROJECT_ROOT not in sys.path:
    sys.path.append(DEFAULT_PROJECT_ROOT)

from pythonProject.src.algorithms.proxy_sample import ProxyStratifiedSampler

# ==============================================================================
# 2. 代理模型质量字典配置
# ==============================================================================
# 单谓词模式配置 (针对 Parler)
SINGLE_PRED_PROXIES = {
    "POSSA_Q0_Worst": "worst_proxy_probability",
    "POSSA_Q1_Low": "ML1_proxy1b_probability",
    "POSSA_Q2_Med": "ML1_proxy2b_probability",
    "POSSA_Q3_High": "ML1_proxy6b_probability",
    "POSSA_Q4_Best": "ML1_proxy4b_probability"
}

# 多谓词模式配置 (针对 Parler-E / Amazon)
MULTI_PRED_PROXIES = {
    "POSSA_Q1_Best": ("ML1_proxy4b_probability", "ML2_proxy1_probability"),
    "POSSA_Q2_High": ("ML1_proxy6b_probability", "ML2_proxy4_probability"),
    "POSSA_Q3_Med":  ("ML1_proxy2b_probability", "ML2_proxy2_probability"),
    "POSSA_Q4_Low":  ("ML1_proxy1b_probability", "ML2_proxy2d2_probability"),
}

# ==============================================================================
# 3. 核心计算子进程函数
# ==============================================================================
def _process_ablation_single_file(
    agg_file: str,
    aggregated_dir: str,
    all_t_true: dict,
    target_ticks: list,
    run_times: int,
    mode: str
):
    """单个核心实例文件的多质量 Proxy 评估"""
    base = agg_file.replace("aggregated_list_", "").replace("aggregated_wide_", "").replace(".csv", "")
    query_basename = base + ".graph"

    T_true = all_t_true.get(query_basename)
    if T_true is None or T_true == 0:
        return []

    filepath = os.path.join(aggregated_dir, agg_file)
    file_records = []

    # 选取对应的 Proxy 字典
    proxy_dict = SINGLE_PRED_PROXIES if mode == "single" else MULTI_PRED_PROXIES

    for tick in target_ticks:
        for exp_name, proxy_setting in proxy_dict.items():
            if mode == "single":
                post_p = proxy_setting
                comment_p = "ML2_proxy1_probability"
            else:
                post_p, comment_p = proxy_setting

            try:
                sampler = ProxyStratifiedSampler(
                    csv_path=filepath,
                    is_multi_predicate=True,
                    post_proxy=post_p,
                    comment_proxy=comment_p,
                    post_oracle="ML1_oracle2_probability",
                    comment_oracle="ML2_oracle2_probability",
                    T_true=T_true,
                    total_budget_frac=tick
                )
            except Exception:
                continue

            if sampler.posts.empty:
                continue

            total_instances = len(sampler.posts)
            budget_n = max(1, int(math.floor(tick * total_instances)))

            sampler.total_budget_frac = tick
            sampler.c_stage = 0.0
            sampler.K = min(5, budget_n)

            for r in range(run_times):
                try:
                    res = sampler.run_possa()
                    oracle_cost = res.get("n_post", 0) + res.get("n_comment", 0)
                    
                    record = {
                        "query_basename": query_basename,
                        "run_id": r + 1,
                        "budget_frac": tick,
                        "budget_n": budget_n,
                        "T_true": T_true,
                        "T_hat": res["T_hat"],
                        "Qerror": res["Qerror"],
                        "n_post": res.get("n_post", 0),
                        "n_comment": res.get("n_comment", 0),
                        "oracle_cost": oracle_cost,
                        "method": exp_name
                    }
                    file_records.append(record)
                except Exception:
                    pass

    return file_records

def parse_ticks(ticks_str: str):
    return [float(x.strip()) for x in ticks_str.split(",") if x.strip() != ""]

# ==============================================================================
# 4. 主调度函数 (基于动态路径)
# ==============================================================================
def run_proxy_quality_ablation(
    base_dir: str,
    dataset: str,
    mode: str,
    run_times: int,
    max_workers: int,
    target_ticks: list,
    out_csv_name: str
):
    # 动态拼接数据集路径 (如: /home/.../PROXY/datasets/parler-E)
    dataset_path = os.path.join(base_dir, "datasets", dataset)
    aggregated_dir = os.path.join(dataset_path, "results", "aggregated_results_count")
    
    # 兼容没有 _count 后缀的目录命名
    if not os.path.exists(aggregated_dir):
        aggregated_dir = os.path.join(dataset_path, "results", "aggregated_results")

    # 自动探测 Ground Truth JSON 路径
    res_dir = os.path.join(dataset_path, "results")
    t_true_candidates = [
        os.path.join(res_dir, "T_true_ML1_oracle2_probability_ML2_oracle2_probability_count.json"),
        os.path.join(res_dir, "T_true_ML1_oracle2_probability_ML2_oracle2_probability.json"),
        os.path.join(res_dir, "T_true_ML3_oracle2_probability_ML2_oracle1_probability_count.json")
    ]
    t_true_path = next((p for p in t_true_candidates if os.path.exists(p)), None)

    if not t_true_path:
        print(f"[Error] 找不到真值 JSON 文件于: {res_dir}")
        return

    output_dir = os.path.join(dataset_path, "results", "efficiency")
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, out_csv_name)

    print(f"\n{'='*60}")
    print(f"🚀 开始 RQ3 消融实验 | 数据集: {dataset} | 模式: {mode.upper()}")
    print(f"   - 根目录: {base_dir}")
    print(f"   - 输出文件: {output_csv}")
    print(f"{'='*60}")

    with open(t_true_path, 'r') as f:
        all_t_true = json.load(f)

    if not os.path.exists(aggregated_dir):
        print(f"[Error] 找不到核心实例目录: {aggregated_dir}")
        return

    agg_files = sorted([f for f in os.listdir(aggregated_dir) if f.endswith(".csv")])

    headers = ["query_basename", "run_id", "budget_frac", "budget_n", "T_true", "T_hat", "Qerror", "n_post", "n_comment", "oracle_cost", "method"]
    pd.DataFrame(columns=headers).to_csv(output_csv, index=False)

    if max_workers is None:
        max_workers = max(1, os.cpu_count() - 2)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _process_ablation_single_file,
                agg_file, aggregated_dir, all_t_true, target_ticks, run_times, mode
            ) for agg_file in agg_files
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating {dataset}"):
            try:
                result_records = future.result()
                if result_records:
                    df_chunk = pd.DataFrame(result_records)
                    df_chunk.to_csv(output_csv, mode='a', header=False, index=False)
            except Exception as e:
                print(f"[Error] {e}")

    print(f"✅ [{dataset}] RQ3 消融评估完成！结果已即时保存至: {output_csv}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ3 Proxy Quality Ablation Runner")
    parser.add_argument("--base_dir", type=str, default=DEFAULT_PROJECT_ROOT, help="Project base path")
    parser.add_argument("--dataset", type=str, default="parler", help="Dataset name (parler, parler-E, etc.)")
    parser.add_argument("--mode", type=str, choices=["single", "multi"], default="single", help="Single/Multi predicate mode")
    parser.add_argument("--run_times", type=int, default=5, help="Number of runs per budget tick")
    parser.add_argument("--max_workers", type=int, default=16, help="Process pool workers")
    parser.add_argument("--target_ticks", type=str, default="0.1", help="Budget fractions")
    parser.add_argument("--out_csv", type=str, default="proxy_quality_ablation_count.csv", help="Output CSV filename")

    args = parser.parse_args()
    ticks = parse_ticks(args.target_ticks)

    run_proxy_quality_ablation(
        base_dir=args.base_dir,
        dataset=args.dataset,
        mode=args.mode,
        run_times=args.run_times,
        max_workers=args.max_workers,
        target_ticks=ticks,
        out_csv_name=args.out_csv
    )