import os
import json
import math
import sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse

project_root = "/home/wangshuo/projects/Neo4j_Exp"
if project_root not in sys.path:
    sys.path.append(project_root)

from pythonProject.src.algorithms.proxy_sample import ProxyStratifiedSampler

# ==========================================
# [核心配置]: 定义 4 种不同质量的 Post Proxy
# ==========================================
# 请将对应的值替换为你 CSV 中真实的列名
PROXY_QUALITIES = {
    "POSSA_Q0_Worst": "worst_proxy_probability",    #最差质量代理 (完全随机)
    "POSSA_Q1_Low": "ML1_proxy1b_probability",      # 替换为最低质量代理列
    "POSSA_Q2_Med": "ML1_proxy2b_probability",      # 替换为中等质量代理列
    "POSSA_Q3_High": "ML1_proxy6b_probability",     # 替换为高质量代理列
    "POSSA_Q4_Best": "ML1_proxy4b_probability"     # 你原配置中最高质量的列
}

def _process_ablation_single_file(
    agg_file: str,
    base_path: str,
    aggregated_dir: str,
    all_t_true: dict,
    target_ticks: list,
    run_times: int
):
    """子进程工作函数：对比同一方法在不同 Proxy 下的表现"""
    
    # 1. 解析文件名与 Ground Truth
    if agg_file.startswith("aggregated_list_"):
        base = agg_file.replace("aggregated_list_", "")
    elif agg_file.startswith("aggregated_wide_"):
        base = agg_file.replace("aggregated_wide_", "")
    else:
        base = agg_file
    query_basename = base.replace(".csv", "") + ".graph"

    T_true = all_t_true.get(query_basename)
    if T_true is None:
        return[]

    filepath = os.path.join(aggregated_dir, agg_file)
    file_records =[]

    # 2. 循环采样率
    for tick in target_ticks:
        
        # 3. 循环 4 种不同质量的 Proxy
        for exp_name, post_proxy_col in PROXY_QUALITIES.items():
            
            # 因为不同的实验使用了不同的 proxy 列，必须在此处重新实例化 Sampler
            try:
                sampler = ProxyStratifiedSampler(
                    csv_path=filepath,
                    is_multi_predicate=True,
                    post_proxy=post_proxy_col,                  # <--- 动态传入不同质量的 proxy
                    comment_proxy="ML2_proxy1_probability",     # COMMENT 保持不变
                    post_oracle="ML1_oracle2_probability",
                    comment_oracle="ML2_oracle2_probability",
                    T_true=T_true,
                    total_budget_frac=tick 
                )
            except Exception:
                # 可能是对应的 proxy 列在某个 CSV 中不存在
                continue

            if sampler.posts.empty:
                continue

            total_instances = len(sampler.posts)
            budget_n = int(math.floor(tick * total_instances))
            
            # 配置 Sampler 参数
            sampler.total_budget_frac = tick
            sampler.c_stage = 0.0 # run_possa 不需要 pilot 阶段
            sampler.K = min(5, budget_n)  # 动态调整层数
            
            # 4. 重复运行 run_possa
            for r in range(run_times):
                try:
                    # 我们现在只调用 run_possa
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
                        "method": exp_name  # 将方法名标记为 "POSSA_Qx" 以区分不同的 Proxy
                    }
                    file_records.append(record)
                except Exception:
                    pass
                    
    return file_records

def _parse_ticks(ticks_str: str):
    if ticks_str is None or str(ticks_str).strip() == "":
        return None
    try:
        ticks =[float(x.strip()) for x in ticks_str.split(",") if x.strip() != ""]
    except ValueError:
        raise ValueError(f"Invalid ticks string: {ticks_str}")
    if not ticks:
        raise ValueError("ticks is empty")
    return ticks

def run_proxy_quality_ablation(
    dataset_name: str = "dataset_test",
    run_times: int = 5,
    max_workers: int = None,
    target_ticks: list = None
):
    """
    运行 Proxy 质量消融实验
    """
    if target_ticks is None:
        TARGET_TICKS =[0.01, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 0.4, 0.5]
    else:
        TARGET_TICKS = target_ticks
    
    base_path = f"/home/wangshuo/resource/datasets/parler_data/{dataset_name}"
    aggregated_dir = os.path.join(base_path, "results", "aggregated_results")
    t_true_path = os.path.join(base_path, "results", "T_true_ML1_oracle2_probability_ML2_oracle2_probability.json")
    
    output_dir = os.path.join(base_path, "results", "efficiency")
    os.makedirs(output_dir, exist_ok=True)
    
    # 修改输出文件名，彰显这是一个 Proxy 消融实验
    output_csv = os.path.join(output_dir, "proxy_quality_ablation_count.csv")

    print(f"\n{'='*15} 开始 Proxy 质量消融实验 {'='*15}")
    print(f"[*] 测试的 Proxy 列表:")
    for name, col in PROXY_QUALITIES.items():
        print(f"    - {name} : {col}")
    print(f"[*] 输出文件: {output_csv}")
    print(f"{'='*50}\n")
    
    # Load T_true
    if not os.path.exists(t_true_path):
        print(f"[Error] T_true not found: {t_true_path}")
        return
    with open(t_true_path, 'r') as f:
        all_t_true = json.load(f)

    # Prepare files
    if not os.path.exists(aggregated_dir):
        print(f"[Error] Aggregated dir not found: {aggregated_dir}")
        return
    agg_files = sorted([f for f in os.listdir(aggregated_dir) if f.endswith(".csv")])

    # Init CSV
    headers =["query_basename", "run_id", "budget_frac", "budget_n", "T_true", "T_hat", "Qerror", "n_post", "n_comment", "oracle_cost", "method"]
    pd.DataFrame(columns=headers).to_csv(output_csv, index=False)

    # Parallel Execution
    if max_workers is None:
        max_workers = max(1, os.cpu_count() - 2)

    print(f"[*] 进程数 (Workers): {max_workers}")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures =[]
        for agg_file in agg_files:
            futures.append(
                executor.submit(
                    _process_ablation_single_file,
                    agg_file, base_path, aggregated_dir, all_t_true, 
                    TARGET_TICKS, run_times
                )
            )
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Queries"):
            try:
                result_records = future.result()
                if result_records:
                    df_chunk = pd.DataFrame(result_records)
                    df_chunk.to_csv(output_csv, mode='a', header=False, index=False)
            except Exception as e:
                print(f"Error: {e}")

    print(f"\n✅ [Done] 结果已保存至: {output_csv}")

if __name__ == "__main__":
    dataset_name = "dataset_three"
    parser = argparse.ArgumentParser(description="Ablation study for different proxy qualities using POSSA")
    parser.add_argument("--dataset_name", type=str, default=dataset_name)
    parser.add_argument("--run_times", type=int, default=5)
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument(
        "--target_ticks",
        type=str,
        default="0.01,0.05,0.075,0.1,0.125,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",  # 默认跑一组常见采样率用于绘图
        help="Comma-separated ticks, e.g. 0.05,0.1,0.2"
    )

    args = parser.parse_args()
    ticks = _parse_ticks(args.target_ticks)

    run_proxy_quality_ablation(
        dataset_name=args.dataset_name,
        run_times=args.run_times,
        max_workers=args.max_workers,
        target_ticks=ticks
    )