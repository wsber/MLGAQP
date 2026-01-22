import os
import json
import math
import sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
project_root = "/home/wangshuo/projects/Neo4j_Exp"
if project_root not in sys.path:
    sys.path.append(project_root)
from pythonProject.src.Structure_first.proxy_sample import ProxyStratifiedSampler

# 请确保 ProxyStratifiedSampler 已经被更新并包含上面的新方法

def _process_comparison_single_file(
    agg_file: str,
    base_path: str,
    aggregated_dir: str,
    all_t_true: dict,
    target_ticks: list,
    run_times: int,
    config: dict
):
    """子进程工作函数"""
    # 1. 准备基础信息
    if agg_file.startswith("aggregated_list_"):
        base = agg_file.replace("aggregated_list_", "")
    elif agg_file.startswith("aggregated_wide_"):
        base = agg_file.replace("aggregated_wide_", "")
    else:
        base = agg_file
    query_basename = base.replace(".csv", "") + ".graph"

    T_true = all_t_true.get(query_basename)
    if T_true is None:
        return []

    filepath = os.path.join(aggregated_dir, agg_file)
    
    # 2. 初始化 Sampler
    try:
        sampler = ProxyStratifiedSampler(
            csv_path=filepath,
            is_multi_predicate=True,
            post_proxy=config["POST_PROXY"],
            comment_proxy=config["COMMENT_PROXY"],
            post_oracle=config["POST_ORACLE"],
            comment_oracle=config["COMMENT_ORACLE"],
            T_true=T_true,
            total_budget_frac=1.0 
        )
    except Exception:
        return []

    if sampler.posts.empty:
        return []

    total_instances = len(sampler.posts)
    file_records = []

    # === 定义 4 种对比方法 ===
    # 1. run_proxy_importance: P分层 + Pilot分配
    # 2. run_proxyE_importance: P*A分层 + Pilot分配
    # 3. run_proxyE_alloc_root_wp: P*A分层 + 启发式分配1 (sum w*sqrt(p))
    # 4. run_proxyE_alloc_w_root_pbar: P*A分层 + 启发式分配2 (sum w * sqrt(mean p))
    
    methods_map = {
        "1_Proxy_Imp_Pilot": sampler.run_proxy_importance,
        "2_ProxyE_Imp_Pilot": sampler.run_proxyE_importance,
        "3_ProxyE_Imp_RootWP": sampler.run_proxyE_alloc_root_wp,
        "4_ProxyE_Imp_WRootMeanP": sampler.run_proxyE_alloc_w_root_pbar
    }

    # 3. 循环采样率
    for tick in target_ticks:
        budget_n = int(math.floor(tick * total_instances))
        sampler.total_budget_frac = tick

        # 4. 循环方法
        for method_name, run_func in methods_map.items():
            # 5. 重复运行
            for r in range(run_times):
                try:
                    res = run_func()
                    
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
                        "method": method_name
                    }
                    file_records.append(record)
                except Exception:
                    pass
                    
    return file_records

def run_allocation_strategy_comparison(
    dataset_name: str = "dataset_test",
    run_times: int = 5,
    max_workers: int = None
):
    """
    运行四种方法的对比实验
    """
    # 采样率配置
    # TARGET_TICKS = [0.05, 0.1, 0.2, 0.3, 0.4] 
    TARGET_TICKS = [0.01, 0.05, 0.1, 0.15,0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    base_path = f"/home/wangshuo/resource/datasets/parler_data/{dataset_name}"
    aggregated_dir = os.path.join(base_path, "results", "aggregated_results")
    t_true_path = os.path.join(base_path, "results", "T_true_ML1_oracle2_probability_ML2_oracle2_probability.json")
    
    output_dir = os.path.join(base_path, "results", "efficiency")
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "allocation_strategy_comparison.csv")
    
    config = {
        "POST_PROXY": "ML1_proxy4b_probability",
        "COMMENT_PROXY": "ML2_proxy1_probability",
        "POST_ORACLE": "ML1_oracle2_probability",
        "COMMENT_ORACLE": "ML2_oracle2_probability"
    }

    print(f"\n{'='*10} 开始分配策略对比实验 (4 Methods) {'='*10}")
    
    # Load T_true
    if not os.path.exists(t_true_path):
        print("T_true not found.")
        return
    with open(t_true_path, 'r') as f:
        all_t_true = json.load(f)

    # Prepare files
    if not os.path.exists(aggregated_dir):
        print("Aggregated dir not found.")
        return
    agg_files = sorted([f for f in os.listdir(aggregated_dir) if f.endswith(".csv")])

    # Init CSV
    headers = ["query_basename", "run_id", "budget_frac", "budget_n", "T_true", "T_hat", "Qerror", "n_post", "n_comment", "oracle_cost", "method"]
    pd.DataFrame(columns=headers).to_csv(output_csv, index=False)

    # Parallel Execution
    if max_workers is None:
        max_workers = max(1, os.cpu_count() - 2)

    print(f"Workers: {max_workers}")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for agg_file in agg_files:
            futures.append(
                executor.submit(
                    _process_comparison_single_file,
                    agg_file, base_path, aggregated_dir, all_t_true, 
                    TARGET_TICKS, run_times, config
                )
            )
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Comparing"):
            try:
                result_records = future.result()
                if result_records:
                    df_chunk = pd.DataFrame(result_records)
                    df_chunk.to_csv(output_csv, mode='a', header=False, index=False)
            except Exception as e:
                print(f"Error: {e}")

    print(f"\n[Done] 结果已保存至: {output_csv}")

if __name__ == "__main__":
    run_allocation_strategy_comparison(dataset_name = "dataset_three" ,run_times=5)