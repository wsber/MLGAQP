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

from pythonProject.src.Structure_first.proxy_sample import ProxyStratifiedSampler


# 定义多谓词场景下 不同质量的 (Post Proxy, Comment Proxy) 对
# 根据你提及的 pa1 > pa2 > pa3 > pa4 以及 pb1 > pb2 > pb3 > pb4 的 F1 得分设置

PROXY_QUALITIES = {
    # (pa1, pb1) - 最好的组合
    "POSSA_Q1_Best": ("ML1_proxy4b_probability", "ML2_proxy1_probability"),  
    # (pa2, pb2)
    "POSSA_Q2_High": ("ML1_proxy6b_probability", "ML2_proxy4_probability"),  
    # (pa3, pb3)
    "POSSA_Q3_Med":  ("ML1_proxy2b_probability", "ML2_proxy2_probability"),  
    # (pa4, pb4) - 最差的组合
    "POSSA_Q4_Low":  ("ML1_proxy1b_probability", "ML2_proxy2d2_probability"), 
}

def _process_ablation_single_file(
    agg_file: str,
    base_path: str,
    aggregated_dir: str,
    all_t_true: dict,
    target_ticks: list,
    run_times: int
):
    
    if agg_file.startswith("aggregated_list_"):
        base = agg_file.replace("aggregated_list_", "")
    elif agg_file.startswith("aggregated_wide_"):
        base = agg_file.replace("aggregated_wide_", "")
    else:
        base = agg_file
    query_basename = base.replace(".csv", "") + ".graph"

    T_true = all_t_true.get(query_basename)
    if T_true is None or T_true == 0:
        return []

    filepath = os.path.join(aggregated_dir, agg_file)
    file_records = []

    
    for tick in target_ticks:
        
        
        for exp_name, (post_proxy_col, comment_proxy_col) in PROXY_QUALITIES.items():
            
            try:
                sampler = ProxyStratifiedSampler(
                    csv_path=filepath,
                    is_multi_predicate=True,
                    post_proxy=post_proxy_col,          
                    comment_proxy=comment_proxy_col,    
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
            budget_n = int(math.floor(tick * total_instances))
            
            
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

def _parse_ticks(ticks_str: str):
    if ticks_str is None or str(ticks_str).strip() == "":
        return None
    try:
        ticks = [float(x.strip()) for x in ticks_str.split(",") if x.strip() != ""]
    except ValueError:
        raise ValueError(f"Invalid ticks string: {ticks_str}")
    if not ticks:
        raise ValueError("ticks is empty")
    return ticks

def run_proxy_quality_ablation(
    parent_dataset = "parler_data",
    dataset_name: str = "dataset_test",
    run_times: int = 5,
    max_workers: int = None,
    target_ticks: list = None
):
    
    if target_ticks is None:
        TARGET_TICKS = [0.01, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 0.4, 0.5]
    else:
        TARGET_TICKS = target_ticks
    
    base_path = f"/home/wangshuo/resource/datasets/{parent_dataset}/{dataset_name}"
    aggregated_dir = os.path.join(base_path, "results", "aggregated_results")
    
    t_true_path = os.path.join(base_path, "results", "T_true_ML1_oracle2_probability_ML2_oracle2_probability_count.json")
    if not os.path.exists(t_true_path):
        t_true_path = os.path.join(base_path, "results", "T_true_ML1_oracle2_probability_ML2_oracle2_probability.json")
    
    output_dir = os.path.join(base_path, "results", "efficiency")
    os.makedirs(output_dir, exist_ok=True)
    
    
    output_csv = os.path.join(output_dir, "proxy_quality_multipred_ablation_count.csv")

    print(f"\n{'='*15} 开始多谓词 Proxy (A,B) 质量消融实验 {'='*15}")
    print(f"[*] 测试的 Proxy 组合列表 (pa, pb):")
    for name, cols in PROXY_QUALITIES.items():
        print(f"    - {name} : Post={cols[0]}, Comment={cols[1]}")
    print(f"[*] 输出文件: {output_csv}")
    print(f"{'='*50}\n")
    
    if not os.path.exists(t_true_path):
        print(f"[Error] T_true not found: {t_true_path}")
        return
    with open(t_true_path, 'r') as f:
        all_t_true = json.load(f)

    if not os.path.exists(aggregated_dir):
        print(f"[Error] Aggregated dir not found: {aggregated_dir}")
        return
    agg_files = sorted([f for f in os.listdir(aggregated_dir) if f.endswith(".csv")])

    headers = ["query_basename", "run_id", "budget_frac", "budget_n", "T_true", "T_hat", "Qerror", "n_post", "n_comment", "oracle_cost", "method"]
    pd.DataFrame(columns=headers).to_csv(output_csv, index=False)

    if max_workers is None:
        max_workers = max(1, os.cpu_count() - 2)

    print(f"[*] 进程数 (Workers): {max_workers}")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
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
    # parent_dataset = "amzon_data"
    # dataset_name = "amazon_extend"
    parent_dataset = "parler_data"
    dataset_name = "dataset_test"
    
    parser = argparse.ArgumentParser(description="Multi-predicate Ablation study for different proxy qualities (pa, pb)")
    parser.add_argument("--dataset_name", type=str, default=dataset_name)
    parser.add_argument("--run_times", type=int, default=5)
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument(
        "--target_ticks",
        type=str,
        # default="0.01,0.05,0.075,0.1,0.125,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        default="0.01,0.05,0.1",
        help="Comma-separated ticks, e.g. 0.05,0.1,0.2"
    )

    args = parser.parse_args()
    ticks = _parse_ticks(args.target_ticks)

    run_proxy_quality_ablation(
        parent_dataset=parent_dataset,
        dataset_name=args.dataset_name,
        run_times=args.run_times,
        max_workers=args.max_workers,
        target_ticks=ticks
    )