#  interpretation: 面这段代码就是多次调用C++ 程序 进行 对数据集, 聚合函数(agg,count) 进行论文中的投影采样和权重估计, 为后续分层重要性采样算法提供 输入数据 (即每个 Query Q 的 \hat\{Psi} 和 \hat{w}(\psi))。:
#  author: shuo wang
#  input: datasets_name, dataset_name, 
#  output: the \hat\{Psi} and \hat{w}(\psi) of each Query Q in workload (dataset_name).


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

    # === 定义对比方法 ===
    methods_map = {
        "UN": sampler.run_baseline_uniform,
        "PO": sampler.run_baseline_proxy,
        "WO": sampler.run_baseline_weight_only,
        "MAB": sampler.run_mab_sampling,
        "8_POSSA": sampler.run_possa,  # 综合方法 POSSA
    }
    methods_requiring_pilot = {"1_Proxy_Imp_Pilot", "2_ProxyE_Imp_Pilot"}
    
    # 3. 循环采样率
    for tick in target_ticks:
        budget_n = int(math.floor(tick * total_instances))
        sampler.total_budget_frac = tick

        # 4. 循环方法
        for method_name, run_func in methods_map.items():
            if method_name in methods_requiring_pilot:
                sampler.c_stage = 0.2
            else:
                sampler.c_stage = 0.0
            sampler.K = min(5, budget_n)  # 动态调整层数
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

def _parse_ticks(ticks_str: str):
    """将 '0.01,0.05,0.075' -> [0.01, 0.05, 0.075]"""
    if ticks_str is None or str(ticks_str).strip() == "":
        return None
    try:
        ticks = [float(x.strip()) for x in ticks_str.split(",") if x.strip() != ""]
    except ValueError:
        raise ValueError(f"Invalid ticks string: {ticks_str}")
    if not ticks:
        raise ValueError("ticks is empty")
    return ticks

def run_allocation_strategy_comparison(
    parent_dataset: str = "amazon_data",
    dataset_name: str = "amazon_extend",
    run_times: int = 5,
    max_workers: int = None,
    target_ticks: list = None,
    agg_mode_init: str = "sum"
):
    """运行四种方法的对比实验"""
    if target_ticks is None:
        TARGET_TICKS = [0.01,0.05,0.075,0.1,0.125,0.15,0.2]
    else:
        TARGET_TICKS = target_ticks
    
    # 【修改 1】动态支持不同父级数据目录
    base_path = f"/home/wangshuo/resource/datasets/{parent_dataset}/{dataset_name}"
    aggregated_dir = os.path.join(base_path, "results", "aggregated_results")
    
    # 【修改 2】配置模型名（分别对应你给出的amazon的列）
    # config = {
    #     "POST_PROXY": "ML3_proxy2_probability",      # 对应 table1 / Product
    #     "COMMENT_PROXY": "ML2_proxy2_probability",   # 对应 table2 / Review
    #     "POST_ORACLE": "ML3_oracle2_probability",
    #     "COMMENT_ORACLE": "ML2_oracle1_probability"
    # }
    config = {
        "POST_PROXY": "ML1_proxy4b_probability",      # 对应 table1 / post
        "COMMENT_PROXY": "ML2_proxy1_probability",   # 对应 table2 / comment
        "POST_ORACLE": "ML1_oracle2_probability",
        "COMMENT_ORACLE": "ML2_oracle2_probability"
    }
    agg_mode = agg_mode_init  
    # 【修改 3】动态拼出 JSON 的全名名称
    safe_post = config["POST_ORACLE"].replace("/", "_")
    safe_comment = config["COMMENT_ORACLE"].replace("/", "_")
    
    # 此处假设读取 COUNT 真实值
    # 如果你是要对比 sum 实验的话需要在这里加上 `sum_product_upvotes` 等字眼。
    t_true_path = os.path.join(base_path, "results", f"T_true_{safe_post}_{safe_comment}_{agg_mode}.json")
    print(f'********{t_true_path}')
    output_dir = os.path.join(base_path, "results", "efficiency")
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, f"allocation_strategy_comparison_ablation_{agg_mode}.csv")
    
    print(f"\n{'='*10} 开始分配策略对比实验 (POSSA) {'='*10}")
    
    if not os.path.exists(t_true_path):
        # 尝试去掉 _count 查找
        fallback_path = os.path.join(base_path, "results", f"T_true_{safe_post}_{safe_comment}.json")
        if not os.path.exists(fallback_path):
            print(f"[严重错误] 未能找到 T_true 文件。尝试了以下路径:\n  - {t_true_path}\n  - {fallback_path}")
            return
        t_true_path = fallback_path
        
    with open(t_true_path, 'r') as f:
        all_t_true = json.load(f)

    if not os.path.exists(aggregated_dir):
        print(f"Aggregated dir not found: {aggregated_dir}")
        return
    agg_files = sorted([f for f in os.listdir(aggregated_dir) if f.endswith(".csv")])

    headers = ["query_basename", "run_id", "budget_frac", "budget_n", "T_true", "T_hat", "Qerror", "n_post", "n_comment", "oracle_cost", "method"]
    pd.DataFrame(columns=headers).to_csv(output_csv, index=False)

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
    parser = argparse.ArgumentParser(description="Run POSS allocation strategy comparison")
    parser.add_argument("--parent_dataset", type=str, default="parler_data")
    parser.add_argument("--dataset_name", type=str, default="dataset_three")
    parser.add_argument("--run_times", type=int, default=5)
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument(
        "--target_ticks",
        type=str,
        default="0.1",
        # default="0.01,0.05,0.075,0.1,0.125,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        help="Comma-separated ticks, e.g. 0.01,0.05,0.1"
    )

    args = parser.parse_args()
    ticks = _parse_ticks(args.target_ticks)

    run_allocation_strategy_comparison(
        parent_dataset=args.parent_dataset,
        dataset_name=args.dataset_name,
        run_times=args.run_times,
        max_workers=args.max_workers,
        target_ticks=ticks,
        agg_mode_init="count"
    )