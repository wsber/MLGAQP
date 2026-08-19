"""
# 1. 运行 Parler (0) 默认只跑 8_POSSA (RQ1~RQ3)
python Proxy_Guided_Stratified_Importance_Sampling_Runner.py -d 0 --agg_mode count --target_ticks "0.01,0.05,0.1"

# 2. 运行 RQ4 消融实验 (跑全部 5 种方法)
python Proxy_Guided_Stratified_Importance_Sampling_Runner.py -d 0 --agg_mode count --target_ticks "0.1" --methods all --out_csv_name allocation_strategy_comparison_ablation_count.csv
"""

import os
import json
import math
import sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# 自动定位项目根目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))

if DEFAULT_PROJECT_ROOT not in sys.path:
    sys.path.append(DEFAULT_PROJECT_ROOT)
from pythonProject.src.algorithms.proxy_sample import ProxyStratifiedSampler

# ======================= 数据集与模型配置映射 =======================
DATASET_CONFIGS = {
    0: {
        "desc": "Parler (单谓词基准)",
        "parent_dataset": "parler",
        "dataset_name": "dataset_three",
        "model_config": {
            "POST_PROXY": "ML1_proxy4b_probability",
            "COMMENT_PROXY": "ML2_proxy1_probability",
            "POST_ORACLE": "ML1_oracle2_probability",
            "COMMENT_ORACLE": "ML2_oracle2_probability"
        }
    },
    1: {
        "desc": "Parler1 / Parler-E (多谓词扩展)",
        "parent_dataset": "parler-E",
        "dataset_name": "dataset_test",
        "model_config": {
            "POST_PROXY": "ML1_proxy4b_probability",
            "COMMENT_PROXY": "ML2_proxy1_probability",
            "POST_ORACLE": "ML1_oracle2_probability",
            "COMMENT_ORACLE": "ML2_oracle2_probability"
        }
    },
    2: {
        "desc": "Amazon (多模态图)",
        "parent_dataset": "amazon",
        "dataset_name": "amazon_extend",
        "model_config": {
            "POST_PROXY": "ML3_proxy2_probability",      
            "COMMENT_PROXY": "ML2_proxy2_probability",   
            "POST_ORACLE": "ML3_oracle2_probability",
            "COMMENT_ORACLE": "ML2_oracle1_probability"
        }
    }
}

def _process_comparison_single_file(
    agg_file: str,
    base_path: str,
    aggregated_dir: str,
    all_t_true: dict,
    target_ticks: list,
    run_times: int,
    config: dict,
    selected_methods: list 
):
    """子进程工作函数"""
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
    
    # =========================================================
    # 【已修复】：严格按照 proxy_sample.py 的真实签名传参
    # =========================================================
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
    except Exception as e:
        print(f"\n[初始化错误] {query_basename}: {e}")
        return []

    if sampler.posts.empty:
        return []

    total_instances = len(sampler.posts)
    file_records = []

    # 所有可用方法池
    ALL_AVAILABLE_METHODS = {
        "UN": sampler.run_baseline_uniform,
        "PO": sampler.run_baseline_proxy,
        "WO": sampler.run_baseline_weight_only,
        "MAB": sampler.run_mab_sampling,
        "8_POSSA": sampler.run_possa,
    }

    methods_map = {m: ALL_AVAILABLE_METHODS[m] for m in selected_methods if m in ALL_AVAILABLE_METHODS}
    methods_requiring_pilot = {"1_Proxy_Imp_Pilot", "2_ProxyE_Imp_Pilot"}
    
    for tick in target_ticks:
        budget_n = max(1, int(math.floor(tick * total_instances)))
        sampler.total_budget_frac = tick

        for method_name, run_func in methods_map.items():
            if method_name in methods_requiring_pilot:
                sampler.c_stage = 0.2
            else:
                sampler.c_stage = 0.0
            sampler.K = min(5, budget_n)  

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
                except Exception as e:
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


def run_allocation_strategy_comparison(
    parent_dataset: str,
    dataset_name: str,
    config: dict,
    run_times: int = 5,
    max_workers: int = None,
    target_ticks: list = None,
    agg_mode_init: str = "count",
    base_dir: str = DEFAULT_PROJECT_ROOT,
    selected_methods: list = ["8_POSSA"],  # 默认仅 8_POSSA，完美契合 RQ1~RQ3
    out_csv_name: str = None               
):
    if target_ticks is None:
        TARGET_TICKS = [0.01, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2]
    else:
        TARGET_TICKS = target_ticks

    base_path = os.path.join(base_dir, "datasets", parent_dataset)
    agg_mode = agg_mode_init.lower()
    
    aggregated_dir = os.path.join(base_path, "results", f"aggregated_results_{agg_mode}")
    if not os.path.exists(aggregated_dir):
        fallback_agg_dir = os.path.join(base_path, "results", "aggregated_results")
        if os.path.exists(fallback_agg_dir):
            aggregated_dir = fallback_agg_dir
        else:
            print(f"[错误] 未找到聚合结果目录: {aggregated_dir}")
            return

    safe_post = config["POST_ORACLE"].replace("/", "_")
    safe_comment = config["COMMENT_ORACLE"].replace("/", "_")
    
    t_true_path = os.path.join(base_path, "results", f"T_true_{safe_post}_{safe_comment}_{agg_mode}.json")
    output_dir = os.path.join(base_path, "results", "efficiency")
    os.makedirs(output_dir, exist_ok=True)
    
    if out_csv_name:
        output_csv = os.path.join(output_dir, out_csv_name)
    else:
        output_csv = os.path.join(output_dir, f"allocation_strategy_comparison_{agg_mode}.csv")
    
    print(f"\n{'='*10} 启动采样评估 [{parent_dataset} | {agg_mode.upper()}] {'='*10}")
    print(f"[*] 执行的方法集合: {selected_methods}")
    print(f"[*] 输出文件路径: {output_csv}")
    
    if not os.path.exists(t_true_path):
        fallback_path = os.path.join(base_path, "results", f"T_true_{safe_post}_{safe_comment}.json")
        if not os.path.exists(fallback_path):
            print(f"[严重错误] 未能找到 T_true 文件: {t_true_path}")
            return
        t_true_path = fallback_path
        
    with open(t_true_path, 'r') as f:
        all_t_true = json.load(f)

    agg_files = sorted([f for f in os.listdir(aggregated_dir) if f.endswith(".csv")])
    if not agg_files:
        print(f"[警告] 目录 {aggregated_dir} 下未找到任何 CSV 文件。")
        return

    headers = ["query_basename", "run_id", "budget_frac", "budget_n", "T_true", "T_hat", "Qerror", "n_post", "n_comment", "oracle_cost", "method"]
    pd.DataFrame(columns=headers).to_csv(output_csv, index=False)

    if max_workers is None:
        max_workers = max(1, os.cpu_count() - 2)

    all_results_buffer = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _process_comparison_single_file,
                agg_file, base_path, aggregated_dir, all_t_true, 
                TARGET_TICKS, run_times, config, selected_methods
            ) for agg_file in agg_files
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Sampling ({parent_dataset}-{agg_mode})"):
            try:
                result_records = future.result()
                if result_records:
                    all_results_buffer.extend(result_records)
                    if len(all_results_buffer) >= 500:
                        pd.DataFrame(all_results_buffer).to_csv(output_csv, mode='a', header=False, index=False)
                        all_results_buffer.clear()
            except Exception as e:
                print(f"Error: {e}")

        if all_results_buffer:
            pd.DataFrame(all_results_buffer).to_csv(output_csv, mode='a', header=False, index=False)
            all_results_buffer.clear()

    print(f"[Done] 评估完成，已保存至: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run POSS allocation strategy comparison")
    
    parser.add_argument("--dataset_id", "-d", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--agg_mode", type=str, default="count", choices=["count", "sum"])
    parser.add_argument("--run_times", type=int, default=5)
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--target_ticks", type=str, default="0.1")
    parser.add_argument("--base_dir", type=str, default=DEFAULT_PROJECT_ROOT)
    
    # 默认只跑 8_POSSA（完全不影响 RQ1~RQ3）
    parser.add_argument(
        "--methods",
        type=str,
        default="8_POSSA",
        help="方法列表，如 '8_POSSA', 'all', 或 'UN,PO,WO,MAB,8_POSSA'"
    )
    # 自定义输出文件名
    parser.add_argument(
        "--out_csv_name",
        type=str,
        default=None,
        help="自定义输出 CSV 文件名"
    )

    args = parser.parse_args()
    ticks = _parse_ticks(args.target_ticks)

    if args.methods.lower() == "all":
        methods_to_run = ["UN", "PO", "WO", "MAB", "8_POSSA"]
    else:
        methods_to_run = [m.strip() for m in args.methods.split(",") if m.strip()]

    selected_cfg = DATASET_CONFIGS[args.dataset_id]

    run_allocation_strategy_comparison(
        parent_dataset=selected_cfg["parent_dataset"],
        dataset_name=selected_cfg["dataset_name"],
        config=selected_cfg["model_config"],
        run_times=args.run_times,
        max_workers=args.max_workers,
        target_ticks=ticks,
        agg_mode_init=args.agg_mode,
        base_dir=args.base_dir,
        selected_methods=methods_to_run,
        out_csv_name=args.out_csv_name
    )