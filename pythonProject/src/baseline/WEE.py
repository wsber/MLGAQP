import os
import json
import sys
import argparse
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# 动态定位项目根目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 假设当前脚本在 pythonProject/src/baseline/，向上回退三级回到 PROXY
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from pythonProject.src.algorithms.proxy_sample import ProxyStratifiedSampler
from pythonProject.src.algorithms.compute_truth import GroundTruthManager

def _worker_process_single_file(agg_file: str, aggregated_dir: str, all_T_true_results: dict, config: dict):
    """
    [子进程工作函数] 处理单个文件，运行 WO 基线并返回指标。
    """
    if agg_file.startswith("aggregated_list_"):
        base = agg_file.replace("aggregated_list_", "")
    elif agg_file.startswith("aggregated_wide_"):
        base = agg_file.replace("aggregated_wide_", "")
    else:
        base = agg_file
    query_basename = base.replace(".csv", "") + ".graph"

    T_true = all_T_true_results.get(query_basename)
    if T_true is None:
        return None

    filepath = os.path.join(aggregated_dir, agg_file)

    try:
        sampler = ProxyStratifiedSampler(
            csv_path=filepath,
            is_multi_predicate=True,
            post_proxy=config["post_proxy"],
            comment_proxy=config["comment_proxy"],
            post_oracle=config["post_oracle"],
            comment_oracle=config["comment_oracle"],
            T_true=T_true,
            total_budget_frac=1.0  # WO 是全局统计，不需要被 budget 截断
        )
        
        if sampler.posts.empty:
            return None

        # 运行基线 (Graph Only 意味着只经过 Fastest 图信息直接估算，对应 WO)
        res = sampler.run_baseline_graph_only()
        
        return {
            "query_name": query_basename,
            "T_true": float(T_true),
            "T_hat": float(res.get("T_hat", 0.0)),
            "abs_error_rate": float(res["Qerror"]), 
            "n_post": res.get("n_post", 0),
            "n_comment": res.get("n_comment", 0),
            "Total_Cost": res.get("n_post", 0) + res.get("n_comment", 0)
        }
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate WO Baseline for COUNT and SUM")
    parser.add_argument("--parent_dataset", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--table1", type=str, required=True)
    parser.add_argument("--table2", type=str, required=True)
    parser.add_argument("--agg_mode", type=str, choices=["count", "sum"], required=True)
    parser.add_argument("--t1_proxy", type=str, required=True)
    parser.add_argument("--t2_proxy", type=str, required=True)
    parser.add_argument("--t1_oracle", type=str, required=True)
    parser.add_argument("--t2_oracle", type=str, required=True)
    parser.add_argument("--workers", type=int, default=16)

    args = parser.parse_args()

    print(f"\n{'='*15} Evaluation: WO Baseline [{args.dataset_name} | {args.agg_mode.upper()}] {'='*15}")
    
    base_path = os.path.join(PROJECT_ROOT, "datasets", args.parent_dataset)
    results_dir = os.path.join(base_path, "results")
    
    # 动态匹配 _count 和 _sum 后缀隔离的聚合结果文件夹
    aggregated_dir = os.path.join(results_dir, f"aggregated_results_{args.agg_mode}")
    if not os.path.exists(aggregated_dir):
        aggregated_dir = os.path.join(results_dir, "aggregated_results")
        
    if not os.path.exists(aggregated_dir):
        print(f"[Error] 聚合结果目录不存在: {aggregated_dir}")
        return

    # 动态加载正确的 T_true 真值 JSON
    gt_candidates = [
        os.path.join(results_dir, f"T_true_{args.t1_oracle}_{args.t2_oracle}_{args.agg_mode}.json"),
        os.path.join(results_dir, f"T_true_{args.t1_oracle}_{args.t2_oracle}.json")
    ]
    t_true_path = next((p for p in gt_candidates if os.path.exists(p)), None)
    
    if not t_true_path:
        print(f"[Error] 未找到对应的 Ground Truth JSON: {results_dir}")
        return

    print(f"[*] 读入 T_true 数据集: {t_true_path}")
    with open(t_true_path, 'r') as f:
        all_T_true_results = json.load(f)

    agg_files = sorted([f for f in os.listdir(aggregated_dir) if f.endswith(".csv")])
    print(f"[*] 共发现 {len(agg_files)} 个核心实例文件。")
    
    config = {
        "post_proxy": args.t1_proxy, "comment_proxy": args.t2_proxy,
        "post_oracle": args.t1_oracle, "comment_oracle": args.t2_oracle
    }

    metrics = {"Qerror": [], "Total_Cost": []}
    detailed_results_list = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_worker_process_single_file, f, aggregated_dir, all_T_true_results, config) for f in agg_files]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating WO ({args.agg_mode})"):
            result = future.result()
            if result:
                detailed_results_list.append(result)
                metrics["Qerror"].append(result["abs_error_rate"])
                metrics["Total_Cost"].append(result["Total_Cost"])

    if not metrics["Qerror"]: 
        print("❌ [Warn] 无有效结果，评估失败。")
        return

    detailed_results_list.sort(key=lambda x: x["abs_error_rate"], reverse=True)

    output_dir = os.path.join(results_dir, "efficiency")
    os.makedirs(output_dir, exist_ok=True)
    
    suffix = f"_{args.agg_mode}"
    output_json = os.path.join(output_dir, f"WO_details_{args.dataset_name}{suffix}.json")
    output_csv = os.path.join(output_dir, f"WO_summary{suffix}.csv")
    
    with open(output_json, 'w') as f:
        json.dump(detailed_results_list, f, indent=4)
        
    avg_qerror = np.mean(metrics["Qerror"])
    df_summary = pd.DataFrame([{
        "dataset_name": args.dataset_name, 
        "method": "WO", 
        "agg_mode": args.agg_mode,
        "num_queries": len(metrics["Qerror"]),
        "mean_are": avg_qerror, 
        "avg_total_cost": np.mean(metrics["Total_Cost"])
    }])
    df_summary.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)

    print(f"\n✅ [完成] 统计汇总已更新至: {output_csv}")
    print(f"📊 Weight Estimation Error (WEE) Floor ({args.agg_mode.upper()}): {avg_qerror:.4f} ({avg_qerror * 100:.2f}%)")
    print("=" * 65)

if __name__ == "__main__":
    main()