import os
import json
import sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

project_root = "/home/wangshuo/projects/Neo4j_Exp"
if project_root not in sys.path:
    sys.path.append(project_root)

from pythonProject.src.Structure_first.proxy_sample import ProxyStratifiedSampler
from pythonProject.src.algorithms.compute_truth import GroundTruthManager

def _worker_process_single_file(
    agg_file: str,
    aggregated_dir: str,
    all_T_true_results: dict,
    config: dict
):
    """
    [子进程工作函数] 处理单个文件，运行 FastestO 算法 (WO 基线) 并返回指标。
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
        # WO 内部只计算 sum(estimateW * oracle), 由于 estimateW 在生成时
        # 已经携带了 Upvotes（由Fastest输出），所以此处只需要对齐 T_true 即可正确计算 sum 的误差。
        sampler = ProxyStratifiedSampler(
            csv_path=filepath,
            is_multi_predicate=True,
            post_proxy=config["post_proxy"],
            comment_proxy=config["comment_proxy"],
            post_oracle=config["post_oracle"],
            comment_oracle=config["comment_oracle"],
            T_true=T_true
        )
        
        if sampler.posts.empty:
            return None

        # 运行基线 (Graph Only 意味着只经过 Fastest 图信息直接估算，对应你文章中的 WO)
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
    except Exception as e:
        return None


def evaluate_fastest_o_multi_predicate_dataset_fast(
    dataset_name: str,
    parent_dataset: str = "parler_data",  
    table1: str = "post",                 
    table2: str = "comment",              
    agg_mode: str = "count",   
    sum_on: str = "post",      
    sum_col: str = "upvotes", 
    post_proxy_col: str = "ML1_proxy4b_probability",
    comment_proxy_col: str = "ML2_proxy1_probability",
    post_oracle_col: str = "ML1_oracle2_probability",
    comment_oracle_col: str = "ML2_oracle2_probability",
    max_workers: int = None
):
    
    print(f"\n{'='*10} Evaluation: FastestO (Graph Only - {agg_mode.upper()}) {'='*10}")
    
    # 1. 处理 Ground Truth 读取
    gt_manager = GroundTruthManager(
        dataset_name=dataset_name,
        post_oracle_col=post_oracle_col,
        comment_oracle_col=comment_oracle_col,
        parent_dataset=parent_dataset,
        table1=table1,
        table2=table2
    )
    
    
    base, ext = os.path.splitext(gt_manager.cache_path)
    if agg_mode == "sum":
        safe_sum_col = str(sum_col).replace("/", "_").replace(":", "_")
        t_true_path = f"{base}_sum.json"
    else:
        t_true_path = f"{base}_count{ext}"
        if not os.path.exists(t_true_path):
            t_true_path = gt_manager.cache_path

    if os.path.exists(t_true_path):
        print(f"读入 T_true 数据集: {t_true_path}")
        with open(t_true_path, 'r') as f:
            all_T_true_results = json.load(f)
    else:
        print(f"[Error] 未能找到真值文件: {t_true_path}")
        return

    base_path = f"/home/wangshuo/resource/datasets/{parent_dataset}/{dataset_name}"
    aggregated_dir = os.path.join(base_path, "results", "aggregated_results")
    
    if not os.path.exists(aggregated_dir):
        print(f"[Error] 目录不存在: {aggregated_dir}")
        return

    agg_files = sorted([f for f in os.listdir(aggregated_dir) if f.endswith(".csv")])
    
    config = {
        "post_proxy": post_proxy_col, "comment_proxy": comment_proxy_col,
        "post_oracle": post_oracle_col, "comment_oracle": comment_oracle_col
    }

    metrics = {"Qerror": [], "Oracle_Cost_Post": [], "Oracle_Cost_Comment": [], "Total_Cost": []}
    detailed_results_list = []

    if max_workers is None:
        max_workers = max(1, os.cpu_count() - 2)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_worker_process_single_file, f, aggregated_dir, all_T_true_results, config) for f in agg_files]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {agg_mode.upper()} Queries"):
            result = future.result()
            if result:
                detailed_results_list.append(result)
                metrics["Qerror"].append(result["abs_error_rate"])
                metrics["Oracle_Cost_Post"].append(result["n_post"])
                metrics["Oracle_Cost_Comment"].append(result["n_comment"])
                metrics["Total_Cost"].append(result["Total_Cost"])


    if not metrics["Qerror"]: 
        print("[Warn] 无有效结果。")
        return

    detailed_results_list.sort(key=lambda x: x["abs_error_rate"], reverse=True)

    output_dir = os.path.join(base_path, "results", "efficiency")
    os.makedirs(output_dir, exist_ok=True)
    
    suffix = "_sum" if agg_mode == "sum" else "_count"
    
    output_json = os.path.join(output_dir, f"WO{suffix}.json")
    try:
        with open(output_json, 'w') as f:
            json.dump(detailed_results_list, f, indent=4)
        print(f" 详细误差分析已保存至 (按误差降序): {output_json}")
    except Exception as e:
        print(f" 保存 JSON 失败: {e}")

    avg_qerror = np.mean(metrics["Qerror"])
    output_csv = os.path.join(output_dir, f"WO_summary_{suffix}.csv")
    df_summary = pd.DataFrame([{
        "dataset_name": dataset_name, 
        "method": "WO", 
        "agg_mode": agg_mode,
        "num_queries": len(metrics["Qerror"]),
        "mean_are": avg_qerror, 
        "avg_total_cost": np.mean(metrics["Total_Cost"])
    }])
    df_summary.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)

    print(f"统计汇总已更新至: {output_csv}")
    print(f"Global Mean ARE ({agg_mode.upper()}): {avg_qerror:.6f}")

if __name__ == "__main__":
    
    evaluate_fastest_o_multi_predicate_dataset_fast(
        dataset_name="amazon_extend",  
        parent_dataset="amazon_data",
        table1="product",
        table2="review",
        agg_mode="sum",
        sum_on="product",
        sum_col="price",
        post_proxy_col="ML3_proxy2_probability",
        comment_proxy_col="ML2_proxy1_probability",
        post_oracle_col="ML3_oracle2_probability",
        comment_oracle_col="ML2_oracle1_probability"
    )
    
    # evaluate_fastest_o_multi_predicate_dataset_fast(
    #     dataset_name="dataset_test",  
    #     parent_dataset = "parler_data",  
    #     table1 = "post",                 
    #     table2  = "comment",              
    #     agg_mode  = "sum",   
    #     sum_on  = "post",      
    #     sum_col  = "upvotes",  
    #     post_proxy_col = "ML1_proxy4b_probability",
    #     comment_proxy_col = "ML2_proxy1_probability",
    #     post_oracle_col = "ML1_oracle2_probability",
    #     comment_oracle_col = "ML2_oracle2_probability",
    #     max_workers = None
    # )