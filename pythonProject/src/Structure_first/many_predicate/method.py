import os
import sys
import json
import argparse
import traceback
import multiprocessing as mp
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# -----------------------
# Project path
# -----------------------
PROJECT_ROOT = os.environ.get("NEO4J_EXP_ROOT", "/home/wangshuo/projects/Neo4j_Exp")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from pythonProject.src.algorithms.proxy_sample import ProxyStratifiedSampler
from pythonProject.src.Structure_first.compute_truth import GroundTruthManager


# -----------------------
# Helpers
# -----------------------
def parse_query_basename(agg_file: str) -> str:
    if agg_file.startswith("aggregated_list_"):
        base = agg_file.replace("aggregated_list_", "")
    elif agg_file.startswith("aggregated_wide_"):
        base = agg_file.replace("aggregated_wide_", "")
    else:
        base = agg_file
    return base.replace(".csv", "")


def build_gt_match_col(core_nodes_config: Dict, query_basename_graph: str) -> str:
    core_nodes = core_nodes_config.get(query_basename_graph, {})
    return ";".join([f"u{vid}" for label in core_nodes for vid in core_nodes[label]])


def trimmed_mean_std(values: List[float]) -> Tuple[float, float]:
    if len(values) <= 2:
        return float(np.mean(values)), float(np.std(values))
    sorted_vals = sorted(values)
    trimmed = sorted_vals[1:-1]
    return float(np.mean(trimmed)), float(np.std(trimmed))


# def format_run_line(rec: Tuple[int, int, str, str, float, str, float, float, int, int]) -> str:
#     _, qi, qbase, gt_col, t_true, method, t_hat, qerr, n_post, n_comment = rec
#     return f"{qi},{qbase},{gt_col},{t_true},{method},{t_hat},{qerr},{n_post},{n_comment}\n"

def format_run_line(rec: Tuple) -> str:
    # 兼容旧代码，防止参数解包错误 (Tuple 长度变长了)
    if len(rec) == 10:
        _, qi, qbase, gt_col, t_true, method, t_hat, qerr, n_post, n_comment = rec
        ci_low, ci_high, eps = 0.0, 0.0, 0.0
    else:
        _, qi, qbase, gt_col, t_true, method, t_hat, qerr, n_post, n_comment, ci_low, ci_high, eps = rec
        
    return f"{qi},{qbase},{gt_col},{t_true},{method},{t_hat},{qerr},{n_post},{n_comment},{ci_low},{ci_high},{eps}\n"



# -----------------------
# Core worker
# -----------------------
def run_evaluation_for_query(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker: 评估单个聚合查询文件，返回汇总结果 + 分次运行记录。
    """
    try:
        aggregated_csv_path = task["aggregated_csv_path"]
        T_true = task["T_true"]
        post_proxy = task["post_proxy"]
        comment_proxy = task["comment_proxy"]
        post_oracle = task["post_oracle"]
        comment_oracle = task["comment_oracle"]
        runs = task["runs"]
        query_index = task["query_index"]
        gt_match_col = task["gt_match_col"]

        query_basename = os.path.basename(aggregated_csv_path)\
            .replace("aggregated_list_", "").replace("aggregated_wide_", "").replace(".csv", "")
        query_basename_graph = query_basename + ".graph"

        sampler = ProxyStratifiedSampler(
            csv_path=aggregated_csv_path,
            is_multi_predicate=True,
            post_proxy=post_proxy,
            comment_proxy=comment_proxy,
            post_oracle=post_oracle,
            comment_oracle=comment_oracle,
            T_true=T_true
        )

        if sampler.posts.empty:
            return {
                "ok": False,
                "error": f"No samples for {query_basename_graph}",
                "summary_records": [],
                "per_run_records": [],
                "node_stats_records": []
            }

        all_methods = {
            # "UN": sampler.run_baseline_uniform,
            # "PO": sampler.run_baseline_proxy,
            # "MAB": sampler.run_mab_sampling,
            # "FOIS_nrs": sampler.run_baseline_proxy_a,
            # "FOIS_rs": sampler.run_baseline_proxy_a_unbiased_test1,
            # "POSS": sampler.run_proxyE_importance,
            "POSSA": sampler.run_possa,
        }

        per_run_records: List[List[Tuple]] = [[] for _ in range(runs)]
        summary_records = []
        node_stats_records = []

        for name, func in all_methods.items():
            T_list, Q_list = [], []
            post_cnts, comment_cnts = [], []

            for t in range(runs):
                try:
                    out = func()
                    T_hat = float(out.get("T_hat", 0.0))
                    Qerror = float(out.get("Qerror", 1.0))
                    n_post = int(out.get("n_post", 0))
                    n_comment = int(out.get("n_comment", 0))

                    T_list.append(T_hat)
                    Q_list.append(Qerror)
                    post_cnts.append(n_post)
                    comment_cnts.append(n_comment)
                    ci_low, ci_high, eps = 0.0, 0.0, 0.0
                    if name == "POSSA" and hasattr(sampler, "calculate_confidence_interval"):
                        try:
                            # 默认 95% 置信度 (delta=0.05)
                            ci_res = sampler.calculate_confidence_interval(out, method='eb', alpha=0.2)
                            # ci_res = sampler.calculate_empirical_bernstein_bound(out, delta=0.05)
                            ci_low = float(ci_res.get("lower_bound", 0.0))
                            ci_high = float(ci_res.get("upper_bound", 0.0))
                            eps = float(ci_res.get("epsilon", 0.0))
                        except Exception as e:
                            # 容错处理，防止 CI 计算失败影响主流程
                            pass

                    # per_run_records[t].append(
                    #     (t, query_index, query_basename_graph, gt_match_col,
                    #      float(T_true), name, T_hat, Qerror, n_post, n_comment)
                    # )
                    per_run_records[t].append(
                        (t, query_index, query_basename_graph, gt_match_col,
                         float(T_true), name, T_hat, Qerror, n_post, n_comment,
                         ci_low, ci_high, eps)
                    )
                except Exception:
                    T_list.append(0.0)
                    Q_list.append(1.0)
                    per_run_records[t].append(
                        (t, query_index, query_basename_graph, gt_match_col,
                         float(T_true), name, 0.0, 1.0, 0, 0, 0.0, 0.0, 0.0)
                    )

            T_mean, T_std = trimmed_mean_std(T_list)
            Q_mean, Q_std = trimmed_mean_std(Q_list)
            avg_post = int(np.mean(post_cnts)) if post_cnts else 0
            avg_comment = int(np.mean(comment_cnts)) if comment_cnts else 0

            summary_records.append({
                "query_index": query_index,
                "query_basename": query_basename_graph,
                "gt_match_col": gt_match_col,
                "T_true": float(T_true),
                "method": name,
                "T_hat_mean": T_mean,
                "T_hat_std": T_std,
                "Qerror_mean": Q_mean,
                "Qerror_std": Q_std,
            })

            node_stats_records.append({
                "query_name": query_basename_graph,
                "method": name,
                "post_sampled_cnt": avg_post,
                "comment_sampled_cnt": avg_comment
            })

        return {
            "ok": True,
            "summary_records": summary_records,
            "per_run_records": per_run_records,
            "node_stats_records": node_stats_records
        }
    except Exception:
        return {
            "ok": False,
            "error": traceback.format_exc(),
            "summary_records": [],
            "per_run_records": [],
            "node_stats_records": []
        }


# -----------------------
# Main multi-process entry
# -----------------------
def multi_predicate_evaluation_mp(
    dataset_name: str,
    run_times: int = 50,
    post_proxy_col: str = "ML1_proxy4b1_probability",
    comment_proxy_col: str = "ML2_proxy1_probability",
    post_oracle_col: str = "ML1_oracle2_probability",
    comment_oracle_col: str = "ML2_oracle2_probability",
    workers: int = None
):
    print(f"\n========== 多谓词评估: {dataset_name} ==========")

    gt_manager = GroundTruthManager(
        dataset_name=dataset_name,
        post_oracle_col=post_oracle_col,
        comment_oracle_col=comment_oracle_col
    )
    all_T_true_results = gt_manager.get_all()
    if not all_T_true_results:
        print("[错误] 未能获取 T_true，评估中止。")
        return

    base_path = f"/home/wangshuo/resource/datasets/parler_data/{dataset_name}"
    aggregated_dir = os.path.join(base_path, "results", "aggregated_results")
    core_config_path = os.path.join(base_path, "data_graph", "core_nodes_config.json")
    final_report_path = os.path.join(base_path, "results", "results_summary2.csv")

    if not os.path.exists(aggregated_dir):
        print(f"[错误] 聚合结果目录不存在: {aggregated_dir}")
        return

    try:
        with open(core_config_path, "r") as f:
            core_nodes_config = json.load(f)
    except FileNotFoundError:
        print(f"[错误] 核心节点配置文件不存在: {core_config_path}")
        return

    agg_files = [f for f in os.listdir(aggregated_dir) if f.endswith(".csv")]
    if not agg_files:
        print(f"[警告] 目录 {aggregated_dir} 下无任何聚合文件。")
        return

    safe_post = post_proxy_col.replace("/", "_")
    safe_comment = comment_proxy_col.replace("/", "_")
    summarys_dir = os.path.join(base_path, "results", "result_summarys", f"{safe_post}_{safe_comment}")
    os.makedirs(summarys_dir, exist_ok=True)
    print(f"[INFO] 分次结果目录: {summarys_dir}")

    sorted_query_basenames = sorted(list(all_T_true_results.keys()))

    tasks = []
    for agg_file in sorted(agg_files):
        query_basename = parse_query_basename(agg_file)
        query_basename_graph = query_basename + ".graph"
        T_true_for_query = all_T_true_results.get(query_basename_graph)
        if T_true_for_query is None:
            print(f"[跳过] T_true 缓存中缺少 {query_basename_graph}")
            continue

        try:
            query_index = sorted_query_basenames.index(query_basename_graph)
        except ValueError:
            query_index = -1

        gt_match_col_str = build_gt_match_col(core_nodes_config, query_basename_graph)
        filepath = os.path.join(aggregated_dir, agg_file)

        tasks.append({
            "aggregated_csv_path": filepath,
            "T_true": T_true_for_query,
            "post_proxy": post_proxy_col,
            "comment_proxy": comment_proxy_col,
            "post_oracle": post_oracle_col,
            "comment_oracle": comment_oracle_col,
            "runs": run_times,
            "query_index": query_index,
            "gt_match_col": gt_match_col_str
        })

    if not tasks:
        print("[结束] 无可评估的查询。")
        return

    if workers is None or workers <= 0:
        workers = max(1, mp.cpu_count() - 1)

    print(f"[INFO] 多进程启动: workers={workers}, queries={len(tasks)}, runs={run_times}")

    per_run_lines: List[List[Tuple]] = [[] for _ in range(run_times)]
    final_report_records = []
    all_node_stats = []

    with mp.Pool(processes=workers) as pool:
        for res in tqdm(
            pool.imap_unordered(run_evaluation_for_query, tasks),
            total=len(tasks),
            desc="Evaluating",
            ncols=80
        ):
            if not res.get("ok"):
                print("[WARN] 子任务失败:")
                print(res.get("error", "unknown error"))
                continue
            final_report_records.extend(res["summary_records"])
            all_node_stats.extend(res["node_stats_records"])

            per_run_records = res["per_run_records"]
            for t in range(run_times):
                per_run_lines[t].extend(per_run_records[t])

    # header = "query_index,query_basename,gt_match_col,T_true,method,T_hat,Qerror,n_post,n_comment\n"
    header = "query_index,query_basename,gt_match_col,T_true,method,T_hat,Qerror,n_post,n_comment,ci_lower,ci_upper,epsilon\n"
    for t in range(run_times):
        run_file = os.path.join(summarys_dir, f"results_summary_run_{t+1}.csv")
        with open(run_file, "w") as f:
            f.write(header)
            for rec in per_run_lines[t]:
                f.write(format_run_line(rec))

    if not final_report_records:
        print("[结束] 没有生成汇总结果。")
        return

    report_df = pd.DataFrame.from_records(final_report_records)
    if "query_index" in report_df.columns and "method" in report_df.columns:
        report_df.sort_values(by=["query_index", "method"], inplace=True)
    report_df.to_csv(final_report_path, index=False)

    if all_node_stats:
        efficiency_dir = os.path.join(base_path, "results", "efficiency")
        os.makedirs(efficiency_dir, exist_ok=True)
        sampled_count_file = os.path.join(efficiency_dir, "sampled_node_count.csv")
        header_needed = not os.path.exists(sampled_count_file)
        pd.DataFrame(all_node_stats).to_csv(
            sampled_count_file, mode="a", index=False, header=header_needed
        )

    print(f"\n✅ 最终评估报告: {final_report_path}")
    print(f"✅ 分次详细数据: {summarys_dir} (共 {run_times} 个 run_*.csv)")
    print("✅ 多进程评估完成。")


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-predicate evaluation with multiprocessing.")
    parser.add_argument("--dataset", type=str, default="dataset_test2")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--post_proxy", type=str, default="ML1_proxy2b_probability")
    parser.add_argument("--comment_proxy", type=str, default="ML2_proxy1_probability")
    parser.add_argument("--post_oracle", type=str, default="ML1_oracle2_probability")
    parser.add_argument("--comment_oracle", type=str, default="ML2_oracle2_probability")
    parser.add_argument("--workers", type=int, default=0, help="0 或负数表示自动设置")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    multi_predicate_evaluation_mp(
        dataset_name=args.dataset,
        run_times=args.runs,
        post_proxy_col=args.post_proxy,
        comment_proxy_col=args.comment_proxy,
        post_oracle_col=args.post_oracle,
        comment_oracle_col=args.comment_oracle,
        workers=args.workers
    )