import os
import re
import sys
import math
import time
import tempfile
import traceback
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import json
import shutil
import pandas as pd
from datetime import datetime
import argparse 

project_root = "/home/wangshuo/projects/Neo4j_Exp"
if project_root not in sys.path:
    sys.path.append(project_root)
from pythonProject.src.Structure_first.fastest_pipeline import FastestGraphConverter, FastestEstimateMerger
from pythonProject.src.Structure_first.graph_sample import FastestRunner
from pythonProject.src.Structure_first.precision_submatching import ExactSubgraphMatcher
from pythonProject.src.Structure_first.proxy_sample import ProxyStratifiedSampler, compute_T_true
from pythonProject.src.Structure_first.proxy_sample import ProxyStratifiedSampler, compute_T_true
from pythonProject.src.Structure_first.proxy_sample import compute_T_true_polars

# ===========================
# 1: 定义参数解析函数
# ===========================
def parse_args():
    parser = argparse.ArgumentParser(description="Run method3 pipeline with custom dataset, proxy model, and oracle model.")
    parser.add_argument("--dataset", type=str, default="dataset_one", help="Name of the dataset (e.g., dataset_one, dataset_three)")
    parser.add_argument("--proxy_model", type=str, default="ML1_proxy4b1_probability", help="Name of the proxy model column")
    parser.add_argument("--oracle_model", type=str, default="ML1_oracle2_probability", help="Name of the oracle model column")  # 新增参数
    parser.add_argument("--run_times", type=int, default=20, help="Number of runs for each query")
    return parser.parse_args()

# 获取命令行参数
args = parse_args()
# ===========================
# ✅ 修改点 2: 使用参数动态配置路径
# ===========================
# 一级测试数据集
datasets_name = "parler_data"
# 一级数据集下根据查询和图结构的差异划分的子测试数据集
dataset_name = args.dataset  
proxy_model_name = args.proxy_model
oracle_model_name = args.oracle_model
run_times_config = args.run_times

print(f"🚀 Running with Dataset: {dataset_name}, Proxy Model: {proxy_model_name}, Run Times: {run_times_config}")


# 原始CSV数据路径
CSV_BASE_DIR = f"/home/wangshuo/resource/datasets/{datasets_name}/{dataset_name}/csv_data"
# 转换后GraphLib数据存放路径
Graph_Lib_Dir = f"/home/wangshuo/resource/datasets/{datasets_name}/{dataset_name}/data_graph"

# 初始化 Runner

# 评估参数
RUN_TIMES = 1  # 每种方法重复次数

current_budget = 20000
infer_label = 1  # 对应 Post 节点的标签
# runner = FastestRunner(build_dir="/home/wangshuo/projects/FaSTest-main/build")
# # 调用 run 方法
# code, output = runner.run(
#     dataset=dataset_name,
#     root_label=infer_label,           # 必须指定推理节点的标签
#     sample_budget=current_budget,     # 设置预算
#     # estimate_with_predicate=True      # <--- 开启单推理谓词模式
#     estimate_with_predicate=False      # <--- 关闭单推理谓词模式
# )


# ===========================
# = 配置区域（按需修改） =
# ===========================

print(dataset_name)

# Fastest对所有查询图的估计结果文件保存路径
SV_FILE = f"/home/wangshuo/resource/datasets/parler_data/{dataset_name}/results/in_estimateW_result.txt"
# 用户指定每个查询图中推理谓词节点
INFER_NODE_FILE = f"/home/wangshuo/resource/datasets/parler_data/{dataset_name}/data_graph/infer_node.txt"
# 图中节点 ID 与原始CSV文件中数据id的映射关系文件
IDMAP_FILE = f"/home/wangshuo/resource/datasets/parler_data/{dataset_name}/data_graph/id_mapping.csv"
# 原始 post.csv 文件路径
POST_CSV = f"/home/wangshuo/resource/datasets/parler_data/{dataset_name}/csv_data/post.csv"
# 存放真实结果的目录
GT_RESULT_DIR = f"/home/wangshuo/resource/datasets/parler_data/{dataset_name}/ground_truth/structure_result"
# 存放本原型系统结果的目录
OUTPUT_DIR = f"/home/wangshuo/resource/datasets/parler_data/{dataset_name}/results"
# 输出的包含每个查询post估计值w（x）的 post CSV 文件路径
POST_WITH_ESTIMATE_CSV = os.path.join(OUTPUT_DIR, "post_with_estimate.csv")
# 各方法结果的总结文件路径
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "results_summary.csv")
SUMMARY_TXT = os.path.join(OUTPUT_DIR, "results_summary.txt")
# 是否保留临时 CSV（用于调试）
KEEP_TEMP = False  
# ===========================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# 定义统计结果保存路径 (确保是全局变量或在函数内定义)
SAMPLED_COUNT_FILE = os.path.join(OUTPUT_DIR, "efficiency/sampled_node_count.csv")

# ---------------------------
# Main pipeline
# ---------------------------


print("[BEGIN] multi-query pipeline")

merger = FastestEstimateMerger(
    sv_file=SV_FILE,
    map_file=IDMAP_FILE,
    post_file=POST_CSV,
    output_file=POST_WITH_ESTIMATE_CSV
)
# 1) parse sv multi
sv_df = merger.parse_sv_multi(SV_FILE)
if sv_df is None or sv_df.empty:
    print("[ERROR] No sv records parsed. Exiting.")

# 2) read infer node list
infer_nodes = merger.read_infer_node_list(INFER_NODE_FILE)

# 3) build post_with_estimate.csv
merged_df = merger.build_post_with_estimates(sv_df=sv_df, idmap_file=IDMAP_FILE, post_csv=POST_CSV,
                                     out_csv=POST_WITH_ESTIMATE_CSV)

def save_node_counts(records: List[Dict]):
    """辅助函数：将节点计数追加到 CSV"""
    if not records: return
    df = pd.DataFrame(records)
    header = not os.path.exists(SAMPLED_COUNT_FILE)
    try:
        df.to_csv(SAMPLED_COUNT_FILE, mode='a', index=False, header=header)
    except Exception as e:
        print(f"[错误] 写入节点统计失败: {e}")

def evaluate_queries(
    merged_df: pd.DataFrame,
    sv_df: pd.DataFrame,
    infer_nodes: List[str],
    idmap_file: str,
    post_csv: str,
    gt_result_dir: str,
    output_dir: str,
    run_times: int = 50,  # ✅ 默认改为 50 次
    new_workload: bool = False,
    proxy_model = 'ML1_proxy4b1_probability',
    oracle_model = "ML1_oracle1_probability",  
    T_true_cache_file: str = f"/home/wangshuo/resource/datasets/parler_data/{dataset_name}/ground_truth/T_true.txt"
)  -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    对每个 query：
      - 运行 run_times 次采样。
      - 每次运行的结果直接保存到 output_dir/result_summarys/results_summary_run_{t}.csv 中。
    """
    print('[check proxy_model]', proxy_model)
    print('[check oracle_model]', oracle_model)
    print('[check T_true_cache_file]', T_true_cache_file)
    # ✅ 1. 准备保存分次结果的目录
    summarys_dir = os.path.join(output_dir, "result_summarys", proxy_model)
    os.makedirs(summarys_dir, exist_ok=True)
    print(f"[INFO] 分次实验结果将保存至: {summarys_dir}")

    # ✅ 2. 初始化/清空这 run_times 个文件 (如果需要覆盖旧结果)
    # 也可以选择追加模式，这里为了清晰，假设每次运行前清空或新建
    # 我们将在循环中以 'a' (append) 模式写入，所以先写入表头
    csv_headers = "query_index,query_basename,gt_match_col,T_true,method,T_hat,Qerror,n_post,n_comment\n"
    for t in range(run_times):
        run_file = os.path.join(summarys_dir, f"results_summary_run_{t+1}.csv")
        with open(run_file, "w") as f:
            f.write(csv_headers)

    rows = [] # 用于计算汇总均值（可选）
    txt_lines = []
    all_node_stats_records = []
    
    # Step 1: 读取/初始化 T_true 缓存
    # ...existing code...
    if os.path.exists(T_true_cache_file) and not new_workload:
        with open(T_true_cache_file, "r") as f:
            try:
                T_true_cache = json.load(f)
                print(f"[INFO] 已加载缓存的 T_true，共 {len(T_true_cache)} 条。")
            except json.JSONDecodeError:
                print(f"[WARN] {T_true_cache_file} 格式错误，将重新计算所有 T_true。")
                T_true_cache = {}
    else:
        T_true_cache = {}

    # order queries by index
    q_order = sv_df[["query_index", "query_basename"]].drop_duplicates().sort_values("query_index")
    q_order = q_order.reset_index(drop=True)

    n_queries = len(q_order)
    if len(infer_nodes) < n_queries:
        print(f"[WARN] infer_nodes length {len(infer_nodes)} < number of parsed queries {n_queries}. We'll map as many as available and default 'u1' for missing.")
    
    # Step 2: 遍历每个 query
    for i, r in q_order.iterrows():
        qi = int(r["query_index"])
        qbase = r["query_basename"]
        colname = f"estimate__{qi}__{qbase}"
        print("\n" + "="*60)
        print(f"[STEP] Query index={qi}, basename={qbase}, estimate column={colname}")

        # choose gt_match_col
        gt_match_col = infer_nodes[i] if i < len(infer_nodes) else "u1"
        print(f"[INFO] Using gt_match_col = {gt_match_col} for query #{qi}")

        # Step 3: 读取或计算 T_true
        # ...existing code...
        if (not new_workload) and (qbase in T_true_cache):
            T_true = float(T_true_cache[qbase])
            print(f"[CACHE] 读取缓存 T_true={T_true:.4f} for {qbase}")
        else:
            # locate GT file for this query
            candidates = [
                os.path.join(gt_result_dir, f"{qbase}_matches.csv"),
                os.path.join(gt_result_dir, f"{qbase}.graph_matches.csv"),
                os.path.join(gt_result_dir, f"{qbase}.matches.csv"),
                os.path.join(gt_result_dir, qbase),
            ]
            gt_path = None
            for p in candidates:
                if p and os.path.exists(p):
                    gt_path = p
                    break
            if gt_path is None:
                for fname in os.listdir(gt_result_dir):
                    if qbase in fname:
                        cand = os.path.join(gt_result_dir, fname)
                        if os.path.isfile(cand):
                            gt_path = cand
                            break
            if gt_path is None:
                print(f"[WARN] Ground-truth file for query {qbase} not found in {gt_result_dir}. Will use T_true=0.")
                T_true = 0.0
            else:
                try:
                    T_true = compute_T_true_polars(
                        gt_path=gt_path,
                        id_mapping_path=idmap_file,
                        post_csv_path=post_csv,
                        gt_match_col=gt_match_col,
                        prob_col=oracle_model,
                        prob_threshold=0.5
                    )
                except Exception as e:
                    print(f"[ERROR] compute_T_true_polars failed for {qbase} with gt_match_col={gt_match_col}: {e}")
                    traceback.print_exc()
                    T_true = 0.0
            # 写入缓存
            T_true_cache[qbase] = float(T_true)
            with open(T_true_cache_file, "w") as f:
                json.dump(T_true_cache, f, indent=2)
            print(f"[CACHE] 已更新缓存: {qbase} -> {T_true:.4f}")
            
        # Step 4: 临时 CSV prepare temp CSV for sampler
        if colname not in merged_df.columns:
            print(f"[WARN] estimate column {colname} not found in merged_df. Using zeros.")
            tmp_df = merged_df.copy()
            tmp_df["estimate"] = 0.0
        else:
            tmp_df = merged_df.copy()
            tmp_df["estimate"] = tmp_df[colname].astype(float).fillna(0.0)

        # create temp csv file
        tmp_csv = os.path.join(output_dir, f"tmp_post_with_estimate_q{qi}__{qbase}_{proxy_model}.csv")
        # ✅ 新增逻辑：检查缓存文件是否存在
        if os.path.exists(tmp_csv):
            print(f"[CACHE] Found existing estimate file, skipping generation: {tmp_csv}")
        else:
            # 文件不存在，从 merged_df 中提取并生成
            if colname not in merged_df.columns:
                print(f"[WARN] estimate column {colname} not found in merged_df. Using zeros.")
                tmp_df = merged_df.copy()
                tmp_df["estimate"] = 0.0
            else:
                tmp_df = merged_df.copy()
                tmp_df["estimate"] = tmp_df[colname].astype(float).fillna(0.0)

            print(f"[INFO] Creating new estimate file: {tmp_csv}")
            tmp_df.to_csv(tmp_csv, index=False)
        if not KEEP_TEMP:
            remove_tmp = True
        else:
            remove_tmp = False

        # instantiate sampler
        sampler = ProxyStratifiedSampler(csv_path=tmp_csv, T_true=T_true,
                                         is_multi_predicate=False,
                                         post_proxy=proxy_model,
                                         post_oracle=oracle_model,
                                        total_budget_frac=0.1,
                                         c_stage=0.15
                                        )

        methods = {
            # "pa_optimal": sampler.run_pa_optimal
            "UN": sampler.run_baseline_uniform,
            "PO": sampler.run_baseline_proxy,
            "MAB": sampler.run_mab_sampling,

            "FOIS_nrs": sampler.run_baseline_proxy_a,
            "FOIS_rs": sampler.run_baseline_proxy_a_unbiased_test1,
            "POSS": sampler.run_proxyE_importance,
            "POSSA": sampler.run_possa,
        }

        for mname, func in methods.items():
            T_list = []
            Q_list = []
            post_cnt_list = []
            comment_cnt_list = []
            
            print(f"\n--- Running {mname} for {run_times} times ---")
            for t in range(run_times):
                try:
                    out = func()
                    T_hat = float(out.get("T_hat", 0.0))
                    Qerror = float(out.get("Qerror", 1.0))
                    n_post = out.get("n_post", 0)
                    n_comment = out.get("n_comment", 0)
                    
                    T_list.append(T_hat)
                    Q_list.append(Qerror)
                    post_cnt_list.append(n_post)
                    comment_cnt_list.append(n_comment)
                    
                    # ✅ 核心修改：将第 t 次的结果直接写入对应的文件
                    run_file = os.path.join(summarys_dir, f"results_summary_run_{t+1}.csv")
                    with open(run_file, "a") as f:
                        # "query_index,query_basename,gt_match_col,T_true,method,T_hat,Qerror,n_post,n_comment"
                        line = f"{qi},{qbase},{gt_match_col},{T_true},{mname},{T_hat},{Qerror},{n_post},{n_comment}\n"
                        f.write(line)
                        
                except Exception as e:
                    print(f"❌ {mname} 第 {t+1} 次执行失败: {repr(e)}")
                    traceback.print_exc()
                    T_list.append(0.0)
                    Q_list.append(1.0)

            # 计算均值与标准差 (仅用于控制台输出和汇总文件，不影响分次文件)
            def trimmed_mean_std(lst):
                if len(lst) <= 2:
                    return np.mean(lst), np.std(lst)
                sorted_lst = sorted(lst)
                trimmed = sorted_lst[1:-1]
                return np.mean(trimmed), np.std(trimmed)
            
            T_mean, T_std = trimmed_mean_std(T_list)
            Q_mean, Q_std = trimmed_mean_std(Q_list)
            avg_post = int(np.mean(post_cnt_list))
            avg_comment = int(np.mean(comment_cnt_list))

            print(f"✅ {mname} 平均结果: T_hat={T_mean:.4f}±{T_std:.4f},  Qerror={Q_mean:.6f}±{Q_std:.6f}")
            
            # 写入汇总结果表 (保留原有逻辑)
            rows.append({
                "query_index": qi,
                "query_basename": qbase,
                "gt_match_col": gt_match_col,
                "T_true": float(T_true),
                "method": mname,
                "T_hat_mean": T_mean,
                "T_hat_std": T_std,
                "Qerror_mean": Q_mean,
                "Qerror_std": Q_std
            })

            txt_lines.append(
                f"{qbase} {gt_match_col} {mname} "
                f"T_hat={T_mean:.6f}±{T_std:.6f} "
                f"Qerror={Q_mean:.6f}±{Q_std:.6f}"
            )
            all_node_stats_records.append({
                "query_name": qbase,
                "method": mname,
                "post_sampled_cnt": avg_post,
                "comment_sampled_cnt": avg_comment
            })

        # cleanup tmp csv
        if remove_tmp:
            try:
                os.remove(tmp_csv)
                print(f"[INFO] Removed temp file: {tmp_csv}")
            except Exception:
                pass

    # write summary files
    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(SUMMARY_CSV, index=False)
    with open(SUMMARY_TXT, "w") as f:
        for ln in txt_lines:
            f.write(ln + "\n")
    if all_node_stats_records:
        save_node_counts(all_node_stats_records)
    
    print(f"[INFO] evaluate_queries: wrote summary to {SUMMARY_CSV}")
    print(f"[INFO] 分次详细结果已保存至 {summarys_dir} (共 {run_times} 个文件)")
    return df_summary, all_node_stats_records

# ✅ 直接调用 evaluate_queries，设置 run_times=50
evaluate_queries(
    merged_df=merged_df,
    sv_df=sv_df,
    infer_nodes=infer_nodes,
    idmap_file=IDMAP_FILE,
    post_csv=POST_CSV,
    gt_result_dir=GT_RESULT_DIR,
    output_dir=OUTPUT_DIR,
    run_times=run_times_config,  # ✅ 这里设置为 50
    new_workload=False, 
    proxy_model=proxy_model_name,
    oracle_model=oracle_model_name,
    T_true_cache_file=f"/home/wangshuo/resource/datasets/parler_data/{dataset_name}/ground_truth/T_true_{oracle_model_name}.txt"
)