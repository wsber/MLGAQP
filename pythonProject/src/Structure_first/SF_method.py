from pythonProject.src.Structure_first.fastest_pipeline import FastestGraphConverter, FastestEstimateMerger
from pythonProject.src.Structure_first.graph_sample import FastestRunner
from pythonProject.src.Structure_first.precision_submatching import ExactSubgraphMatcher
from pythonProject.src.Structure_first.proxy_sample import ProxyStratifiedSampler, compute_T_true
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
# 以下是单推理谓词的结构优先的整体流程示例

if __name__ == "__main__":

    # 以下是单推理谓词的结构优先的整体流程示例

    datasets_name = "parler_data"
    dataset_name = "dataset_one"
    CSV_BASE_DIR = f"/home/wangshuo/resource/datasets/{datasets_name}/{dataset_name}/csv_data"
    Graph_Lib_Dir = f"/home/wangshuo/resource/datasets/{datasets_name}/{dataset_name}/data_graph"
    # # 1. 将 CSV 数据转换为 GraphLib 格式，这个格式中边是带标签的
    # converter = FastestGraphConverter(CSV_BASE_DIR,Graph_Lib_Dir)
    # converter.run_without_author_user_post()
    # converter.remove_edge_labels()
    # 2.基于GraphLib数据，调用Fastest进行树采样或图采样，得到特定标签下所有或部分点的估计值
    runner = FastestRunner(
        build_dir="/home/wangshuo/projects/FaSTest-main/build"
    )
    # 默认执行 ./Fastest -d parler --ROOT_LABEL 1 (表示推理谓词所在节点的标签)
    code, output = runner.run(dataset="dataset_one", root_label=1)
    # 2.1同时调用准确的子图匹配算法得到真实结果的基数，这些准确子图匹配算法只能处理边不带标签的图
    # # 2.1.1 所以首先基于原来边带标签的图，精简成边不带标签的图
    # converter = FastestGraphConverter(CSV_BASE_DIR,Graph_Lib_Dir)
    # converter.simplify_graph_merge_edges_update_degree(input_path=Graph_Lib_Dir+'/parler.graph',
    #                                                    output_path='/home/wangshuo/resource/datasets/parler_data/dataset_one/ground_truth/data_graph/parler.graph')
    # # 2.1.2 然后对精简后的数据图调用准确子图匹配算法进行匹配，这里得到的GT对应的查询图，应该与Fastest对应的查询图等价（就算边没有标签也没有影响，且两点不存在多条边）
    matcher = ExactSubgraphMatcher(
        exe_path="/home/wangshuo/projects/SubgraphMatching/build/matching/SubgraphMatching.out",
        default_args=["-filter", "GQL", "-order", "GQL", "-engine", "LFTJ", "-num", "MAX"],
        timeout=300,
    )

    matcher.run_batch(
        data_graph="/home/wangshuo/resource/datasets/parler_data/dataset_one/data_graph/parler.graph",
        query_graph_dir="/home/wangshuo/resource/datasets/parler_data/dataset_one/query_graph",
        output_dir="/home/wangshuo/resource/datasets/parler_data/dataset_one/ground_truth",
    )
    # !/usr/bin/env python3
    # -*- coding: utf-8 -*-

    """
    处理 multi-query 的 Fastest 输出(in_estimateW_result.txt)：
      - 解析多个 Query 的每节点估计值（支持多个 Query 块）
      - 根据 INFER_NODE_FILE 中每行指定的 gt_match_col（例如 u1,u2）按顺序对应每个 Query
      - 将每个 Query 的 estimate 列并入 post_with_estimate.csv（列名 estimate__<query_basename>）
      - 针对每个 Query 用 ProxyStratifiedSampler 做评估（使用 compute_T_true 得到 T_true）
      - 输出 summary CSV / TXT

    使用说明：
     - 请确保能 import compute_T_true, FastestEstimateMerger, ProxyStratifiedSampler
       （把它们放在同一目录下的 gt_tools.py，然后 from gt_tools import ...）
    """


    # 改成你实际的模块名或把函数/类粘到顶部，这里假定它们在 gt_tools.py 中
    # from gt_tools import compute_T_true, FastestEstimateMerger, ProxyStratifiedSampler
    from pythonProject.src.Structure_first.proxy_sample import ProxyStratifiedSampler, compute_T_true

    # 如果你把 compute_T_true / FastestEstimateMerger / ProxyStratifiedSampler 放在同一脚本中，
    # 注释掉上面的 import 并保证函数/类在脚本中存在。

    # ===========================
    # = 配置区域（按需修改） =
    # ===========================
    SV_FILE = "/home/wangshuo/resource/datasets/parler_data/dataset_one/results/in_estimateW_result.txt"
    INFER_NODE_FILE = "/home/wangshuo/resource/datasets/parler_data/dataset_one/data_graph/infer_node.txt"
    IDMAP_FILE = "/home/wangshuo/resource/datasets/parler_data/dataset_one/data_graph/id_mapping.csv"
    POST_CSV = "/home/wangshuo/resource/datasets/parler_data/dataset_one/csv_data/post.csv"
    GT_RESULT_DIR = "/home/wangshuo/resource/datasets/parler_data/dataset_one/ground_truth/structure_result"
    OUTPUT_DIR = "/home/wangshuo/resource/datasets/parler_data/dataset_one/results"
    POST_WITH_ESTIMATE_CSV = os.path.join(OUTPUT_DIR, "post_with_estimate.csv")
    SUMMARY_CSV = os.path.join(OUTPUT_DIR, "results_summary.csv")
    SUMMARY_TXT = os.path.join(OUTPUT_DIR, "results_summary.txt")

    # 评估参数
    RUN_TIMES = 2  # 每种方法重复次数
    KEEP_TEMP = False  # 是否保留临时 CSV（用于调试）
    # ===========================

    os.makedirs(OUTPUT_DIR, exist_ok=True)


    # ---------------------------
    # Step D: 对每个 Query 计算 T_true 并评估
    # ---------------------------
    def evaluate_queries(
            merged_df: pd.DataFrame,
            sv_df: pd.DataFrame,
            infer_nodes: List[str],
            idmap_file: str,
            post_csv: str,
            gt_result_dir: str,
            output_dir: str,
            run_times: int = 3
    ) -> pd.DataFrame:
        """
        对每个 query（按 sv_df 中 query_index 出现顺序）：
          - 使用 infer_nodes 对应的 gt_match_col 计算 T_true（若找不到 GT 文件则 T_true=0）
          - 将对应的 estimate__col 作为 'estimate' 列写入临时 csv
          - 用 ProxyStratifiedSampler 执行四种方法（每种重复 run_times 次），取均值与 std
          - 返回 summary dataframe
        """
        rows = []
        txt_lines = []

        # order queries by index
        q_order = sv_df[["query_index", "query_basename"]].drop_duplicates().sort_values("query_index")
        q_order = q_order.reset_index(drop=True)

        n_queries = len(q_order)
        if len(infer_nodes) < n_queries:
            print(
                f"[WARN] infer_nodes length {len(infer_nodes)} < number of parsed queries {n_queries}. We'll map as many as available and default 'u1' for missing.")

        for i, r in q_order.iterrows():
            qi = int(r["query_index"])
            qbase = r["query_basename"]
            colname = f"estimate__{qi}__{qbase}"
            print("\n" + "=" * 60)
            print(f"[STEP] Query index={qi}, basename={qbase}, estimate column={colname}")

            # choose gt_match_col from infer_nodes by index (if available)
            gt_match_col = infer_nodes[i] if i < len(infer_nodes) else "u1"
            print(f"[INFO] Using gt_match_col = {gt_match_col} for query #{qi}")

            # locate GT file for this query
            # attempt a few plausible filenames in gt_result_dir
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
                # try to find any file in gt_result_dir that contains the qbase as substring
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
                    # compute_T_true should be available (user provided)
                    T_true = compute_T_true(
                        gt_path=gt_path,
                        id_mapping_path=idmap_file,
                        post_csv_path=post_csv,
                        gt_match_col=gt_match_col,
                        prob_col="ML1_oracle1_probability",
                        prob_threshold=0.5
                    )
                except Exception as e:
                    print(f"[ERROR] compute_T_true failed for {qbase} with gt_match_col={gt_match_col}: {e}")
                    traceback.print_exc()
                    T_true = 0.0

            # prepare temp CSV for sampler: copy merged_df and set 'estimate' = that col
            if colname not in merged_df.columns:
                print(f"[WARN] estimate column {colname} not found in merged_df. Using zeros.")
                tmp_df = merged_df.copy()
                tmp_df["estimate"] = 0.0
            else:
                tmp_df = merged_df.copy()
                tmp_df["estimate"] = tmp_df[colname].astype(float).fillna(0.0)

            # create temp csv file
            tmp_csv = os.path.join(output_dir, f"tmp_post_with_estimate_q{qi}__{qbase}.csv")
            tmp_df.to_csv(tmp_csv, index=False)
            if not KEEP_TEMP:
                remove_tmp = True
            else:
                remove_tmp = False

            # instantiate sampler (user-provided)
            sampler = ProxyStratifiedSampler(csv_path=tmp_csv, T_true=T_true)

            methods = {
                "proxy_importance": sampler.run_proxy_importance,
                "proxy_uniform": sampler.run_proxy_uniform,
                "proxyE_importance": sampler.run_proxyE_importance,
                "proxyE_uniform": sampler.run_proxyE_uniform
            }

            for mname, func in methods.items():
                T_list = []
                Q_list = []
                print(f"\n--- Running {mname} for {run_times} times ---")
                for t in range(run_times):
                    try:
                        out = func()
                        T_hat = float(out.get("T_hat", 0.0))
                        Qerror = float(out.get("Qerror", 1.0))
                        T_list.append(T_hat)
                        Q_list.append(Qerror)
                        # ✅ 实时打印每次结果
                        print(f"[{mname} | run {t + 1}/{run_times}]  T_hat={T_hat:.4f},  Qerror={Qerror:.6f}")
                    except Exception as e:
                        print(f"❌ {mname} 第 {t + 1} 次执行失败: {repr(e)}")
                        traceback.print_exc()
                        T_list.append(0.0)
                        Q_list.append(1.0)

                # 计算均值与标准差
                T_mean = float(np.mean(T_list))
                T_std = float(np.std(T_list))
                Q_mean = float(np.mean(Q_list))
                Q_std = float(np.std(Q_list))

                print(f"✅ {mname} 平均结果: T_hat={T_mean:.4f}±{T_std:.4f},  Qerror={Q_mean:.6f}±{Q_std:.6f}")

                # 写入结果表
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

                # 同时写入 TXT 文本（summary）
                txt_lines.append(
                    f"{qbase} {gt_match_col} {mname} "
                    f"T_hat={T_mean:.6f}±{T_std:.6f} "
                    f"Qerror={Q_mean:.6f}±{Q_std:.6f}"
                )

            # cleanup tmp csv
            if remove_tmp:
                try:
                    os.remove(tmp_csv)
                except Exception:
                    pass

        # write summary files
        df_summary = pd.DataFrame(rows)
        df_summary.to_csv(SUMMARY_CSV, index=False)
        with open(SUMMARY_TXT, "w") as f:
            for ln in txt_lines:
                f.write(ln + "\n")
        print(f"[INFO] evaluate_queries: wrote summary to {SUMMARY_CSV} and {SUMMARY_TXT}")
        return df_summary


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

    # 4) evaluate each query
    summary_df = evaluate_queries(
        merged_df=merged_df,
        sv_df=sv_df,
        infer_nodes=infer_nodes,
        idmap_file=IDMAP_FILE,
        post_csv=POST_CSV,
        gt_result_dir=GT_RESULT_DIR,
        output_dir=OUTPUT_DIR,
        run_times=RUN_TIMES
    )

    print("[END] pipeline completed")
    if summary_df is not None:
        print(summary_df.head())
