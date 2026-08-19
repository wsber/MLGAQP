#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

# ==============================================================================
# 【核心机制】：零硬编码！全自动动态推算项目根目录
# 无论谁把代码 clone 到什么路径，这里都会自动解析到 .../PROXY
# ==============================================================================
CURRENT_FILE_PATH = Path(__file__).resolve()
# 向上回退 3 级到达项目根目录 (EXACT.py -> baseline -> src -> pythonProject -> PROXY)
DEFAULT_PROJECT_ROOT = str(CURRENT_FILE_PATH.parents[3])

if DEFAULT_PROJECT_ROOT not in sys.path:
    sys.path.insert(0, DEFAULT_PROJECT_ROOT)

from pythonProject.src.algorithms.compute_truth import GroundTruthManager

# ==============================================================================
# 数据集内置预设配置
# ==============================================================================
DATASET_PRESETS = {
    "amazon": {
        "table1": "product", "table2": "review",
        "post_oracle_col": "ML3_oracle2_probability",
        "comment_oracle_col": "ML2_oracle1_probability",
        "sum_on": "product", "sum_col": "price", "sum_labels": [12]
    },
    "parler": {
        "table1": "post", "table2": "comment",
        "post_oracle_col": "ML1_oracle2_probability",
        "comment_oracle_col": "ML2_oracle2_probability",
        "sum_on": "post", "sum_col": "upvotes", "sum_labels": [1]
    },
    "parler-E": {
        "table1": "post", "table2": "comment",
        "post_oracle_col": "ML1_oracle2_probability",
        "comment_oracle_col": "ML2_oracle2_probability",
        "sum_on": "post", "sum_col": "upvotes", "sum_labels": [2]
    }
}

def compute_dataset_ground_truth(base_dir: str, dataset: str, agg_mode: str):
    cfg = DATASET_PRESETS.get(dataset)
    if not cfg:
        raise ValueError(f"未受支持的数据集: {dataset}，可选: {list(DATASET_PRESETS.keys())}")

    # 动态拼接数据集路径 (如: /any_user_path/PROXY/datasets/parler)
    dataset_base = os.path.join(base_dir, "datasets", dataset)
    print("=" * 70)
    print(f"🚀 开始计算 Ground Truth ({agg_mode.upper()}) | 数据集: {dataset}")
    print(f"   • 项目根目录: {base_dir}")
    print(f"   • 数据集路径: {dataset_base}")
    print(f"   • 谓词 Oracle: {cfg['post_oracle_col']} & {cfg['comment_oracle_col']}")
    if agg_mode == "sum":
        print(f"   • 求和目标: 表={cfg['sum_on']}, 列={cfg['sum_col']}, 作用Label={cfg['sum_labels']}")
    print("=" * 70)

    gt = GroundTruthManager(
        dataset_name=dataset,
        post_oracle_col=cfg["post_oracle_col"],
        comment_oracle_col=cfg["comment_oracle_col"],
        parent_dataset=dataset,
        table1=cfg["table1"],
        table2=cfg["table2"]
    )

    # 动态设置 GroundTruthManager 内部路径，防止相对路径在不同工作目录下迷失
    gt.base_path = dataset_base
    gt.csv_dir = os.path.join(dataset_base, "csv_data")
    gt.t1_csv_path = os.path.join(dataset_base, "csv_data", f"{cfg['table1']}.csv")
    gt.t2_csv_path = os.path.join(dataset_base, "csv_data", f"{cfg['table2']}.csv")
    if hasattr(gt, 'table1_csv_path'): gt.table1_csv_path = gt.t1_csv_path
    if hasattr(gt, 'table2_csv_path'): gt.table2_csv_path = gt.t2_csv_path
    gt.gt_dir = os.path.join(dataset_base, "ground_truth", "structure_result")
    gt.results_dir = os.path.join(dataset_base, "results")

    # 1. 动态定位 core_nodes_config 文件
    core_candidates = [
        os.path.join(dataset_base, "data_graph", f"core_nodes_config_{agg_mode}.json"),
        os.path.join(dataset_base, "data_graph", "core_nodes_config.json")
    ]
    core_path = next((p for p in core_candidates if os.path.exists(p)), None)
    if not core_path:
        raise FileNotFoundError(f"未找到 core_nodes_config 文件于: {dataset_base}/data_graph/")
    
    with open(core_path, "r", encoding="utf-8") as f:
        core = json.load(f)

    # 2. 预加载源数据
    if agg_mode == "sum":
        source_data = gt._load_and_prepare_sources(agg_mode="sum", sum_on=cfg["sum_on"], sum_col=cfg["sum_col"])
    else:
        source_data = gt._load_and_prepare_sources(agg_mode="count")

    # 3. 读取目标查询列表
    ans_candidates = [
        os.path.join(dataset_base, "ground_truth", f"parler_ans_{agg_mode}.txt"),
        os.path.join(dataset_base, "ground_truth", "parler_ans.txt")
    ]
    ans_file_path = next((p for p in ans_candidates if os.path.exists(p)), None)
    if not ans_file_path:
        raise FileNotFoundError(f"未找到 ans 文件于: {dataset_base}/ground_truth/")

    target_queries = []
    with open(ans_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            qname = line.split()[0]
            if not qname.endswith(".graph"):
                qname = f"{qname}.graph"
            target_queries.append(qname)

    print(f"[*] 成功加载 {len(target_queries)} 个待计算查询 (读取自 {os.path.basename(ans_file_path)})")

    # 4. 循环计算 T_true
    all_T_true = {}
    target_labels = cfg["sum_labels"]

    for qbase_graph in tqdm(target_queries, desc=f"Computing T_true ({agg_mode.upper()})"):
        gt_candidates_file = [
            os.path.join(dataset_base, "ground_truth", "structure_result", f"{qbase_graph}_matches.csv"),
            os.path.join(dataset_base, "ground_truth", f"{qbase_graph}_matches.csv")
        ]
        gt_path = next((p for p in gt_candidates_file if os.path.exists(p)), None)

        if not gt_path:
            all_T_true[qbase_graph] = 0.0
            continue

        qconf = core.get(qbase_graph)
        if not qconf:
            all_T_true[qbase_graph] = 0.0
            continue

        target_uids = []
        for lbl in target_labels:
            uids = qconf.get(str(lbl)) or qconf.get(int(lbl))
            if uids: target_uids.extend(uids)

        if not target_uids and agg_mode == "sum":
            all_T_true[qbase_graph] = 0.0
            continue

        sum_match_cols = [f"u{int(uid)}" for uid in target_uids] if target_uids else None

        t_val = gt._compute_multi_predicate_polars(
            gt_path=gt_path,
            core_nodes_config=core,
            source_data=source_data,
            prob_threshold=0.5,
            agg_mode=agg_mode,
            sum_on=cfg["sum_on"] if agg_mode == "sum" else None,
            sum_col=cfg["sum_col"] if agg_mode == "sum" else None,
            sum_match_col=sum_match_cols,
        )
        all_T_true[qbase_graph] = float(t_val)

    # 5. 导出 JSON 真值文件
    out_dir = os.path.join(dataset_base, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_json_path = os.path.join(out_dir, f"T_true_{cfg['post_oracle_col']}_{cfg['comment_oracle_col']}_{agg_mode}.json")

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(all_T_true, f, indent=4, ensure_ascii=False)

    print(f"✅ [完成] {dataset} ({agg_mode.upper()}) T_true 已成功保存至:\n👉 {out_json_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute T_true Ground Truth for Subgraph Matching")
    # 默认值使用自动解析出的 DEFAULT_PROJECT_ROOT，他人无需传此参数也能自适应运行
    parser.add_argument("--base_dir", default=DEFAULT_PROJECT_ROOT, help="项目根目录")
    parser.add_argument("--dataset", required=True, choices=["amazon", "parler", "parler-E"], help="数据集名称")
    parser.add_argument("--agg_mode", required=True, choices=["count", "sum"], help="聚合模式")
    args = parser.parse_args()

    compute_dataset_ground_truth(base_dir=args.base_dir, dataset=args.dataset, agg_mode=args.agg_mode)