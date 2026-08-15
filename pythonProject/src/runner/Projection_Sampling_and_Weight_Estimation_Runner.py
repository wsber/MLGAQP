import os
import sys
import argparse
from typing import Dict
import pandas as pd
import numpy as np
import time

# 动态添加项目根目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pythonProject.src.Structure_first.graph_sample import FastestRunner

# =========================================================================
# 1. 高速内存数据处理核心
# =========================================================================

def load_and_prepare_mappings(id_mapping_path: str) -> pd.DataFrame:
    """读取 id_mapping.csv 并准备连接用的 DataFrame"""
    id_map_df = pd.read_csv(id_mapping_path, dtype={'internal_id': str, 'orig_id': str, 'type': str})
    id_map_df.rename(columns={'internal_id': 'node_id'}, inplace=True)
    if 'type' in id_map_df.columns:
        id_map_df['type'] = id_map_df['type'].str.lower()
    return id_map_df

def load_source_csvs(table1_csv_path: str, table2_csv_path: str, table1_name: str, table2_name: str) -> Dict[str, pd.DataFrame]:
    """读取实体数据表"""
    t1_df = pd.read_csv(table1_csv_path, dtype=str)
    t2_df = pd.read_csv(table2_csv_path, dtype=str)
    if 'id:ID' in t1_df.columns: t1_df.rename(columns={'id:ID': 'orig_id'}, inplace=True)
    if 'id:ID' in t2_df.columns: t2_df.rename(columns={'id:ID': 'orig_id'}, inplace=True)
    return {table1_name: t1_df, table2_name: t2_df}

def process_all_queries_in_memory(
    raw_csv_path: str, id_map_df: pd.DataFrame, sources: Dict[str, pd.DataFrame], 
    output_dir: str, table1: str, table2: str
):
    """
    【核心提速引擎】：全局哈希连接与内存聚合，彻底消灭循环中的 IO 和 Merge
    """
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n[*] 开始全局加载原始大表: {raw_csv_path}")
    df_raw = pd.read_csv(raw_csv_path)
    if df_raw.empty:
        print("[警告] 原始结果表为空！")
        return
        
    df_raw['node_id'] = df_raw['node_id'].astype(str)
    # 清洗 query_name，去掉后缀
    df_raw['query_name'] = df_raw['query_name'].astype(str).str.replace(r"(_estimateW_result)?\.csv$", "", regex=True)

    print(f"[+] 原始表加载完毕，共 {len(df_raw)} 行。开始全局 Merge...")

    # 1. 全局 Merge ID 映射
    df_mapped = pd.merge(df_raw, id_map_df, on='node_id', how='left')

    # 定义期望的 ML 列
    expected_ml1_cols = [
        'ML1_oracle1_probability', 'ML1_oracle2_probability', 'ML1_proxy1b_probability', 
        'ML1_proxy2b_probability', 'ML1_proxy_adv', 'ML1_proxy4b_probability', 
        'ML1_proxy6b_probability', 'ML3_oracle1_probability', 'ML3_proxy1_probability',
        'ML3_oracle2_probability', 'ML3_proxy3_probability', 'ML3_proxy2_probability'
    ]
    expected_ml2_cols = [
        'ML2_oracle1_probability','ML2_oracle2_probability', 'ML2_proxy2d2_probability', 
        'ML2_proxy4d2_probability', 'ML2_proxy1_probability', 'ML2_proxy2_probability', 'ML2_proxy4_probability'
    ]

    print("[+] 开始处理 Table 1 (如 Product/Post) 数据...")
    # 2. 全局过滤并 Merge Table 1
    t1_mask = df_mapped['type'] == table1.lower()
    df_t1 = pd.merge(df_mapped[t1_mask], sources[table1], on='orig_id', how='left')
    
    actual_ml1_cols = [col for col in expected_ml1_cols if col in df_t1.columns]
    agg_dict_t1 = {col: list for col in actual_ml1_cols}
    agg_dict_t1['orig_id'] = list
    # 按 query_name 和 instance_id 全局聚合
    agg_t1 = df_t1.groupby(['query_name', 'instance_id']).agg(agg_dict_t1).reset_index()
    agg_t1.rename(columns={'orig_id': 'post_id_list'}, inplace=True)

    print("[+] 开始处理 Table 2 (如 Review/Comment) 数据...")
    # 3. 全局过滤并 Merge Table 2
    t2_mask = df_mapped['type'] == table2.lower()
    df_t2 = pd.merge(df_mapped[t2_mask], sources[table2], on='orig_id', how='left')
    
    actual_ml2_cols = [col for col in expected_ml2_cols if col in df_t2.columns]
    agg_dict_t2 = {col: list for col in actual_ml2_cols}
    agg_dict_t2['orig_id'] = list
    # 按 query_name 和 instance_id 全局聚合
    agg_t2 = df_t2.groupby(['query_name', 'instance_id']).agg(agg_dict_t2).reset_index()
    agg_t2.rename(columns={'orig_id': 'comment_id_list'}, inplace=True)

    print("[+] 正在组合基础数据并合并最终宽表...")
    # 4. 提取基础信息表
    base_info = df_raw[['query_name', 'instance_id', 'estimateW', 'global_estimateW']].drop_duplicates()
    
    # 5. 组装最终结果
    final_df = pd.merge(base_info, agg_t1, on=['query_name', 'instance_id'], how='left')
    final_df = pd.merge(final_df, agg_t2, on=['query_name', 'instance_id'], how='left')

    # 规范化列顺序
    final_columns_order = [
        'query_name', 'instance_id', 'estimateW', 'global_estimateW', 
        'post_id_list', 'comment_id_list' 
    ] + expected_ml1_cols + expected_ml2_cols
    
    # 只保留实际存在的列
    final_columns_order = [col for col in final_columns_order if col in final_df.columns]
    final_df = final_df.reindex(columns=final_columns_order)

    print(f"[*] 全局内存处理完毕！耗时: {time.time() - start_time:.2f} 秒。正在高速导出到硬盘...")
    
    # 6. 一次性遍历导出小文件 (仅涉及写出，极快)
    num_files = 0
    grouped_final = final_df.groupby('query_name')
    for query_name, group_df in grouped_final:
        out_path = os.path.join(output_dir, f"aggregated_list_{query_name}.csv")
        # 写出时丢掉 query_name 列
        group_df.drop(columns=['query_name']).to_csv(out_path, index=False)
        num_files += 1
        
    print(f"[✓] 导出完成！共生成 {num_files} 个最终文件。总耗时: {time.time() - start_time:.2f} 秒。")

# =========================================================================
# 2. 全流程主控制流水线
# =========================================================================

def run_pipeline(
    base_dir: str, dataset: str, sample_budget: int, agg_func: str,
    sum_table: str, sum_col: str, sum_label: int, table1: str, table2: str,
    c_build_dir: str, run_cpp: bool
):
    print("=" * 70)
    print(f"🚀 开始执行【极速版】核心实例抽取与聚合流水线 ({dataset})")
    print("=" * 70)

    dataset_path = os.path.join(base_dir, "datasets", dataset)
    results_dir = os.path.join(dataset_path, "results")
    aggregated_dir = os.path.join(results_dir, "aggregated_results")
    
    id_mapping_path = os.path.join(dataset_path, "data_graph", "id_mapping.csv")
    t1_csv_path = os.path.join(dataset_path, "csv_data", f"{table1}.csv")
    t2_csv_path = os.path.join(dataset_path, "csv_data", f"{table2}.csv")
    raw_ins_csv_path = os.path.join(results_dir, "ins_estimateW_result.csv")

    if run_cpp:
        print(f"\n[Step 0] 启动 C++ FastestRunner ...")
        c_build = c_build_dir if c_build_dir else os.path.join(base_dir, "cProject", "build")
        runner = FastestRunner(build_dir=c_build)
        extra_args = ["--BASE_DIR", os.path.join(base_dir, "datasets"), "--AGG_FUNC", agg_func]
        if agg_func.lower() == "sum":
            extra_args.extend(["--SUM_TABLE", sum_table, "--SUM_COL", sum_col, "--SUM_LABEL", str(sum_label)])
        code, output = runner.run(dataset=dataset, root_label=-1, sample_budget=sample_budget, extra_args=extra_args)
        if code != 0:
            print(f"[错误] C++ 端执行失败 (Code {code})"); return

    print(f"\n[Step 1] 加载元数据与实体表...")
    id_map_df = load_and_prepare_mappings(id_mapping_path)
    sources = load_source_csvs(t1_csv_path, t2_csv_path, table1, table2)

    print(f"\n[Step 2] 启动全局内存向量化聚合 (Global In-Memory Aggregation)...")
    # 直接废弃拆分小文件，一把梭处理完！
    process_all_queries_in_memory(
        raw_csv_path=raw_ins_csv_path,
        id_map_df=id_map_df,
        sources=sources,
        output_dir=aggregated_dir,
        table1=table1,
        table2=table2
    )
        
    print("\n" + "=" * 70)
    print(f"🎉 数据集 [{dataset}] 处理完毕！")
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=PROJECT_ROOT)
    parser.add_argument("--dataset", type=str, default="parler")
    parser.add_argument("--sample_budget", type=int, default=60000)
    parser.add_argument("--agg_func", type=str, default="count", choices=["count", "sum"])
    parser.add_argument("--sum_table", type=str, default="product")
    parser.add_argument("--sum_col", type=str, default="price")
    parser.add_argument("--sum_label", type=int, default=12)
    parser.add_argument("--table1", type=str, default="product")
    parser.add_argument("--table2", type=str, default="review")
    parser.add_argument("--run_cpp", action="store_true") # 默认是不跑
    parser.add_argument("--c_build_dir", type=str, default=None)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run_pipeline(
        base_dir=args.base_dir, dataset=args.dataset, sample_budget=args.sample_budget,
        agg_func=args.agg_func, sum_table=args.sum_table, sum_col=args.sum_col,
        sum_label=args.sum_label, table1=args.table1, table2=args.table2,
        c_build_dir=args.c_build_dir, run_cpp=args.run_cpp
    )