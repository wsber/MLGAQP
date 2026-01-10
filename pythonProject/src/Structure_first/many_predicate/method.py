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
project_root = "/home/wangshuo/projects/Neo4j_Exp"
if project_root not in sys.path:
    sys.path.append(project_root)
from pythonProject.src.Structure_first.fastest_pipeline import FastestGraphConverter, FastestEstimateMerger
from pythonProject.src.Structure_first.graph_sample import FastestRunner
from pythonProject.src.Structure_first.precision_submatching import ExactSubgraphMatcher
from pythonProject.src.Structure_first.proxy_sample import ProxyStratifiedSampler, compute_T_true

# 一级测试数据集
datasets_name = "parler_data"
# 一级数据集下根据查询和图结构的差异划分的子测试数据集
# dataset_name = "dataset_test"
dataset_name = "dataset_test2"
# 原始CSV数据路径
CSV_BASE_DIR = f"/home/wangshuo/resource/datasets/{datasets_name}/{dataset_name}/csv_data"
# 转换后GraphLib数据存放路径
Graph_Lib_Dir = f"/home/wangshuo/resource/datasets/{datasets_name}/{dataset_name}/data_graph"
# converter = FastestGraphConverter(CSV_BASE_DIR,Graph_Lib_Dir)
# converter.run_without_author_user_post()
# converter.remove_edge_labels()
# dataset_name = "dataset_one"
runner = FastestRunner(build_dir="/home/wangshuo/projects/FaSTest-main/build")
print(dataset_name)
sample_budget = 20000
# 默认执行 ./Fastest -d parler --ROOT_LABEL 1 (表示推理谓词所在节点的标签)
code, output = runner.run(dataset=dataset_name, root_label=-1,sample_budget=sample_budget)
import pandas as pd
import os

def split_results_by_query(input_file_path, output_dir):
    """
    拆分多查询的结果文件为单查询的结果文件。

    Args:
        input_file_path (str): 输入的 ins_estimateW_result.csv 文件路径。
        output_dir (str): 拆分后的小文件要保存的目录。
    """
    print(f"--- 开始处理文件: {input_file_path} ---")

    # 检查输入文件是否存在
    if not os.path.exists(input_file_path):
        print(f"[错误] 输入文件不存在: {input_file_path}")
        return

    # 检查输出目录是否存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    try:
        # 使用 Pandas 读取整个 CSV 文件
        df = pd.read_csv(input_file_path)
        print(f"成功读取 {len(df)} 行数据。")

        # 检查 'query_name' 列是否存在
        if 'query_name' not in df.columns:
            print("[错误] CSV文件中缺少 'query_name' 列。请检查文件格式。")
            return

        # 按照 'query_name' 列的值进行分组
        grouped = df.groupby('query_name')
        
        num_files_created = 0
        for query_name, group_df in grouped:
            # 1. 构造输出文件名
            base_name = os.path.splitext(query_name)[0]
            
            # +++ 修改下面这行，将后缀名从 .txt 改为 .csv +++
            output_filename = f"{base_name}.csv"
            
            output_filepath = os.path.join(output_dir, output_filename)
            
            print(f"  正在处理查询 '{query_name}' -> 输出到 '{output_filepath}'...")

            # 2. 准备要写入的数据
            output_df = group_df.drop('query_name', axis=1)

            # 3. 将处理后的数据写入新的 CSV 文件
            output_df.to_csv(output_filepath, index=False)
            
            num_files_created += 1

        print(f"\n--- 处理完成 ---")
        print(f"总共拆分出 {num_files_created} 个查询文件。")

    except Exception as e:
        print(f"[严重错误] 处理过程中发生异常: {e}")

# ===================================================================
# --- Jupyter Notebook 配置区 ---
# 请在这里修改您的输入文件路径和输出目录路径
# ===================================================================

# 假设您的数据集名称是 'dataset_test'
dataset_name = 'dataset_test2'

# 构造输入文件和输出目录的路径
# 请确保这里的路径与您的实际文件结构一致
base_path = f"/home/wangshuo/resource/datasets/parler_data/{dataset_name}/results/"
input_csv_path = os.path.join(base_path, "ins_estimateW_result.csv")
# 将拆分后的文件保存到 'structure_estimate' 子目录中
output_directory = os.path.join(base_path, "structure_estimate/")

# 调用主函数
split_results_by_query(input_csv_path, output_directory)

# ===================================================================

import pandas as pd
import os
from typing import Dict

# ... (load_and_prepare_mappings 和 load_source_csvs 函数保持不变) ...
def load_and_prepare_mappings(id_mapping_path: str) -> pd.DataFrame:
    """读取id_mapping.csv并准备好用于连接的DataFrame。"""
    if not os.path.exists(id_mapping_path):
        raise FileNotFoundError(f"ID映射文件不存在: {id_mapping_path}")
        
    id_map_df = pd.read_csv(id_mapping_path, dtype={'internal_id': str, 'orig_id': str, 'type': str})
    id_map_df.rename(columns={'internal_id': 'node_id'}, inplace=True)
    
    print(f"加载了 {len(id_map_df)} 条ID映射记录。")
    return id_map_df

def load_source_csvs(post_csv_path: str, comment_csv_path: str) -> Dict[str, pd.DataFrame]:
    """读取原始的 post.csv 和 comment.csv 文件。"""
    if not os.path.exists(post_csv_path):
        raise FileNotFoundError(f"post.csv 文件不存在: {post_csv_path}")
    if not os.path.exists(comment_csv_path):
        raise FileNotFoundError(f"comment.csv 文件不存在: {comment_csv_path}")
        
    post_df = pd.read_csv(post_csv_path, dtype=str)
    comment_df = pd.read_csv(comment_csv_path, dtype=str)
    
    if 'id:ID' in post_df.columns:
        post_df.rename(columns={'id:ID': 'orig_id'}, inplace=True)
    if 'id:ID' in comment_df.columns:
        comment_df.rename(columns={'id:ID': 'orig_id'}, inplace=True)
        
    print(f"加载了 {len(post_df)} 行 post 数据和 {len(comment_df)} 行 comment 数据。")
    return {"Post": post_df, "Comment": comment_df}


def process_single_query_file_correctly(query_file_path: str, id_map_df: pd.DataFrame, sources: Dict[str, pd.DataFrame], output_dir: str):
    """
    处理单个长格式查询文件，聚合 ML 值和【节点ID列表】。
    """
    query_basename = os.path.basename(query_file_path).replace("_estimateW_result.csv", "")
    print(f"\n--- 正在处理查询: {query_basename} ---")
    
    instance_df = pd.read_csv(query_file_path)
    if instance_df.empty:
        print("文件为空，跳过。")
        return
        
    instance_df['node_id'] = instance_df['node_id'].astype(str)
    
    # 1. 连接ID映射
    merged_with_map = pd.merge(instance_df, id_map_df, on='node_id', how='left')

    # 2. 定义期望的 ML 列
    expected_ml1_cols = ['ML1_oracle1_probability', 'ML1_proxy4b1_probability', 'ML1_proxy2b1_probability']
    expected_ml2_cols = ['ML2_oracle2_probability', 'ML2_proxy2d2_probability', 'ML2_proxy4d2_probability', 'ML2_proxy1_probability']
    
    # 3. 分别处理 Post 和 Comment 数据流
    
    # --- Post 数据流 ---
    posts_data = merged_with_map[merged_with_map['type'] == 'Post'].copy()
    posts_joined = pd.merge(posts_data, sources['Post'], on='orig_id', how='left')
    
    # --- Comment 数据流 ---
    comments_data = merged_with_map[merged_with_map['type'] == 'Comment'].copy()
    comments_joined = pd.merge(comments_data, sources['Comment'], on='orig_id', how='left')
    
    # 4. 最终组装与聚合
    
    # --- 分别对 Post 和 Comment 进行聚合 ---
    
    # === 【修改点 1】：聚合 Post 数据（增加 orig_id） ===
    actual_ml1_cols = [col for col in expected_ml1_cols if col in posts_joined.columns]
    if not posts_joined.empty:
        # 定义聚合字典：ML列聚合为列表，orig_id 也聚合为列表
        agg_dict = {col: list for col in actual_ml1_cols}
        agg_dict['orig_id'] = list  # +++ 新增：收集节点ID +++
        
        agg_posts = posts_joined.groupby('instance_id').agg(agg_dict).reset_index()
        # 重命名 orig_id 为 post_id_list
        agg_posts.rename(columns={'orig_id': 'post_id_list'}, inplace=True) # +++ 重命名 +++
    else:
        # 如果为空，创建带有所需列的空 DataFrame
        agg_posts = pd.DataFrame(columns=['instance_id', 'post_id_list'] + actual_ml1_cols)

    # === 【修改点 2】：聚合 Comment 数据（增加 orig_id） ===
    actual_ml2_cols = [col for col in expected_ml2_cols if col in comments_joined.columns]
    if not comments_joined.empty:
        agg_dict = {col: list for col in actual_ml2_cols}
        agg_dict['orig_id'] = list  # +++ 新增：收集节点ID +++
        
        agg_comments = comments_joined.groupby('instance_id').agg(agg_dict).reset_index()
        # 重命名 orig_id 为 comment_id_list
        agg_comments.rename(columns={'orig_id': 'comment_id_list'}, inplace=True) # +++ 重命名 +++
    else:
        agg_comments = pd.DataFrame(columns=['instance_id', 'comment_id_list'] + actual_ml2_cols)
        
    # 创建基础表
    base_agg_df = instance_df[['instance_id', 'estimateW', 'global_estimateW']].groupby('instance_id').first().reset_index()
    
    # --- 合并聚合后的结果 ---
    final_df = pd.merge(base_agg_df, agg_posts, on='instance_id', how='left')
    final_df = pd.merge(final_df, agg_comments, on='instance_id', how='left')

    print(f"聚合完成，生成 {len(final_df)} 条实例记录。")

    # 5. 保存结果
    output_filename = f"aggregated_list_{query_basename}.csv"
    output_filepath = os.path.join(output_dir, output_filename)
    
    # === 【修改点 3】：更新最终列顺序，加入 ID 列表列 ===
    final_columns_order = [
        'instance_id', 'estimateW', 'global_estimateW', 
        'post_id_list', 'comment_id_list' # +++ 确保这两列被包含 +++
    ] + expected_ml1_cols + expected_ml2_cols
    
    final_df = final_df.reindex(columns=final_columns_order)
    
    final_df.to_csv(output_filepath, index=False)
    print(f"[完成] 结果已保存到: {output_filepath}")



dataset_name = 'dataset_test2'
"""主执行函数"""
print(f"====== 开始处理数据集: {dataset_name} ======")

# --- 路径配置 ---
base_path = f"/home/wangshuo/resource/datasets/parler_data/{dataset_name}"
estimate_dir = os.path.join(base_path, "results", "structure_estimate")
id_mapping_path = os.path.join(base_path, "data_graph", "id_mapping.csv")
post_csv_path = os.path.join(base_path, "csv_data", "post.csv")
comment_csv_path = os.path.join(base_path, "csv_data", "comment.csv")
output_dir = os.path.join(base_path, "results", "aggregated_results")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"创建输出目录: {output_dir}")

# --- 执行流程 ---
try:
    id_map_df = load_and_prepare_mappings(id_mapping_path)
    sources = load_source_csvs(post_csv_path, comment_csv_path)

    query_files = [f for f in os.listdir(estimate_dir) if f.endswith('.csv')]
    if not query_files:
        print(f"[警告] 在目录 {estimate_dir} 中没有找到任何 .csv 结果文件。")
        
    for query_file in sorted(query_files):
        query_file_path = os.path.join(estimate_dir, query_file)
        # 调用新的、正确的处理函数
        process_single_query_file_correctly(query_file_path, id_map_df, sources, output_dir)
        
    print(f"\n====== 数据集 {dataset_name} 处理完毕 ======")

except FileNotFoundError as e:
    print(f"[严重错误] 依赖文件未找到: {e}")
except Exception as e:
    print(f"[严重错误] 处理过程中发生未知异常: {e}")



from pythonProject.src.Structure_first.proxy_sample import multi_predicate_evaluation
from pythonProject.src.Structure_first.proxy_sample import evaluate_graph_only_baseline
# ===========================
dataset_to_process = 'dataset_test2'
# multi_predicate_evaluation(dataset_to_process)
multi_predicate_evaluation(dataset_name=dataset_to_process, run_times=20)