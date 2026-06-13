
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


datasets_name = "datasets"
# datasets_name = "parler_data"
# dataset_name = "dataset_three"
dataset_name = "amazon"
# 原始CSV数据路径
CSV_BASE_DIR = f"/home/wangshuo/resource/datasets/{datasets_name}/{dataset_name}/csv_data"
# 转换后GraphLib数据存放路径
Graph_Lib_Dir = f"/home/wangshuo/resource/datasets/{datasets_name}/{dataset_name}/data_graph"

runner = FastestRunner(build_dir="/home/wangshuo/projects/FaSTest-main/build")
print(dataset_name)
sample_budget = 60000
extra_args = ["--AGG_FUNC","count","--SUM_TABLE","product","--SUM_COL","average_rating","--SUM_LABEL","12"]
# extra_args = ["--AGG_FUNC","sum","--SUM_TABLE","post","--SUM_COL","upvotes","--SUM_LABEL","1"]
# code, output = runner.run(dataset=dataset_name, root_label=-1, sample_budget=sample_budget, extra_args=extra_args)

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


base_path = f"/home/wangshuo/resource/datasets/{datasets_name}/{dataset_name}/results/"
input_csv_path = os.path.join(base_path, "ins_estimateW_result.csv")
output_directory = os.path.join(base_path, "structure_estimate/")

# 调用主函数
# split_results_by_query(input_csv_path, output_directory)


def load_and_prepare_mappings(id_mapping_path: str) -> pd.DataFrame:
    """读取id_mapping.csv并准备好用于连接的DataFrame。"""
    if not os.path.exists(id_mapping_path):
        raise FileNotFoundError(f"ID映射文件不存在: {id_mapping_path}")
        
    id_map_df = pd.read_csv(id_mapping_path, dtype={'internal_id': str, 'orig_id': str, 'type': str})
    id_map_df.rename(columns={'internal_id': 'node_id'}, inplace=True)
    
    # [修改点1]：为了防止大小写不一致 (Product vs product)，统一把类型转为小写
    if 'type' in id_map_df.columns:
        id_map_df['type'] = id_map_df['type'].str.lower()
        
    print(f"加载了 {len(id_map_df)} 条ID映射记录。")
    return id_map_df

def load_source_csvs(table1_csv_path: str, table2_csv_path: str, table1_name: str, table2_name: str) -> Dict[str, pd.DataFrame]:
    """读取两张实体数据表，并存储到以它们名字命名的字典中。"""
    if not os.path.exists(table1_csv_path):
        raise FileNotFoundError(f"{table1_name}.csv 文件不存在: {table1_csv_path}")
    if not os.path.exists(table2_csv_path):
        raise FileNotFoundError(f"{table2_name}.csv 文件不存在: {table2_csv_path}")
        
    t1_df = pd.read_csv(table1_csv_path, dtype=str)
    t2_df = pd.read_csv(table2_csv_path, dtype=str)
    
    # 统一把 id:ID 重命名为 orig_id
    if 'id:ID' in t1_df.columns:
        t1_df.rename(columns={'id:ID': 'orig_id'}, inplace=True)
    if 'id:ID' in t2_df.columns:
        t2_df.rename(columns={'id:ID': 'orig_id'}, inplace=True)
        
    print(f"加载了 {len(t1_df)} 行 {table1_name} 数据和 {len(t2_df)} 行 {table2_name} 数据。")
    return {table1_name: t1_df, table2_name: t2_df}


def process_single_query_file_correctly(
    query_file_path: str, 
    id_map_df: pd.DataFrame, 
    sources: Dict[str, pd.DataFrame], 
    output_dir: str,
    table1: str,
    table2: str
):
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
    expected_ml1_cols = ['ML1_oracle1_probability', 'ML1_oracle2_probability', 
                         'ML1_proxy1b_probability', 'ML1_proxy2b_probability',
                         'ML1_proxy4b_probability', 'ML1_proxy6b_probability','ML3_oracle1_probability','ML3_proxy1_probability',
                         'ML3_oracle2_probability','ML3_proxy3_probability','ML3_proxy2_probability']
    
    expected_ml2_cols = ['ML2_oracle1_probability','ML2_oracle2_probability', 'ML2_proxy2d2_probability', 
                         'ML2_proxy4d2_probability', 'ML2_proxy1_probability',
                         'ML2_proxy2_probability', 'ML2_proxy4_probability']
    
    # 3. 分别处理两组数据流 
    # [修改点2]：替换原来写死的 'Post' 和 'Comment'，根据传入的小写表名匹配
    t1_data = merged_with_map[merged_with_map['type'] == table1.lower()].copy()
    t1_joined = pd.merge(t1_data, sources[table1], on='orig_id', how='left')
    
    t2_data = merged_with_map[merged_with_map['type'] == table2.lower()].copy()
    t2_joined = pd.merge(t2_data, sources[table2], on='orig_id', how='left')
    
    # 4. 分别进行聚合
    
    # --- Table 1 数据聚合 ---
    actual_ml1_cols = [col for col in expected_ml1_cols if col in t1_joined.columns]
    if not t1_joined.empty:
        agg_dict = {col: list for col in actual_ml1_cols}
        agg_dict['orig_id'] = list
        agg_t1 = t1_joined.groupby('instance_id').agg(agg_dict).reset_index()
        # [修改点3]：即便表名变成了 product，我们也必须重命名为 post_id_list 用以短路伪装
        agg_t1.rename(columns={'orig_id': 'post_id_list'}, inplace=True) 
    else:
        agg_t1 = pd.DataFrame(columns=['instance_id', 'post_id_list'] + actual_ml1_cols)

    # --- Table 2 数据聚合 ---
    actual_ml2_cols = [col for col in expected_ml2_cols if col in t2_joined.columns]
    if not t2_joined.empty:
        agg_dict = {col: list for col in actual_ml2_cols}
        agg_dict['orig_id'] = list
        agg_t2 = t2_joined.groupby('instance_id').agg(agg_dict).reset_index()
        # 同理：即便这里是 review，为了接驳后面的 proxy_sample.py 也必须保留 comment_id_list
        agg_t2.rename(columns={'orig_id': 'comment_id_list'}, inplace=True)
    else:
        agg_t2 = pd.DataFrame(columns=['instance_id', 'comment_id_list'] + actual_ml2_cols)
        
    # 创建基础表
    base_agg_df = instance_df[['instance_id', 'estimateW', 'global_estimateW']].groupby('instance_id').first().reset_index()
    
    # --- 合并聚合后的结果 ---
    final_df = pd.merge(base_agg_df, agg_t1, on='instance_id', how='left')
    final_df = pd.merge(final_df, agg_t2, on='instance_id', how='left')

    print(f"聚合完成，生成 {len(final_df)} 条实例记录。")

    # 5. 保存结果
    output_filename = f"aggregated_list_{query_basename}.csv"
    output_filepath = os.path.join(output_dir, output_filename)
    
    # 更新最终列顺序
    final_columns_order = [
        'instance_id', 'estimateW', 'global_estimateW', 
        'post_id_list', 'comment_id_list' 
    ] + expected_ml1_cols + expected_ml2_cols
    
    final_df = final_df.reindex(columns=final_columns_order)
    final_df.to_csv(output_filepath, index=False)
    print(f"[完成] 结果已保存到: {output_filepath}")

def main(parent_dataset: str, dataset_name: str, table1: str, table2: str):
    """主执行函数"""
    print(f"====== 开始处理数据集: {dataset_name} ({table1} & {table2}) ======")
    
    # [修改点4]：路径支持动态化配置，不再将 'parler_data' 写死
    base_path = f"/home/wangshuo/resource/datasets/{parent_dataset}/{dataset_name}"
    estimate_dir = os.path.join(base_path, "results", "structure_estimate")
    id_mapping_path = os.path.join(base_path, "data_graph", "id_mapping.csv")
    
    # 动态匹配 CSV
    t1_csv_path = os.path.join(base_path, "csv_data", f"{table1}.csv")
    t2_csv_path = os.path.join(base_path, "csv_data", f"{table2}.csv")
    
    output_dir = os.path.join(base_path, "results", "aggregated_results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    try:
        # 获取基础信息
        id_map_df = load_and_prepare_mappings(id_mapping_path)
        sources = load_source_csvs(t1_csv_path, t2_csv_path, table1, table2)

        query_files = [f for f in os.listdir(estimate_dir) if f.endswith('.csv')]
        if not query_files:
            print(f"[警告] 在目录 {estimate_dir} 中没有找到任何 .csv 结果文件。")
            return
            
        for query_file in sorted(query_files):
            query_file_path = os.path.join(estimate_dir, query_file)
            
            process_single_query_file_correctly(
                query_file_path, id_map_df, sources, output_dir, table1, table2
            )
            
        print(f"\n====== 数据集 {dataset_name} 处理完毕 ======")

    except FileNotFoundError as e:
        print(f"[严重错误] 依赖文件未找到: {e}")
    except Exception as e:
        print(f"[严重错误] 处理过程中发生未知异常: {e}")

if __name__ == '__main__':
    # --- Jupyter Notebook 或脚本配置区 ---
    # 随时根据需要修改这里的内容
    cfg_parent_dataset = datasets_name      
    cfg_dataset_name   =  dataset_name   # 或 dataset_three等
    cfg_table1         = "product"          # 或 post   product
    cfg_table2         = "review"           # 或 comment  review
    
    main(cfg_parent_dataset, cfg_dataset_name, cfg_table1, cfg_table2)