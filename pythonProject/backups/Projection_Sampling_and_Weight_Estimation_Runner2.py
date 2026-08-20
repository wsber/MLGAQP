import os
import sys
import argparse
from typing import List, Dict
import pandas as pd
import numpy as np
import concurrent.futures
from functools import partial
from tqdm import tqdm

# 动态添加项目根目录，避免硬编码导入失败
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 向上寻找项目根目录 (Runner -> src -> pythonProject -> PROXY)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pythonProject.src.Structure_first.graph_sample import FastestRunner

# =========================================================================
# 1. 核心数据处理函数
# =========================================================================

def split_results_by_query(input_file_path: str, output_dir: str):
    """
    拆分 C++ 生成的多查询汇总结果文件 (ins_estimateW_result.csv) 为单查询独立 CSV 文件。
    """
    print(f"\n[*] 正在拆分多查询结果文件: {input_file_path}")

    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"[错误] 输入文件不存在: {input_file_path}")

    os.makedirs(output_dir, exist_ok=True)

    try:
        df = pd.read_csv(input_file_path)
        print(f"[+] 成功读取 {len(df)} 行实例数据。")

        if 'query_name' not in df.columns:
            raise ValueError("[错误] CSV 文件中缺少 'query_name' 列，请检查 C++ 输出格式。")

        grouped = df.groupby('query_name')
        num_files_created = 0
        
        for query_name, group_df in grouped:
            base_name = os.path.splitext(query_name)[0]
            output_filename = f"{base_name}.csv"
            output_filepath = os.path.join(output_dir, output_filename)

            # 剔除 query_name 列后保存
            output_df = group_df.drop('query_name', axis=1)
            output_df.to_csv(output_filepath, index=False)
            num_files_created += 1

        print(f"[✓] 拆分完成，共生成 {num_files_created} 个单查询 CSV 文件至: {output_dir}")

    except Exception as e:
        print(f"[严重错误] 拆分文件过程中发生异常: {e}")
        raise e


def load_and_prepare_mappings(id_mapping_path: str) -> pd.DataFrame:
    """读取 id_mapping.csv 并准备连接用的 DataFrame"""
    if not os.path.exists(id_mapping_path):
        raise FileNotFoundError(f"ID映射文件不存在: {id_mapping_path}")
        
    id_map_df = pd.read_csv(id_mapping_path, dtype={'internal_id': str, 'orig_id': str, 'type': str})
    id_map_df.rename(columns={'internal_id': 'node_id'}, inplace=True)
    
    # 统一将类型转为小写，防止大小写不一致 (如 Product vs product)
    if 'type' in id_map_df.columns:
        id_map_df['type'] = id_map_df['type'].str.lower()
        
    print(f"[+] 加载了 {len(id_map_df)} 条 ID 映射记录。")
    return id_map_df


def load_source_csvs(table1_csv_path: str, table2_csv_path: str, table1_name: str, table2_name: str) -> Dict[str, pd.DataFrame]:
    """读取实体数据表 (如 product.csv/review.csv 或 post.csv/comment.csv)"""
    if not os.path.exists(table1_csv_path):
        raise FileNotFoundError(f"{table1_name}.csv 文件不存在: {table1_csv_path}")
    if not os.path.exists(table2_csv_path):
        raise FileNotFoundError(f"{table2_name}.csv 文件不存在: {table2_csv_path}")
        
    t1_df = pd.read_csv(table1_csv_path, dtype=str)
    t2_df = pd.read_csv(table2_csv_path, dtype=str)
    
    # 统一将表头 id:ID 重命名为 orig_id
    if 'id:ID' in t1_df.columns:
        t1_df.rename(columns={'id:ID': 'orig_id'}, inplace=True)
    if 'id:ID' in t2_df.columns:
        t2_df.rename(columns={'id:ID': 'orig_id'}, inplace=True)
        
    print(f"[+] 加载了 {len(t1_df)} 行 {table1_name} 数据和 {len(t2_df)} 行 {table2_name} 数据。")
    return {table1_name: t1_df, table2_name: t2_df}


def process_single_query_file_correctly(
    query_file: str, 
    estimate_dir: str,
    id_map_df: pd.DataFrame, 
    sources: Dict[str, pd.DataFrame], 
    output_dir: str,
    table1: str,
    table2: str
):
    """
    【供多进程调用的核心函数】处理单个长格式查询文件，聚合 ML 概率值和节点 ID 列表。
    """
    query_file_path = os.path.join(estimate_dir, query_file)
    query_basename = os.path.basename(query_file_path).replace(".csv", "").replace("_estimateW_result", "")
    
    instance_df = pd.read_csv(query_file_path)
    if instance_df.empty:
        return False, f"文件为空: {query_basename}"
        
    instance_df['node_id'] = instance_df['node_id'].astype(str)
    
    # 1. 连接 ID 映射
    merged_with_map = pd.merge(instance_df, id_map_df, on='node_id', how='left')

    # 2. 定义期望提取的 ML 列
    expected_ml1_cols = [
        'ML1_oracle1_probability', 'ML1_oracle2_probability', 
        'ML1_proxy1b_probability', 'ML1_proxy2b_probability',
        'ML1_proxy4b_probability', 'ML1_proxy6b_probability',
        'ML3_oracle1_probability', 'ML3_proxy1_probability',
        'ML3_oracle2_probability', 'ML3_proxy3_probability', 'ML3_proxy2_probability'
    ]
    
    expected_ml2_cols = [
        'ML2_oracle1_probability', 'ML2_oracle2_probability', 
        'ML2_proxy2d2_probability', 'ML2_proxy4d2_probability', 
        'ML2_proxy1_probability', 'ML2_proxy2_probability', 'ML2_proxy4_probability'
    ]
    
    # 3. 分别连接两个实体表的数据
    t1_data = merged_with_map[merged_with_map['type'] == table1.lower()].copy()
    t1_joined = pd.merge(t1_data, sources[table1], on='orig_id', how='left')
    
    t2_data = merged_with_map[merged_with_map['type'] == table2.lower()].copy()
    t2_joined = pd.merge(t2_data, sources[table2], on='orig_id', how='left')
    
    # 4. 核心聚合逻辑
    # --- Table 1 聚合 ---
    actual_ml1_cols = [col for col in expected_ml1_cols if col in t1_joined.columns]
    if not t1_joined.empty:
        agg_dict = {col: list for col in actual_ml1_cols}
        agg_dict['orig_id'] = list
        agg_t1 = t1_joined.groupby('instance_id').agg(agg_dict).reset_index()
        agg_t1.rename(columns={'orig_id': 'post_id_list'}, inplace=True) 
    else:
        agg_t1 = pd.DataFrame(columns=['instance_id', 'post_id_list'] + actual_ml1_cols)

    # --- Table 2 聚合 ---
    actual_ml2_cols = [col for col in expected_ml2_cols if col in t2_joined.columns]
    if not t2_joined.empty:
        agg_dict = {col: list for col in actual_ml2_cols}
        agg_dict['orig_id'] = list
        agg_t2 = t2_joined.groupby('instance_id').agg(agg_dict).reset_index()
        agg_t2.rename(columns={'orig_id': 'comment_id_list'}, inplace=True)
    else:
        agg_t2 = pd.DataFrame(columns=['instance_id', 'comment_id_list'] + actual_ml2_cols)
        
    # 提取结构权重基础表
    base_agg_df = instance_df[['instance_id', 'estimateW', 'global_estimateW']].groupby('instance_id').first().reset_index()
    
    # 合并聚合结果
    final_df = pd.merge(base_agg_df, agg_t1, on='instance_id', how='left')
    final_df = pd.merge(final_df, agg_t2, on='instance_id', how='left')

    # 5. 标准化列顺序并保存
    output_filename = f"aggregated_list_{query_basename}.csv"
    output_filepath = os.path.join(output_dir, output_filename)
    
    final_columns_order = [
        'instance_id', 'estimateW', 'global_estimateW', 
        'post_id_list', 'comment_id_list' 
    ] + expected_ml1_cols + expected_ml2_cols
    
    final_df = final_df.reindex(columns=final_columns_order)
    final_df.to_csv(output_filepath, index=False)
    
    return True, None


# =========================================================================
# 2. 全流程主控制流水线 (Pipeline Entrypoint)
# =========================================================================

def run_pipeline(
    base_dir: str,
    dataset: str,
    sample_budget: int = 60000,
    agg_func: str = "count",
    sum_table: str = "product",
    sum_col: str = "price",
    sum_label: int = 12,
    table1: str = "product",
    table2: str = "review",
    c_build_dir: str = None,
    run_cpp: bool = False,
    workers: int = 8
):
    """
    全流程执行函数：可供命令行或 Jupyter Notebook 直接调用。
    """
    print("=" * 70)
    print(f"🚀 开始执行核心实例抽取与聚合流水线")
    print(f"   • 项目根目录 (base_dir) : {base_dir}")
    print(f"   • 数据集名称 (dataset)  : {dataset}")
    print(f"   • 聚合类型 (agg_func)   : {agg_func}")
    print(f"   • 并发核心数 (workers)  : {workers}")
    print("=" * 70)

    dataset_path = os.path.join(base_dir, "datasets", dataset)
    results_dir = os.path.join(dataset_path, "results")
    estimate_dir = os.path.join(results_dir, "structure_estimate")
    aggregated_dir = os.path.join(results_dir, "aggregated_results")
    
    id_mapping_path = os.path.join(dataset_path, "data_graph", "id_mapping.csv")
    t1_csv_path = os.path.join(dataset_path, "csv_data", f"{table1}.csv")
    t2_csv_path = os.path.join(dataset_path, "csv_data", f"{table2}.csv")
    raw_ins_csv_path = os.path.join(results_dir, "ins_estimateW_result.csv")

    # [Step 0] 触发 C++ 执行
    if run_cpp:
        if c_build_dir is None:
            c_build_dir = os.path.join(base_dir, "cProject", "build")
        print(f"\n[Step 0] 启动 C++ FastestRunner (Build: {c_build_dir})...")
        runner = FastestRunner(build_dir=c_build_dir)
        
        extra_args = ["--BASE_DIR", os.path.join(base_dir, "datasets"), "--AGG_FUNC", agg_func]
        if agg_func.lower() == "sum":
            extra_args.extend(["--SUM_TABLE", sum_table, "--SUM_COL", sum_col, "--SUM_LABEL", str(sum_label)])
            
        code, output = runner.run(dataset=dataset, root_label=-1, sample_budget=sample_budget, extra_args=extra_args)
        if code != 0:
            print(f"[错误] C++ 端执行失败 (Code {code})，输出日志:\n{output}")
            return

    # [Step 1] 拆分结果文件
    print(f"\n[Step 1] 拆分结构估计结果...")
    split_results_by_query(raw_ins_csv_path, estimate_dir)

    # [Step 2] 加载大表映射
    print(f"\n[Step 2] 加载 ID 映射和实体 CSV 数据...")
    id_map_df = load_and_prepare_mappings(id_mapping_path)
    sources = load_source_csvs(t1_csv_path, t2_csv_path, table1, table2)

    # [Step 3] 多进程并发聚合
    print(f"\n[Step 3] 启动多进程并发生成聚合列表 (Worker={workers})...")
    os.makedirs(aggregated_dir, exist_ok=True)
    
    query_files = [f for f in os.listdir(estimate_dir) if f.endswith('.csv')]
    if not query_files:
        print(f"[警告] 未在 {estimate_dir} 找到任何拆分后的 CSV 文件。")
        return
        
    # 构造 worker 函数（固定只读的 DataFrame 参数，借用 Linux 写时复制共享内存）
    worker_func = partial(
        process_single_query_file_correctly,
        estimate_dir=estimate_dir,
        id_map_df=id_map_df,
        sources=sources,
        output_dir=aggregated_dir,
        table1=table1,
        table2=table2
    )

    success_cnt = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker_func, qf): qf for qf in query_files}
        
        # tqdm 炫酷进度条
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(query_files), desc="Aggregating", ncols=100):
            try:
                success, err_msg = future.result()
                if success:
                    success_cnt += 1
                else:
                    tqdm.write(f"[警告] {err_msg}")
            except Exception as e:
                tqdm.write(f"[严重错误] 文件处理异常: {e}")

    print("\n" + "=" * 70)
    print(f"🎉 数据集 [{dataset}] 处理完毕！成功聚合 {success_cnt}/{len(query_files)} 个查询。")
    print(f"📁 最终聚合数据已保存至: {aggregated_dir}")
    print("=" * 70)


# =========================================================================
# 3. 命令行参数解析
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Projection Sampling and Weight Estimation Pipeline")
    
    parser.add_argument("--base_dir", type=str, default=PROJECT_ROOT, 
                        help="项目根目录路径")
    parser.add_argument("--dataset", type=str, default="parler", 
                        help="数据集名称 (例如 parler, dataset_three, amazon_extend)")
    parser.add_argument("--sample_budget", type=int, default=60000, 
                        help="结构采样预算 (默认: 60000)")
    
    parser.add_argument("--agg_func", type=str, default="count", choices=["count", "sum"], 
                        help="聚合类型: count 或 sum")
    parser.add_argument("--sum_table", type=str, default="product")
    parser.add_argument("--sum_col", type=str, default="price")
    parser.add_argument("--sum_label", type=int, default=12)
    
    parser.add_argument("--table1", type=str, default="product")
    parser.add_argument("--table2", type=str, default="review")
    
    # 修复 action='store_true' 逻辑
    parser.add_argument("--run_cpp", action="store_true", 
                        help="加上此 Flag 则触发 C++ FastestRunner 生成 csv。默认跳过。")
    
    parser.add_argument("--c_build_dir", type=str, default=None)
    
    # 新增并发线程数控制
    parser.add_argument("--workers", type=int, default=16, 
                        help="并行处理核心数 (默认 16，根据服务器 CPU 核心数调整)")
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    run_pipeline(
        base_dir=args.base_dir,
        dataset=args.dataset,
        sample_budget=args.sample_budget,
        agg_func=args.agg_func,
        sum_table=args.sum_table,
        sum_col=args.sum_col,
        sum_label=args.sum_label,
        table1=args.table1,
        table2=args.table2,
        c_build_dir=args.c_build_dir,
        run_cpp=args.run_cpp,
        workers=args.workers
    )