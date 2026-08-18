import os
import sys
import argparse
from typing import List, Dict
import pandas as pd
import numpy as np
import concurrent.futures
from tqdm import tqdm

# 动态添加项目根目录，避免硬编码导入失败
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pythonProject.src.Structure_first.graph_sample import FastestRunner

# =========================================================================
# 全局只读变量 (利用 Linux 的 Fork Copy-on-Write 机制，实现零开销内存共享)
# =========================================================================
GLOBAL_T1_DATA = None
GLOBAL_T2_DATA = None
EXPECTED_ML1_COLS = [
    'ML1_oracle1_probability', 'ML1_oracle2_probability', 
    'ML1_proxy1b_probability', 'ML1_proxy2b_probability',
    'ML1_proxy4b_probability', 'ML1_proxy6b_probability',
    'ML3_oracle1_probability', 'ML3_proxy1_probability',
    'ML3_oracle2_probability', 'ML3_proxy3_probability', 'ML3_proxy2_probability'
]
EXPECTED_ML2_COLS = [
    'ML2_oracle1_probability', 'ML2_oracle2_probability', 
    'ML2_proxy2d2_probability', 'ML2_proxy4d2_probability', 
    'ML2_proxy1_probability', 'ML2_proxy2_probability', 'ML2_proxy4_probability'
]

# =========================================================================
# 1. 核心数据处理函数
# =========================================================================

def split_results_by_query(input_file_path: str, output_dir: str):
    print(f"\n[*] 正在拆分多查询结果文件: {input_file_path}")
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"[错误] 输入文件不存在: {input_file_path}")

    os.makedirs(output_dir, exist_ok=True)

    try:
        # 使用 pyarrow 引擎加速读取 (如果已安装 pyarrow)
        try:
            df = pd.read_csv(input_file_path, engine='pyarrow')
        except ValueError:
            df = pd.read_csv(input_file_path)
            
        print(f"[+] 成功读取 {len(df)} 行实例数据。")

        if 'query_name' not in df.columns:
            raise ValueError("[错误] CSV 缺少 'query_name' 列。")

        # 优化: 避免 drop 的重复分配开销
        grouped = df.groupby('query_name')
        num_files_created = 0
        cols_to_keep = [c for c in df.columns if c != 'query_name']
        
        for query_name, group_df in grouped:
            base_name = os.path.splitext(query_name)[0]
            output_filepath = os.path.join(output_dir, f"{base_name}.csv")
            group_df[cols_to_keep].to_csv(output_filepath, index=False)
            num_files_created += 1

        print(f"[✓] 拆分完成，共生成 {num_files_created} 个文件。")
    except Exception as e:
        print(f"[严重错误] 拆分文件发生异常: {e}")
        raise e

def load_and_prepare_globals(id_mapping_path: str, t1_csv_path: str, t2_csv_path: str, table1: str, table2: str):
    """
    [核心优化] 在主进程中预先完成 ID映射 与 实体数据 的连接操作。
    生成两个全局 DataFrame，直接按 node_id 索引，子进程只需要做极小的 Inner Join。
    """
    global GLOBAL_T1_DATA, GLOBAL_T2_DATA
    
    print(f"\n[*] 正在预计算全局数据映射 (极大提升并发速度)...")
    id_map_df = pd.read_csv(id_mapping_path, dtype={'internal_id': str, 'orig_id': str, 'type': str})
    id_map_df.rename(columns={'internal_id': 'node_id'}, inplace=True)
    id_map_df['type'] = id_map_df['type'].str.lower()
    
    # 提取 T1 和 T2 的纯映射
    t1_map = id_map_df[id_map_df['type'] == table1.lower()][['node_id', 'orig_id']]
    t2_map = id_map_df[id_map_df['type'] == table2.lower()][['node_id', 'orig_id']]
    
    # 读取原始数据
    try:
        t1_df = pd.read_csv(t1_csv_path, dtype=str, engine='pyarrow')
        t2_df = pd.read_csv(t2_csv_path, dtype=str, engine='pyarrow')
    except:
        t1_df = pd.read_csv(t1_csv_path, dtype=str)
        t2_df = pd.read_csv(t2_csv_path, dtype=str)

    if 'id:ID' in t1_df.columns: t1_df.rename(columns={'id:ID': 'orig_id'}, inplace=True)
    if 'id:ID' in t2_df.columns: t2_df.rename(columns={'id:ID': 'orig_id'}, inplace=True)

    # 仅保留需要的 ML 列，节约内存
    t1_cols_to_keep = ['orig_id'] + [c for c in EXPECTED_ML1_COLS if c in t1_df.columns]
    t2_cols_to_keep = ['orig_id'] + [c for c in EXPECTED_ML2_COLS if c in t2_df.columns]

    # 执行全局 Join
    GLOBAL_T1_DATA = pd.merge(t1_map, t1_df[t1_cols_to_keep], on='orig_id', how='inner')
    GLOBAL_T2_DATA = pd.merge(t2_map, t2_df[t2_cols_to_keep], on='orig_id', how='inner')
    
    print(f"[+] 预计算完成: {table1} 表预置 {len(GLOBAL_T1_DATA)} 行, {table2} 表预置 {len(GLOBAL_T2_DATA)} 行。")


def process_single_query_file_fast(query_file: str, estimate_dir: str, output_dir: str):
    """
    【极速版子进程执行单元】只做最基础的 Inner Join 和聚合。
    """
    query_file_path = os.path.join(estimate_dir, query_file)
    query_basename = os.path.basename(query_file_path).replace(".csv", "").replace("_estimateW_result", "")
    
    try:
        instance_df = pd.read_csv(query_file_path, dtype={'node_id': str}, engine='pyarrow')
    except:
        instance_df = pd.read_csv(query_file_path, dtype={'node_id': str})
        
    if instance_df.empty:
        return False, f"文件为空: {query_basename}"

    # 提取基础权重结构 (去重)
    base_agg_df = instance_df[['instance_id', 'estimateW', 'global_estimateW']].drop_duplicates(subset=['instance_id'])
    base_agg_df.set_index('instance_id', inplace=True)
    
    ins_nodes = instance_df[['instance_id', 'node_id']]

    # --- Table 1 极速聚合 ---
    # 利用 Inner Join 同时完成筛选和数据连接，省去 groupby 后对全表过滤
    t1_joined = pd.merge(ins_nodes, GLOBAL_T1_DATA, on='node_id', how='inner')
    if not t1_joined.empty:
        t1_joined.drop(columns=['node_id'], inplace=True)
        agg_t1 = t1_joined.groupby('instance_id').agg(list)
        agg_t1.rename(columns={'orig_id': 'post_id_list'}, inplace=True)
    else:
        agg_t1 = pd.DataFrame()

    # --- Table 2 极速聚合 ---
    t2_joined = pd.merge(ins_nodes, GLOBAL_T2_DATA, on='node_id', how='inner')
    if not t2_joined.empty:
        t2_joined.drop(columns=['node_id'], inplace=True)
        agg_t2 = t2_joined.groupby('instance_id').agg(list)
        agg_t2.rename(columns={'orig_id': 'comment_id_list'}, inplace=True)
    else:
        agg_t2 = pd.DataFrame()

    # --- 合并结果 ---
    # 使用 join 比 merge(on=...) 更快，因为已经设置了 instance_id 为 index
    final_df = base_agg_df.join(agg_t1, how='left').join(agg_t2, how='left').reset_index()

    # 填充确保所有期待的列都存在（如果缺失，设为 NaN 列表或直接 NaN，符合原意图即可）
    final_columns_order = [
        'instance_id', 'estimateW', 'global_estimateW', 
        'post_id_list', 'comment_id_list'
    ] + EXPECTED_ML1_COLS + EXPECTED_ML2_COLS
    
    # 补齐缺失列
    for col in final_columns_order:
        if col not in final_df.columns:
            final_df[col] = np.nan
            
    final_df = final_df[final_columns_order]
    
    output_filepath = os.path.join(output_dir, f"aggregated_list_{query_basename}.csv")
    final_df.to_csv(output_filepath, index=False)
    
    return True, None


# =========================================================================
# 2. 全流程主控制流水线
# =========================================================================

def run_pipeline(
    base_dir: str, dataset: str, sample_budget: int = 60000,
    agg_func: str = "count", sum_table: str = "product", sum_col: str = "price",
    sum_label: int = 12, table1: str = "product", table2: str = "review",
    c_build_dir: str = None, run_cpp: bool = False, workers: int = 8
):
    print("=" * 70)
    print(f"🚀 开始执行核心实例抽取与聚合流水线 (极致提速版)")
    print(f"   • 项目根目录 : {base_dir}")
    print(f"   • 数据集名称 : {dataset}")
    print(f"   • 并发核心数 : {workers}")
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
        if c_build_dir is None: c_build_dir = os.path.join(base_dir, "cProject", "build")
        print(f"\n[Step 0] 启动 C++ FastestRunner...")
        runner = FastestRunner(build_dir=c_build_dir)
        extra_args = ["--BASE_DIR", os.path.join(base_dir, "datasets"), "--AGG_FUNC", agg_func]
        if agg_func.lower() == "sum":
            extra_args.extend(["--SUM_TABLE", sum_table, "--SUM_COL", sum_col, "--SUM_LABEL", str(sum_label)])
        code, output = runner.run(dataset=dataset, root_label=-1, sample_budget=sample_budget, extra_args=extra_args)
        if code != 0:
            print(f"[错误] C++ 端执行失败 (Code {code})"); return

    # [Step 1] 拆分结果文件
    split_results_by_query(raw_ins_csv_path, estimate_dir)

    # [Step 2] 预计算全局 Join 缓存 (消除进程内重复计算)
    load_and_prepare_globals(id_mapping_path, t1_csv_path, t2_csv_path, table1, table2)

    # [Step 3] 多进程并发聚合
    print(f"\n[Step 3] 启动多进程并发生成聚合列表 (Worker={workers})...")
    os.makedirs(aggregated_dir, exist_ok=True)
    
    query_files = [f for f in os.listdir(estimate_dir) if f.endswith('.csv')]
    if not query_files:
        print(f"[警告] 未找到 CSV 文件。"); return

    success_cnt = 0
    # 无需通过 args 传递复杂字典对象，完美避开序列化开销
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_single_query_file_fast, qf, estimate_dir, aggregated_dir): qf for qf in query_files}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(query_files), desc="Aggregating", ncols=100):
            try:
                success, err_msg = future.result()
                if success: success_cnt += 1
                else: tqdm.write(f"[警告] {err_msg}")
            except Exception as e:
                tqdm.write(f"[严重错误] 文件处理异常: {e}")

    print("\n" + "=" * 70)
    print(f"🎉 处理完毕！成功聚合 {success_cnt}/{len(query_files)} 个查询。输出至: {aggregated_dir}")
    print("=" * 70)


# =========================================================================
# 3. 命令行参数解析
# =========================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Projection Sampling Pipeline")
    parser.add_argument("--base_dir", type=str, default=PROJECT_ROOT)
    parser.add_argument("--dataset", type=str, default="parler")
    parser.add_argument("--sample_budget", type=int, default=60000)
    parser.add_argument("--agg_func", type=str, default="count", choices=["count", "sum"])
    parser.add_argument("--sum_table", type=str, default="product")
    parser.add_argument("--sum_col", type=str, default="price")
    parser.add_argument("--sum_label", type=int, default=12)
    parser.add_argument("--table1", type=str, default="product")
    parser.add_argument("--table2", type=str, default="review")
    parser.add_argument("--run_cpp", action="store_true")
    parser.add_argument("--c_build_dir", type=str, default=None)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run_pipeline(
        base_dir=args.base_dir, dataset=args.dataset, sample_budget=args.sample_budget,
        agg_func=args.agg_func, sum_table=args.sum_table, sum_col=args.sum_col,
        sum_label=args.sum_label, table1=args.table1, table2=args.table2,
        c_build_dir=args.c_build_dir, run_cpp=args.run_cpp, workers=args.workers
    )