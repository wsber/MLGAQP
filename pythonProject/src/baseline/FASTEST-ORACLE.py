import os
import sys
import time
import argparse
import concurrent.futures

# 动态定位项目根目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from pythonProject.src.Structure_first.graph_sample import FastestRunner

# ==============================================================================
# 数据集配置清单 (支持全局通用参数 + count/sum 模式独立覆盖)
# ==============================================================================
WORKLOADS_CONFIG = {
    "parler": {
        "dataset_name": "parler",
        "root_label": 1,
        "sample_budget": 60000,
        "post_oracle_col": "ML1_oracle2_probability",
        "comment_oracle_col": "ML2_oracle2_probability",
        "multi_proxy_prob": "ML1_proxy4b_probability",
        "runs": 5,
        "oracle_table1": "post",
        "oracle_table2": "comment",
        "sum_table": "post",
        "sum_col": "upvotes",
        "sum_label": "1"
    },
    "parler-E": {
        "dataset_name": "parler-E",
        "count": {
            "root_label": 1,
            "sample_budget": 20000,
            "oracle_table1": "post",
            "oracle_table2": "comment",
            "post_oracle_col": "ML1_oracle2_probability",
            "comment_oracle_col": "ML2_oracle2_probability",
            "multi_proxy_prob": "ML1_proxy4b_probability",
            "runs": 5
        },
        "sum": {
            "root_label": 2,
            "sample_budget": 60000,
            "oracle_table1": "post",
            "oracle_table2": "comment",
            "post_oracle_col": "ML1_oracle2_probability",
            "comment_oracle_col": "ML2_oracle2_probability",
            "multi_proxy_prob": "ML1_proxy4b_probability",
            "sum_table": "post",
            "sum_col": "upvotes",
            "sum_label": "2",
            "runs": 5
        }
    },
    "amazon": {
        "dataset_name": "amazon",
        "root_label": 1,
        "sample_budget": 60000,
        "oracle_table1": "product",
        "oracle_table2": "review",
        "post_oracle_col": "ML3_oracle2_probability",
        "comment_oracle_col": "ML2_oracle1_probability",
        "multi_proxy_prob": "ML3_proxy2_probability",
        "runs": 5,
        "sum_table": "product",
        "sum_col": "price",
        "sum_label": "12"
    }
}

def build_extra_args(base_dir: str, dataset: str, cfg: dict, mode: str) -> list:
    """根据配置动态拼装 C++ 二进制参数"""
    dataset_path = os.path.join(base_dir, "datasets", dataset)
    
    # 动态组装输入输出路径
    budget_curve_in = os.path.join(dataset_path, "results", "efficiency", f"allocation_strategy_comparison_{mode}.csv")
    budget_curve_out = os.path.join(dataset_path, "results", "efficiency", f"FastestO_budget_curve_{mode}.csv")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(budget_curve_out), exist_ok=True)
    
    args = [
        "--BASE_DIR", os.path.join(base_dir, "datasets"),
        "--ESTIMATE_WITH_PREDICATE",
        "--POST_ORACLE_COL", cfg["post_oracle_col"],
        "--COMMENT_ORACLE_COL", cfg["comment_oracle_col"],
        "--AGG_FUNC", mode,
        "--MULTI_PROXY_PROB", cfg["multi_proxy_prob"],
        "--BUDGET_CURVE_IN", budget_curve_in,
        "--FASTESTO_BUDGET_CURVE",
        "--FASTESTO_RUNS", str(cfg["runs"]),
        "--FASTESTO_BUDGET_CURVE_OUT", budget_curve_out
    ]
    
    if cfg.get("oracle_table1"):
        args += ["--ORACLE_TABLE1", cfg["oracle_table1"]]
    if cfg.get("oracle_table2"):
        args += ["--ORACLE_TABLE2", cfg["oracle_table2"]]
        
    if mode == "sum":
        args += [
            "--SUM_TABLE", cfg["sum_table"],
            "--SUM_COL", cfg["sum_col"],
            "--SUM_LABEL", str(cfg["sum_label"])
        ]
        
    return args

def run_single_dataset_task(task_tuple):
    """单个 (dataset, mode) 任务执行函数 (供多进程调用)"""
    dataset_key, mode, base_dir, build_dir = task_tuple
    config_item = WORKLOADS_CONFIG[dataset_key]
    
    # 合并局部模式参数
    active_config = config_item.copy()
    if mode in config_item:
        active_config.update(config_item[mode])
        
    dataset = active_config["dataset_name"]
    root_label = active_config["root_label"]
    budget = active_config["sample_budget"]
    
    extra_args = build_extra_args(base_dir, dataset, active_config, mode)
    
    print(f"\n[任务启动] 数据集: {dataset} | 模式: {mode.upper()}")
    runner = FastestRunner(build_dir=build_dir)
    
    start_time = time.time()
    code, output = runner.run(
        dataset=dataset,
        root_label=root_label,
        sample_budget=budget,
        extra_args=extra_args
    )
    duration = time.time() - start_time
    
    if code == 0:
        print(f"✅ [成功] {dataset} ({mode.upper()}) 执行完毕，耗时: {duration:.2f} 秒。")
        return True, dataset, mode, duration
    else:
        print(f"❌ [失败] {dataset} ({mode.upper()}) 异常退出，错误码: {code}")
        return False, dataset, mode, duration

def main():
    parser = argparse.ArgumentParser(description="FaSTest-Oracle Multi-Dataset Runner")
    parser.add_argument("--mode", choices=["count", "sum", "all"], default="all", help="聚合模式")
    parser.add_argument("--dataset", choices=["parler", "parler-E", "amazon", "all"], default="all", help="目标数据集")
    parser.add_argument("--parallel", action="store_true", help="是否开启多进程并发执行")
    parser.add_argument("--workers", type=int, default=3, help="最大并发进程数")
    parser.add_argument("--base_dir", default=PROJECT_ROOT, help="项目根目录")
    parser.add_argument("--build_dir", default=os.path.join(PROJECT_ROOT, "cProject", "build"), help="C++ 可执行程序目录")
    args = parser.parse_args()

    # 确定要跑的 modes 和 datasets
    modes = ["count", "sum"] if args.mode == "all" else [args.mode]
    datasets = list(WORKLOADS_CONFIG.keys()) if args.dataset == "all" else [args.dataset]
    
    # 生成任务列表
    task_list = []
    for d in datasets:
        for m in modes:
            task_list.append((d, m, args.base_dir, args.build_dir))

    print("=" * 70)
    print(f" 🚀 FaSTest-Oracle 批量实验启动 | 总任务数: {len(task_list)} | 并发模式: {args.parallel}")
    print("=" * 70)

    total_start = time.time()

    if args.parallel and len(task_list) > 1:
        # 并发执行模式
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(run_single_dataset_task, task_list))
    else:
        # 顺序执行模式
        results = [run_single_dataset_task(t) for t in task_list]

    print("\n" + "=" * 70)
    print(f" 🎉 全部实验完成 | 总耗时: {time.time() - total_start:.2f} 秒")
    print("=" * 70)

if __name__ == "__main__":
    main()