import os
import sys
import time

# 确保项目根目录在 sys.path 中以加载 FastestRunner
project_root = "/home/wangshuo/projects/Neo4j_Exp"
if project_root not in sys.path:
    sys.path.append(project_root)

from pythonProject.src.Structure_first.graph_sample import FastestRunner

# ==========================================
# 核心配置区：设置你本次想运行的聚合模式 ("count" 或 "sum")
# ==========================================
agg_mode = "count"  # 可选值为: "count" 或 "sum"
# ==========================================

# 三个数据集的完整执行参数配置字典
# 为应对 dataset_test 在 count 和 sum 模式下参数差异较大的情况，本配置支持分模式重写
WORKLOADS_CONFIG = {
    "dataset_three": {
        "parent_dir": "parler_data",
        "root_label": 1,
        "sample_budget": 60000,
        "post_oracle_col": "ML1_oracle2_probability",
        "comment_oracle_col": "ML2_oracle2_probability",
        "multi_proxy_prob": "ML1_proxy4b_probability",
        "runs": 5,
        "oracle_table1": None,  # 未指定则不添加该参数
        "oracle_table2": None,
        "sum_table": "post",
        "sum_col": "upvotes",
        "sum_label": "1"
    },
    "dataset_test": {
        "parent_dir": "parler_data",
        # count 模式独立重写参数（对应你命令中 budget=20000，以及有 oracle_table1/2 的情况）
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
        # sum 模式独立重写参数
        "sum": {
            "root_label": 2,
            "sample_budget": 60000,
            "oracle_table1": None,
            "oracle_table2": None,
            "post_oracle_col": "ML1_oracle2_probability",
            "comment_oracle_col": "ML2_oracle2_probability",
            "multi_proxy_prob": "ML1_proxy4b_probability",
            "sum_table": "post",
            "sum_col": "upvotes",
            "sum_label": "2",
            "runs": 2
        }
    },
    "amazon_extend": {
        "parent_dir": "amazon_data",
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

def build_extra_args(dataset: str, cfg: dict, mode: str) -> list:
    """根据配置动态拼装 C++ 二进制所需的命令行参数"""
    parent_dir = cfg["parent_dir"]
    
    # 动态组装路径
    budget_curve_in = f"/home/wangshuo/resource/datasets/{parent_dir}/{dataset}/results/efficiency/allocation_strategy_comparison_{mode}.csv"
    budget_curve_out = f"/home/wangshuo/resource/datasets/{parent_dir}/{dataset}/results/efficiency/FastestO_budget_curve_{mode}.csv"
    
    args = [
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
    
    # 如果指定了实体表，动态追加
    if cfg.get("oracle_table1"):
        args += ["--ORACLE_TABLE1", cfg["oracle_table1"]]
    if cfg.get("oracle_table2"):
        args += ["--ORACLE_TABLE2", cfg["oracle_table2"]]
        
    # 如果是 sum 模式，动态追加聚合求和专用字段
    if mode == "sum":
        args += [
            "--SUM_TABLE", cfg["sum_table"],
            "--SUM_COL", cfg["sum_col"],
            "--SUM_LABEL", cfg["sum_label"]
        ]
        
    return args

def run_batch_experiments(mode: str):
    """顺序执行多数据集实验"""
    print("=" * 70)
    print(f" 开始执行批量实验 | 聚合模式: {mode.upper()} ")
    print("=" * 70)
    
    runner = FastestRunner(build_dir="/home/wangshuo/projects/FaSTest-main/build")
    start_time_all = time.time()
    
    for dataset, config_item in WORKLOADS_CONFIG.items():
        print(f"\n[任务启动] 正在运行数据集: {dataset} ...")
        
        # 1. 解析参数：如果该数据集下有针对当前 mode 的局部重写配置，则进行合并
        active_config = config_item.copy()
        if mode in config_item:
            active_config.update(config_item[mode])
            
        root_label = active_config["root_label"]
        budget = active_config["sample_budget"]
        
        # 2. 动态编译额外命令行参数
        extra_args = build_extra_args(dataset, active_config, mode)
        
        # 打印即将运行的等效 Shell 命令，便于排查
        print(f"等效命令:\n/home/wangshuo/projects/FaSTest-main/build/Fastest -d {dataset} --ROOT_LABEL {root_label} --SAMPLE_BUDGET {budget} " + " ".join(extra_args))
        
        # 3. 运行 C++ 任务
        task_start_time = time.time()
        try:
            code, output = runner.run(
                dataset=dataset,
                root_label=root_label,
                sample_budget=budget,
                extra_args=extra_args
            )
            
            task_duration = time.time() - task_start_time
            if code == 0:
                print(f"[成功] 数据集 {dataset} 执行完毕，耗时: {task_duration:.2f} 秒。")
            else:
                print(f"[失败] 数据集 {dataset} 运行异常，退出码: {code}")
                # 打印部分输出便于定位问题
                print(f"错误输出:\n{output[-500:] if output else '无输出'}")
                
        except Exception as e:
            print(f"[异常] 运行数据集 {dataset} 时发生未捕获异常: {e}")
            
    print("\n" + "=" * 70)
    print(f" 批量实验运行结束 | 总耗时: {time.time() - start_time_all:.2f} 秒 ")
    print("=" * 70)

if __name__ == '__main__':
    # 检验模式参数
    if agg_mode not in ["count", "sum"]:
        print(f"[错误] 未知的聚合模式: {agg_mode}，仅支持 'count' 或 'sum'")
        sys.exit(1)
        
    run_batch_experiments(agg_mode)