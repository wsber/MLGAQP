#!/usr/bin/env bash
# ==============================================================================
# 脚本名称: run_amazon_sum.sh
# 作用: 仅运行 Amazon 的 SUM 模式 (Step 1 权重物化 + Step 2 POSSA 采样)
# ==============================================================================

# 1. 基础路径与环境配置
PYTHON_EXEC="/home/wangshuo/software/anaconda3/envs/proxy/bin/python"
source /home/wangshuo/software/anaconda3/etc/profile.d/conda.sh
conda activate proxy

set -e  # 遇到错误立即终止

PROJECT_ROOT="/home/wangshuo/projects/PROXY"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

if [ -n "$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
fi

# 2. 实验参数配置
DATASET_NAME="amazon"          # Amazon 数据集目录名 (若目录名为 amazon_extend 则改为 amazon_extend)
DATASET_ID=2                   # Proxy_Guided_Runner 中 2 对应 Amazon
SAMPLE_BUDGET=60000            # C++ 结构采样预算
RUN_TIMES=5                    # 每个采样率重复 5 轮
MAX_WORKERS=16                 # 并发进程数
# TARGET_TICKS="0.01,0.05,0.075,0.1,0.125,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8"
TARGET_TICKS="0.1"

# === Amazon SUM 聚合专属配置 ===
SUM_TABLE="product"            # 聚合目标表 (product.csv)
SUM_COL="price"                # 聚合数值列 (商品价格)
SUM_LABEL=12                   # Amazon 查询图中被聚合的 Product 节点标签 (Label 12)
TABLE1="product"
TABLE2="review"

# 脚本路径
RUNNER_STEP1="${PROJECT_ROOT}/pythonProject/src/runner/Projection_Sampling_and_Weight_Estimation_Runner.py"
RUNNER_STEP2="${PROJECT_ROOT}/pythonProject/src/runner/Proxy_Guided_Stratified_Importance_Sampling_Runner.py"

echo "=============================================================================="
echo "🚀 开始 Amazon [SUM 模式] 一键评估"
echo "工作目录: ${PROJECT_ROOT}"
echo "目标数据集: ${DATASET_NAME} (ID: ${DATASET_ID})"
echo "聚合配置: 表=${SUM_TABLE}, 列=${SUM_COL}, Label=${SUM_LABEL}"
echo "采样率梯度: ${TARGET_TICKS}"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# STEP 1: 语义投影采样与权重物化 (SUM 模式)
# ------------------------------------------------------------------------------
echo -e "\n[Step 1/2] 正在调用 C++ 引擎估计 SUM 投影空间权重并生成 aggregated 文件..."

"$PYTHON_EXEC" "${RUNNER_STEP1}" \
    --base_dir "${PROJECT_ROOT}" \
    --dataset "${DATASET_NAME}" \
    --sample_budget ${SAMPLE_BUDGET} \
    --agg_func sum \
    --sum_table "${SUM_TABLE}" \
    --sum_col "${SUM_COL}" \
    --sum_label ${SUM_LABEL} \
    --table1 "${TABLE1}" \
    --table2 "${TABLE2}" \
    --workers ${MAX_WORKERS} \
    --run_cpp

echo -e "✅ [Step 1/2] SUM 投影权重物化已完成 (保存在 aggregated_results_sum/)！"

# ------------------------------------------------------------------------------
# STEP 2: 代理引导的分层重要性采样 (SUM 模式)
# ------------------------------------------------------------------------------
echo -e "\n[Step 2/2] 正在执行 SUM 模式的分层重要性采样实验 (POSSA)..."

"$PYTHON_EXEC" "${RUNNER_STEP2}" \
    -d ${DATASET_ID} \
    --agg_mode sum \
    --target_ticks "${TARGET_TICKS}" \
    --run_times ${RUN_TIMES} \
    --max_workers ${MAX_WORKERS} \
    --base_dir "${PROJECT_ROOT}"

echo -e "✅ [Step 2/2] SUM 分层采样评估已全部完成！"

# ------------------------------------------------------------------------------
# 结果提示
# ------------------------------------------------------------------------------
OUTPUT_FILE="${PROJECT_ROOT}/datasets/${DATASET_NAME}/results/efficiency/allocation_strategy_comparison_sum.csv"
echo -e "\n=============================================================================="
echo "🎉 Amazon [SUM 模式] 全部执行完毕！"
echo "实验结果已保存至:"
echo "👉 ${OUTPUT_FILE}"
echo "=============================================================================="