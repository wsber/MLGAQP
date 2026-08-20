#!/usr/bin/env bash
# ==============================================================================
# 脚本名称: run_parler_sum.sh
# 作用: Parler 数据集 SUM 聚合模式 (Step 1 权重估计 + Step 2 POSSA 采样)
# ==============================================================================

# 1. 基础路径与环境配置
# ==============================================================================
PYTHON_EXEC="/home/wangshuo/software/anaconda3/envs/proxy/bin/python"
source /home/wangshuo/software/anaconda3/etc/profile.d/conda.sh
conda activate proxy

set -e  # 遇到任何错误立即终止脚本执行

PROJECT_ROOT="/home/wangshuo/projects/PROXY"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# 自动配置 Conda 的 C++ 动态库路径 (防止 GLIBCXX 报错)
if [ -n "$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
fi

# 2. 实验参数配置
PARENT_DATASET="parler"
DATASET_ID=0                   # 0: parler, 1: parler-e, 2: amazon
SAMPLE_BUDGET=60000            # C++ 树采样预算
RUN_TIMES=5                    # 每个采样率的重复轮数
MAX_WORKERS=16                 # 并行进程数
# TARGET_TICKS="0.01,0.05,0.075,0.1,0.125,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8"
TARGET_TICKS="0.1"

# === SUM 聚合专属配置 ===
SUM_TABLE="post"               # 聚合目标表
SUM_COL="upvotes"              # 聚合数值列名 (帖子点赞数)
SUM_LABEL=1                    # Parler 中被聚合的 Post 节点标签 (Parler 为 1, Parler-E 为 2)
TABLE1="post"
TABLE2="comment"

# 脚本路径
RUNNER_STEP1="${PROJECT_ROOT}/pythonProject/src/runner/Projection_Sampling_and_Weight_Estimation_Runner.py"
RUNNER_STEP2="${PROJECT_ROOT}/pythonProject/src/runner/Proxy_Guided_Stratified_Importance_Sampling_Runner.py"

echo "=============================================================================="
echo "🚀 开始 Parler [SUM 模式] 一键评估"
echo "工作目录: ${PROJECT_ROOT}"
echo "目标数据集: ${PARENT_DATASET} (ID: ${DATASET_ID})"
echo "聚合配置: 表=${SUM_TABLE}, 列=${SUM_COL}, Label=${SUM_LABEL}"
echo "采样率梯度: ${TARGET_TICKS}"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# STEP 1: 语义投影采样与权重物化 (SUM 模式)
# ------------------------------------------------------------------------------
echo -e "\n[Step 1/2] 正在调用 C++ 引擎估计 SUM 投影空间权重并物化..."

"$PYTHON_EXEC" "${RUNNER_STEP1}" \
    --run_cpp \
    --base_dir "${PROJECT_ROOT}" \
    --dataset "${PARENT_DATASET}" \
    --sample_budget ${SAMPLE_BUDGET} \
    --agg_func sum \
    --sum_table "${SUM_TABLE}" \
    --sum_col "${SUM_COL}" \
    --sum_label ${SUM_LABEL} \
    --table1 "${TABLE1}" \
    --table2 "${TABLE2}" \
    --workers ${MAX_WORKERS}

echo -e "✅ [Step 1/2] SUM 投影权重物化已完成 (保存在 aggregated_results_sum/)！"

# ------------------------------------------------------------------------------
# STEP 2: 代理引导的分层重要性采样 (SUM 模式)
# ------------------------------------------------------------------------------
echo -e "\n[Step 2/2] 正在执行 SUM 模式的分层重要性采样实验 (RQ1 & RQ2)..."

"$PYTHON_EXEC" "${RUNNER_STEP2}" \
    -d ${DATASET_ID} \
    --agg_mode sum \
    --target_ticks "${TARGET_TICKS}" \
    --run_times ${RUN_TIMES} \
    --max_workers ${MAX_WORKERS} \
    --base_dir "${PROJECT_ROOT}"

echo -e "✅ [Step 2/2] SUM 分层采样评估已全部完成！"

# ------------------------------------------------------------------------------
# 结果汇总提示
# ------------------------------------------------------------------------------
OUTPUT_FILE="${PROJECT_ROOT}/datasets/${PARENT_DATASET}/results/efficiency/allocation_strategy_comparison_sum.csv"
echo -e "\n=============================================================================="
echo "🎉 SUM 流程全部执行完毕！"
echo "实验结果已保存至:"
echo "👉 ${OUTPUT_FILE}"
echo "=============================================================================="