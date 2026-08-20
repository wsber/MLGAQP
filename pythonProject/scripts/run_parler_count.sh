#!/usr/bin/env bash
# ==============================================================================
# 脚本名称: run_parler_count.sh
# 作用: Parler 聚合模式 (Step 1 权重估计 + Step 2 POSSA 采样)
# ==============================================================================

# 1. 基础路径与环境配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_EXEC=$(command -v python)
if [ -z "$PYTHON_EXEC" ]; then
    echo "❌ 错误: 未找到 Python，请确保执行脚本前已激活虚拟环境！"
    exit 1
fi

set -e

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

if [ -n "$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
fi

# 2. 实验参数配置
PARENT_DATASET="parler"
DATASET_ID=0                   # 0: parler
SAMPLE_BUDGET=60000
RUN_TIMES=5
MAX_WORKERS=16
TARGET_TICKS="0.1"

RUNNER_STEP1="${PROJECT_ROOT}/pythonProject/src/runner/Projection_Sampling_and_Weight_Estimation_Runner.py"
RUNNER_STEP2="${PROJECT_ROOT}/pythonProject/src/runner/Proxy_Guided_Stratified_Importance_Sampling_Runner.py"

echo "=============================================================================="
echo "🚀 开始 Parler [COUNT 模式] 一键评估"
echo "工作目录: ${PROJECT_ROOT}"
echo "=============================================================================="

# STEP 1
echo -e "\n[Step 1/2] 正在调用 C++ 引擎估计 COUNT 投影空间权重..."
"$PYTHON_EXEC" "${RUNNER_STEP1}" \
    --run_cpp \
    --base_dir "${PROJECT_ROOT}" \
    --dataset "${PARENT_DATASET}" \
    --sample_budget ${SAMPLE_BUDGET} \
    --agg_func count \
    --table1 post \
    --table2 comment
echo -e "✅ [Step 1/2] COUNT 投影权重物化已完成！"

# STEP 2
echo -e "\n[Step 2/2] 正在执行 COUNT 模式的分层重要性采样实验 (RQ1 & RQ2)..."
"$PYTHON_EXEC" "${RUNNER_STEP2}" \
    -d ${DATASET_ID} \
    --agg_mode count \
    --target_ticks "${TARGET_TICKS}" \
    --run_times ${RUN_TIMES} \
    --max_workers ${MAX_WORKERS} \
    --base_dir "${PROJECT_ROOT}"
echo -e "✅ [Step 2/2] COUNT 分层采样评估已全部完成！"

OUTPUT_FILE="${PROJECT_ROOT}/datasets/${PARENT_DATASET}/results/efficiency/allocation_strategy_comparison_count.csv"
echo -e "\n=============================================================================="
echo "🎉 COUNT 流程全部执行完毕！"
echo "👉 结果: ${OUTPUT_FILE}"
echo "=============================================================================="