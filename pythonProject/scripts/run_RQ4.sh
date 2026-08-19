#!/usr/bin/env bash
# ==============================================================================
# 脚本名称: run_RQ4.sh
# 作用: RQ4 分配策略消融实验 (Ablation Study) 一键全并行执行脚本
#       1. 对比 5 种方法: UN, PO, WO, MAB, 8_POSSA
#       2. 全并发运行 Parler(0), Parler-E(1), Amazon(2) 的 COUNT 和 SUM
#       3. 输出文件: allocation_strategy_comparison_ablation_{count,sum}.csv
#       4. 日志统一重定向至 logs/rq4_*.log
# ==============================================================================

# 1. 基础环境与全局路径
# ==============================================================================
PYTHON_EXEC="/home/wangshuo/software/anaconda3/envs/proxy/bin/python"
source /home/wangshuo/software/anaconda3/etc/profile.d/conda.sh
conda activate proxy

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 自动解析真实项目根目录 (.../PROXY)
if [ -d "${SCRIPT_DIR}/../../pythonProject" ]; then
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
elif [ -d "${SCRIPT_DIR}/../pythonProject" ]; then
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    PROJECT_ROOT="${SCRIPT_DIR}"
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

if [ -n "$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
fi

LOG_DIR="${PROJECT_ROOT}"
mkdir -p "${LOG_DIR}"

# 2. 实验参数配置
# ==============================================================================
RUN_TIMES=5
MAX_WORKERS=16
# 消融实验通常聚焦在关键预算点 (如 10%)，若需全梯度可替换为完整梯度
TARGET_TICKS="0.1"
# TARGET_TICKS="0.01,0.05,0.075,0.1,0.125,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8"

RUNNER_SAMPLER="${PROJECT_ROOT}/pythonProject/src/runner/Proxy_Guided_Stratified_Importance_Sampling_Runner.py"

echo "=============================================================================="
echo "🚀 开始 RQ4 分配策略消融实验 (5种方法全量对比 - 并发模式)"
echo "项目根目录   : ${PROJECT_ROOT}"
echo "对比方法集合 : UN, PO, WO, MAB, 8_POSSA"
echo "采样率预算   : ${TARGET_TICKS}"
echo "日志目录     : ${LOG_DIR}"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# 3. 定义 6 大独立并行任务
# ------------------------------------------------------------------------------

# 任务 1: Parler COUNT 消融
run_parler_count() {
    "$PYTHON_EXEC" "${RUNNER_SAMPLER}" \
        -d 0 --agg_mode count \
        --target_ticks "${TARGET_TICKS}" \
        --run_times ${RUN_TIMES} \
        --max_workers ${MAX_WORKERS} \
        --base_dir "${PROJECT_ROOT}" \
        --methods all \
        --out_csv_name "allocation_strategy_comparison_ablation_count.csv"
}

# 任务 2: Parler SUM 消融
run_parler_sum() {
    "$PYTHON_EXEC" "${RUNNER_SAMPLER}" \
        -d 0 --agg_mode sum \
        --target_ticks "${TARGET_TICKS}" \
        --run_times ${RUN_TIMES} \
        --max_workers ${MAX_WORKERS} \
        --base_dir "${PROJECT_ROOT}" \
        --methods all \
        --out_csv_name "allocation_strategy_comparison_ablation_sum.csv"
}

# 任务 3: Parler-E COUNT 消融
run_parler_e_count() {
    "$PYTHON_EXEC" "${RUNNER_STEP1:-$RUNNER_SAMPLER}" \
        -d 1 --agg_mode count \
        --target_ticks "${TARGET_TICKS}" \
        --run_times ${RUN_TIMES} \
        --max_workers ${MAX_WORKERS} \
        --base_dir "${PROJECT_ROOT}" \
        --methods all \
        --out_csv_name "allocation_strategy_comparison_ablation_count.csv"
}

# 任务 4: Parler-E SUM 消融
run_parler_e_sum() {
    "$PYTHON_EXEC" "${RUNNER_SAMPLER}" \
        -d 1 --agg_mode sum \
        --target_ticks "${TARGET_TICKS}" \
        --run_times ${RUN_TIMES} \
        --max_workers ${MAX_WORKERS} \
        --base_dir "${PROJECT_ROOT}" \
        --methods all \
        --out_csv_name "allocation_strategy_comparison_ablation_sum.csv"
}

# 任务 5: Amazon COUNT 消融
run_amazon_count() {
    "$PYTHON_EXEC" "${RUNNER_SAMPLER}" \
        -d 2 --agg_mode count \
        --target_ticks "${TARGET_TICKS}" \
        --run_times ${RUN_TIMES} \
        --max_workers ${MAX_WORKERS} \
        --base_dir "${PROJECT_ROOT}" \
        --methods all \
        --out_csv_name "allocation_strategy_comparison_ablation_count.csv"
}

# 任务 6: Amazon SUM 消融
run_amazon_sum() {
    "$PYTHON_EXEC" "${RUNNER_SAMPLER}" \
        -d 2 --agg_mode sum \
        --target_ticks "${TARGET_TICKS}" \
        --run_times ${RUN_TIMES} \
        --max_workers ${MAX_WORKERS} \
        --base_dir "${PROJECT_ROOT}" \
        --methods all \
        --out_csv_name "allocation_strategy_comparison_ablation_sum.csv"
}

# ------------------------------------------------------------------------------
# 4. 后台并发调度与日志记录
# ------------------------------------------------------------------------------
echo -e "\n[*] 正在启动后台并发任务..."

run_parler_count   > "${LOG_DIR}/pythonProject/logs/RQ4_parler_count.log" 2>&1 &
PID_P_COUNT=$!
echo "  • [PID $PID_P_COUNT] Parler COUNT 消融 -> logs/RQ4_parler_count.log"

run_parler_sum     > "${LOG_DIR}/pythonProject/logs/RQ4_parler_sum.log" 2>&1 &
PID_P_SUM=$!
echo "  • [PID $PID_P_SUM] Parler SUM 消融 -> logs/RQ4_parler_sum.log"

run_parler_e_count > "${LOG_DIR}/pythonProject/logs/RQ4_parler_e_count.log" 2>&1 &
PID_PE_COUNT=$!
echo "  • [PID $PID_PE_COUNT] Parler-E COUNT 消融 -> logs/rq4_parler_e_count.log"

run_parler_e_sum   > "${LOG_DIR}/pythonProject/logs/RQ4_parler_e_sum.log" 2>&1 &
PID_PE_SUM=$!
echo "  • [PID $PID_PE_SUM] Parler-E SUM 消融 -> logs/rq4_parler_e_sum.log"

run_amazon_count   > "${LOG_DIR}/pythonProject/logs/RQ4_amazon_count.log" 2>&1 &
PID_A_COUNT=$!
echo "  • [PID $PID_A_COUNT] Amazon COUNT 消融 -> logs/rq4_amazon_count.log"

run_amazon_sum     > "${LOG_DIR}/pythonProject/logs/RQ4_amazon_sum.log" 2>&1 &
PID_A_SUM=$!
echo "  • [PID $PID_A_SUM] Amazon SUM 消融 -> logs/rq4_amazon_sum.log"

echo -e "\n[*] 等待所有 RQ4 消融任务执行完成..."

wait $PID_P_COUNT $PID_P_SUM $PID_PE_COUNT $PID_PE_SUM $PID_A_COUNT $PID_A_SUM

echo -e "\n=============================================================================="
echo "🎉 RQ4 分配策略消融实验全部顺利完成！"
echo "生成的核心对比文件:"
echo "  • Parler   : datasets/parler/results/efficiency/allocation_strategy_comparison_ablation_{count,sum}.csv"
echo "  • Parler-E : datasets/parler-E/results/efficiency/allocation_strategy_comparison_ablation_{count,sum}.csv"
echo "  • Amazon   : datasets/amazon/results/efficiency/allocation_strategy_comparison_ablation_{count,sum}.csv"
echo "=============================================================================="