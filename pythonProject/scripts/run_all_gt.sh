#!/usr/bin/env bash
# ==============================================================================
# 脚本名称: run_all_gt.sh
# 作用: 全并发生产 Amazon, Parler, Parler-E 的 COUNT 和 SUM 真实基数 (Ground Truth)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_EXEC=$(command -v python)
if [ -z "$PYTHON_EXEC" ]; then
    echo "❌ 错误: 未找到 Python，请确保执行脚本前已激活虚拟环境！"
    exit 1
fi

set -e

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

LOG_DIR="${PROJECT_ROOT}/pythonProject/logs"
mkdir -p "${LOG_DIR}"

SCRIPT_PATH="${PROJECT_ROOT}/pythonProject/src/baseline/EXACT.py"

echo "=============================================================================="
echo "🚀 启动全数据集 T_true (Ground Truth) 并发生产流水线"
echo "项目根目录 : ${PROJECT_ROOT}"
echo "日志目录   : ${LOG_DIR}"
echo "=============================================================================="

# 定义单任务函数
run_gt_task() {
    local dataset=$1
    local mode=$2
    "$PYTHON_EXEC" "${SCRIPT_PATH}" --base_dir "${PROJECT_ROOT}" --dataset "${dataset}" --agg_mode "${mode}"
}

# ------------------------------------------------------------------------------
# 启动 6 大后台并发任务
# ------------------------------------------------------------------------------
echo -e "\n[*] 正在启动 6 项真值生产任务..."

run_gt_task "amazon" "count" > "${LOG_DIR}/gt_amazon_count.log" 2>&1 &
PID_A_COUNT=$!
echo "  • [PID $PID_A_COUNT] Amazon COUNT  -> logs/gt_amazon_count.log"

run_gt_task "amazon" "sum"   > "${LOG_DIR}/gt_amazon_sum.log" 2>&1 &
PID_A_SUM=$!
echo "  • [PID $PID_A_SUM] Amazon SUM    -> logs/gt_amazon_sum.log"

run_gt_task "parler" "count" > "${LOG_DIR}/gt_parler_count.log" 2>&1 &
PID_P_COUNT=$!
echo "  • [PID $PID_P_COUNT] Parler COUNT  -> logs/gt_parler_count.log"

run_gt_task "parler" "sum"   > "${LOG_DIR}/gt_parler_sum.log" 2>&1 &
PID_P_SUM=$!
echo "  • [PID $PID_P_SUM] Parler SUM    -> logs/gt_parler_sum.log"

run_gt_task "parler-E" "count" > "${LOG_DIR}/gt_parler_e_count.log" 2>&1 &
PID_PE_COUNT=$!
echo "  • [PID $PID_PE_COUNT] Parler-E COUNT -> logs/gt_parler_e_count.log"

run_gt_task "parler-E" "sum"   > "${LOG_DIR}/gt_parler_e_sum.log" 2>&1 &
PID_PE_SUM=$!
echo "  • [PID $PID_PE_SUM] Parler-E SUM   -> logs/gt_parler_e_sum.log"

echo -e "\n[*] 等待所有真值生产任务完成..."
wait $PID_A_COUNT $PID_A_SUM $PID_P_COUNT $PID_P_SUM $PID_PE_COUNT $PID_PE_SUM

echo -e "\n=============================================================================="
echo "🎉 全数据集 T_true 真值文件计算完毕！"
echo "已在各数据集 results/ 目录下生成对应的 T_true_*_{count,sum}.json 文件。"
echo "=============================================================================="