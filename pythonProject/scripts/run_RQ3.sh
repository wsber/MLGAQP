#!/usr/bin/env bash
# ==============================================================================
# 脚本名称: run_RQ3.sh
# 作用: RQ3 代理模型质量敏感性消融实验 (一键并行执行)
#       1. 并行运行 Parler (单谓词 Q0~Q4)
#       2. 并行运行 Parler-E (多谓词 Q1~Q4)
#       3. 所有任务执行日志实时输出至 logs/ 目录
# ==============================================================================

# 1. 基础环境与全局路径
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

if [ -n "$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
fi

# 日志目录
LOG_DIR="${PROJECT_ROOT}/pythonProject/logs"
mkdir -p "${LOG_DIR}"

# 2. 实验参数配置
# ==============================================================================
RUN_TIMES=5
MAX_WORKERS=16
TARGET_TICKS="0.1"
RUNNER_RQ3="${PROJECT_ROOT}/pythonProject/src/runner/proxy_quality_runner.py"

echo "=============================================================================="
echo "🚀 开始 RQ3 代理模型质量敏感性消融实验 (并发模式)"
echo "项目根目录 : ${PROJECT_ROOT}"
echo "执行脚本   : ${RUNNER_RQ3}"
echo "采样率梯度 : ${TARGET_TICKS}"
echo "日志目录   : ${LOG_DIR}"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# 任务 1: Parler 单谓词 Proxy 消融 (Single-predicate: Q0~Q4)
# ------------------------------------------------------------------------------
run_parler_single_pred() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Task 1] 启动 Parler 单谓词 Proxy 质量消融..."
    "$PYTHON_EXEC" "${RUNNER_RQ3}" \
        --base_dir "${PROJECT_ROOT}" \
        --dataset "parler" \
        --mode "single" \
        --run_times ${RUN_TIMES} \
        --max_workers ${MAX_WORKERS} \
        --target_ticks "${TARGET_TICKS}" \
        --out_csv "proxy_quality_ablation_count.csv"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Task 1] ✅ Parler 单谓词 Proxy 消融完成！"
}

# ------------------------------------------------------------------------------
# 任务 2: Parler-E 多谓词 Proxy 对消融 (Multi-predicate: Q1~Q4)
# ------------------------------------------------------------------------------
run_parler_e_multi_pred() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Task 2] 启动 Parler-E 多谓词 Proxy 质量消融..."
    "$PYTHON_EXEC" "${RUNNER_RQ3}" \
        --base_dir "${PROJECT_ROOT}" \
        --dataset "parler-E" \
        --mode "multi" \
        --run_times ${RUN_TIMES} \
        --max_workers ${MAX_WORKERS} \
        --target_ticks "${TARGET_TICKS}" \
        --out_csv "proxy_quality_multipred_ablation_count.csv"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Task 2] ✅ Parler-E 多谓词 Proxy 消融完成！"
}

# ------------------------------------------------------------------------------
# 3. 后台并发调度与日志记录
# ------------------------------------------------------------------------------
echo -e "\n[*] 正在启动后台并发任务..."

run_parler_single_pred > "${LOG_DIR}/RQ3_parler_single_proxy.log" 2>&1 &
PID_SINGLE=$!
echo "  • [PID $PID_SINGLE] Parler 单谓词任务已启动 -> logs/RQ3_parler_single_proxy.log"

run_parler_e_multi_pred > "${LOG_DIR}/RQ3_parler_e_multi_proxy.log" 2>&1 &
PID_MULTI=$!
echo "  • [PID $PID_MULTI] Parler-E 多谓词任务已启动 -> logs/RQ3_parler_e_multi_proxy.log"

echo -e "\n[*] 等待所有 RQ3 任务执行完成..."

wait $PID_SINGLE
EXIT_SINGLE=$?

wait $PID_MULTI
EXIT_MULTI=$?

# 错误捕获
if [ $EXIT_SINGLE -ne 0 ]; then
    echo "❌ 错误: Parler 单谓词消融任务执行异常，请检查: ${LOG_DIR}/RQ3_parler_single_proxy.log"
fi

if [ $EXIT_MULTI -ne 0 ]; then
    echo "❌ 错误: Parler-E 多谓词消融任务执行异常，请检查: ${LOG_DIR}/RQ3_parler_e_multi_proxy.log"
fi

if [ $EXIT_SINGLE -eq 0 ] && [ $EXIT_MULTI -eq 0 ]; then
    echo -e "\n=============================================================================="
    echo "🎉 RQ3 全部消融实验圆满完成！"
    echo "输出结果文件:"
    echo "  1. Parler:   datasets/parler/results/efficiency/proxy_quality_ablation_count.csv"
    echo "  2. Parler-E: datasets/parler-E/results/efficiency/proxy_quality_multipred_ablation_count.csv"
    echo "=============================================================================="
fi