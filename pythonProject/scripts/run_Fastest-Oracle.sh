#!/usr/bin/env bash
# ==============================================================================
# 脚本名称: run_Fastesto_Oracle.sh
# 作用: 一键后台并发跑完所有数据集的 FaSTest-Oracle 基线曲线实验
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

LOG_DIR="${PROJECT_ROOT}/pythonProject/logs"
mkdir -p "${LOG_DIR}"

# 脚本与可执行文件真实路径
RUNNER_SCRIPT="${PROJECT_ROOT}/pythonProject/src/baseline/FASTEST-ORACLE.py"
BUILD_DIR="${PROJECT_ROOT}/cProject/build"

echo "=============================================================================="
echo "🚀 启动 FaSTest-Oracle 全数据集并发基线测试"
echo "项目根目录 : ${PROJECT_ROOT}"
echo "C++ 目录   : ${BUILD_DIR}"
echo "日志目录   : ${LOG_DIR}"
echo "=============================================================================="

run_parler() {
    "$PYTHON_EXEC" "${RUNNER_SCRIPT}" \
        --dataset parler \
        --mode all \
        --base_dir "${PROJECT_ROOT}" \
        --build_dir "${BUILD_DIR}"
}

run_parler_e() {
    "$PYTHON_EXEC" "${RUNNER_SCRIPT}" \
        --dataset parler-E \
        --mode all \
        --base_dir "${PROJECT_ROOT}" \
        --build_dir "${BUILD_DIR}"
}

run_amazon() {
    "$PYTHON_EXEC" "${RUNNER_SCRIPT}" \
        --dataset amazon \
        --mode all \
        --base_dir "${PROJECT_ROOT}" \
        --build_dir "${BUILD_DIR}"
}

echo -e "\n[*] 正在启动三大数据集的后台任务..."

run_parler   > "${LOG_DIR}/fastesto_parler.log" 2>&1 &
PID_P=$!
echo "  • [PID $PID_P] Parler 任务已启动   -> logs/fastesto_parler.log"

run_parler_e > "${LOG_DIR}/fastesto_parler_e.log" 2>&1 &
PID_PE=$!
echo "  • [PID $PID_PE] Parler-E 任务已启动 -> logs/fastesto_parler_e.log"

run_amazon   > "${LOG_DIR}/fastesto_amazon.log" 2>&1 &
PID_A=$!
echo "  • [PID $PID_A] Amazon 任务已启动   -> logs/fastesto_amazon.log"

echo -e "\n[*] 等待所有基线任务完成..."
wait $PID_P $PID_PE $PID_A

echo -e "\n=============================================================================="
echo "🎉 所有数据集的 FaSTest-Oracle 基线实验执行完毕！"
echo "生成结果文件位于各数据集的 results/efficiency/FastestO_budget_curve_{count,sum}.csv"
echo "=============================================================================="