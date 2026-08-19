#!/usr/bin/env bash
# ==============================================================================
# 脚本名称: run_fastesto_all.sh
# 作用: 一键后台并发跑完所有数据集的 FaSTest-Oracle 基线曲线实验
# ==============================================================================

PYTHON_EXEC="/home/wangshuo/software/anaconda3/envs/proxy/bin/python"
source /home/wangshuo/software/anaconda3/etc/profile.d/conda.sh
conda activate proxy

set -e

# 定位项目根目录 (自动追溯到 /home/wangshuo/projects/PROXY)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

LOG_DIR="${PROJECT_ROOT}/pythonProject/logs"
mkdir -p "${LOG_DIR}"

# 脚本与可执行文件真实路径
RUNNER_SCRIPT="${PROJECT_ROOT}/pythonProject/src/baseline/FASTEST-ORACLE.py"
# 【关键修正点】：指定你最新的 C++ build 目录
BUILD_DIR="${PROJECT_ROOT}/cProject/build"

echo "=============================================================================="
echo "🚀 启动 FaSTest-Oracle 全数据集并发基线测试"
echo "项目根目录 : ${PROJECT_ROOT}"
echo "C++ 目录   : ${BUILD_DIR}"
echo "日志目录   : ${LOG_DIR}"
echo "=============================================================================="

# 1. Parler 任务 (COUNT & SUM)
run_parler() {
    "$PYTHON_EXEC" "${RUNNER_SCRIPT}" \
        --dataset parler \
        --mode all \
        --base_dir "${PROJECT_ROOT}" \
        --build_dir "${BUILD_DIR}"
}

# 2. Parler-E 任务 (COUNT & SUM)
run_parler_e() {
    "$PYTHON_EXEC" "${RUNNER_SCRIPT}" \
        --dataset parler-E \
        --mode all \
        --base_dir "${PROJECT_ROOT}" \
        --build_dir "${BUILD_DIR}"
}

# 3. Amazon 任务 (COUNT & SUM)
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