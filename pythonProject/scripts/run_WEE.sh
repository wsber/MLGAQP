#!/usr/bin/env bash
# ==============================================================================
# 脚本名称: run_WEE.sh
# 作用: 并行执行所有数据集在 COUNT 和 SUM 下的 WO (Weight Only) 基准评估
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

# 日志目录
LOG_DIR="${PROJECT_ROOT}/pythonProject/logs"
mkdir -p "${LOG_DIR}"

RUNNER_SCRIPT="${PROJECT_ROOT}/pythonProject/src/baseline/WEE.py"
MAX_WORKERS=16

echo "=============================================================================="
echo "🚀 启动 WO Baseline 全数据集并发评估"
echo "项目根目录 : ${PROJECT_ROOT}"
echo "=============================================================================="

run_amazon() {
    local mode=$1
    echo "[*] Amazon ${mode^^}..."
    "$PYTHON_EXEC" "${RUNNER_SCRIPT}" \
        --parent_dataset amazon \
        --dataset_name amazon_extend \
        --agg_mode ${mode} \
        --table1 product \
        --t1_proxy ML3_proxy2_probability \
        --t1_oracle ML3_oracle2_probability \
        --table2 review \
        --t2_proxy ML2_proxy2_probability \
        --t2_oracle ML2_oracle1_probability \
        --workers ${MAX_WORKERS}
}

run_parler() {
    local mode=$1
    echo "[*] Parler ${mode^^}..."
    "$PYTHON_EXEC" "${RUNNER_SCRIPT}" \
        --parent_dataset parler \
        --dataset_name dataset_three \
        --agg_mode ${mode} \
        --table1 post \
        --t1_proxy ML1_proxy4b_probability \
        --t1_oracle ML1_oracle2_probability \
        --table2 comment \
        --t2_proxy ML2_proxy1_probability \
        --t2_oracle ML2_oracle2_probability \
        --workers ${MAX_WORKERS}
}

run_parler_e() {
    local mode=$1
    echo "[*] Parler-E ${mode^^}..."
    "$PYTHON_EXEC" "${RUNNER_SCRIPT}" \
        --parent_dataset parler-E \
        --dataset_name dataset_test \
        --agg_mode ${mode} \
        --table1 post \
        --t1_proxy ML1_proxy4b_probability \
        --t1_oracle ML1_oracle2_probability \
        --table2 comment \
        --t2_proxy ML2_proxy1_probability \
        --t2_oracle ML2_oracle2_probability \
        --workers ${MAX_WORKERS}
}

# ------------------------------------------------------------------------------
# 启动后台并发任务
# ------------------------------------------------------------------------------
echo -e "\n[*] 正在启动后台并发任务..."

run_amazon count   > "${LOG_DIR}/wo_amazon_count.log" 2>&1 &
PID_A_C=$!
run_amazon sum     > "${LOG_DIR}/wo_amazon_sum.log" 2>&1 &
PID_A_S=$!

run_parler count   > "${LOG_DIR}/wo_parler_count.log" 2>&1 &
PID_P_C=$!
run_parler sum     > "${LOG_DIR}/wo_parler_sum.log" 2>&1 &
PID_P_S=$!

run_parler_e count > "${LOG_DIR}/wo_parler_e_count.log" 2>&1 &
PID_PE_C=$!
run_parler_e sum   > "${LOG_DIR}/wo_parler_e_sum.log" 2>&1 &
PID_PE_S=$!

echo "[*] 等待所有进程计算完成..."
wait $PID_A_C $PID_A_S $PID_P_C $PID_P_S $PID_PE_C $PID_PE_S

echo -e "\n=============================================================================="
echo "🎉 所有 WO 基线评估已完成！"
echo "各数据集的详细 JSON 和 Summary CSV 已保存至各自的 results/efficiency/ 目录中。"
echo "=============================================================================="