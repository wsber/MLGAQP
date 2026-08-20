#!/usr/bin/env bash
# ==============================================================================
# 脚本名称: run_all_exact_structureO.sh
# 作用: 一键全并发执行 Amazon, Parler, Parler-E 的 Exact_structureO 基准测试 (COUNT & SUM)
# ==============================================================================

PYTHON_EXEC="/home/wangshuo/software/anaconda3/envs/proxy/bin/python"
source /home/wangshuo/software/anaconda3/etc/profile.d/conda.sh
conda activate proxy

set -e

# 动态解析项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "${SCRIPT_DIR}/../../pythonProject" ]; then
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
elif [ -d "${SCRIPT_DIR}/../pythonProject" ]; then
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    PROJECT_ROOT="${SCRIPT_DIR}"
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

LOG_DIR="${PROJECT_ROOT}/pythonProject/logs"
mkdir -p "${LOG_DIR}"

SCRIPT_PATH="${PROJECT_ROOT}/pythonProject/src/baseline/ENUM.py"
MAX_WORKERS=16

echo "=============================================================================="
echo "🚀 启动 ENUM 全数据集并发基线测试 (COUNT & SUM)"
echo "项目根目录 : ${PROJECT_ROOT}"
echo "日志目录   : ${LOG_DIR}"
echo "=============================================================================="

# 定义任务函数
run_task() {
    local dataset=$1
    local mode=$2
    "$PYTHON_EXEC" "${SCRIPT_PATH}" \
        --base_dir "${PROJECT_ROOT}" \
        --dataset_name "${dataset}" \
        --agg_mode "${mode}" \
        --max_workers ${MAX_WORKERS}
}

# ------------------------------------------------------------------------------
# 启动 6 项后台并发任务
# ------------------------------------------------------------------------------
echo -e "\n[*] 正在启动后台并发任务..."

run_task "amazon" "count"   > "${LOG_DIR}/enum_amazon_count.log" 2>&1 &
PID_A_C=$!
echo "  • [PID $PID_A_C] Amazon COUNT    -> logs/enum_amazon_count.log"

run_task "amazon" "sum"     > "${LOG_DIR}/enum_amazon_sum.log" 2>&1 &
PID_A_S=$!
echo "  • [PID $PID_A_S] Amazon SUM      -> logs/enum_amazon_sum.log"

run_task "parler" "count"   > "${LOG_DIR}/enum_parler_count.log" 2>&1 &
PID_P_C=$!
echo "  • [PID $PID_P_C] Parler COUNT    -> logs/enum_parler_count.log"

run_task "parler" "sum"     > "${LOG_DIR}/enum_parler_sum.log" 2>&1 &
PID_P_S=$!
echo "  • [PID $PID_P_S] Parler SUM      -> logs/enum_parler_sum.log"

run_task "parler-E" "count" > "${LOG_DIR}/enum_parler_e_count.log" 2>&1 &
PID_PE_C=$!
echo "  • [PID $PID_PE_C] Parler-E COUNT  -> logs/enum_parler_e_count.log"

run_task "parler-E" "sum"   > "${LOG_DIR}/enum_parler_e_sum.log" 2>&1 &
PID_PE_S=$!
echo "  • [PID $PID_PE_S] Parler-E SUM    -> logs/enum_parler_e_sum.log"

echo -e "\n[*] 等待所有基线任务完成..."
wait $PID_A_C $PID_A_S $PID_P_C $PID_P_S $PID_PE_C $PID_PE_S

echo -e "\n=============================================================================="
echo "🎉 所有 ENUM 基线测试已全部完成！"
echo "各数据集的输出曲线保存在各自的 results/efficiency/Exact_structureO_budget_curve_{count,sum}.csv"
echo "=============================================================================="