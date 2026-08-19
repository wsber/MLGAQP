#!/usr/bin/env bash
# ==============================================================================
# 脚本名称: run_all_proj_abae.sh
# 作用: 一键并行执行所有数据集的 Projection-ABae 基线实验 (COUNT & SUM)
# ==============================================================================

PYTHON_EXEC="/home/wangshuo/software/anaconda3/envs/proxy/bin/python"
source /home/wangshuo/software/anaconda3/etc/profile.d/conda.sh
conda activate proxy

set -e

# 1. 自动精准定位项目根目录
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

PYTHON_SCRIPT="${PROJECT_ROOT}/pythonProject/src/baseline/PROJ-ABAE.py"

# 全局运行参数
RUN_TIMES=10      
WORKERS=16        
BUDGET_FRAC=0.1  
PILOT_RATIO=0.1  

echo "=============================================================================="
echo "🚀 启动 Projection-ABae 全数据集并发基线测试"
echo "项目根目录 : ${PROJECT_ROOT}"
echo "Python 脚本: ${PYTHON_SCRIPT}"
echo "日志目录   : ${LOG_DIR}"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# 2. 定义各数据集与模式的执行函数 (完全匹配 Python 的 argparse)
# ------------------------------------------------------------------------------

run_task() {
    local dataset=$1
    local agg_mode=$2
    
    "$PYTHON_EXEC" "${PYTHON_SCRIPT}" \
        --base_dir "${PROJECT_ROOT}" \
        --dataset "${dataset}" \
        --agg_mode "${agg_mode}" \
        --budget_frac ${BUDGET_FRAC} \
        --pilot_ratio ${PILOT_RATIO} \
        --runs ${RUN_TIMES} \
        --workers ${WORKERS}
}

# ------------------------------------------------------------------------------
# 3. 后台并发调度与日志重定向
# ------------------------------------------------------------------------------
echo -e "\n[*] 正在启动三大数据集的后台并发任务..."

run_task "parler" "count" > "${LOG_DIR}/proj_abae_parler_count.log" 2>&1 &
PID_P_COUNT=$!
echo "  • [PID $PID_P_COUNT] Parler COUNT 已启动"

run_task "parler" "sum" > "${LOG_DIR}/proj_abae_parler_sum.log" 2>&1 &
PID_P_SUM=$!
echo "  • [PID $PID_P_SUM] Parler SUM 已启动"

run_task "parler-E" "count" > "${LOG_DIR}/proj_abae_parler_e_count.log" 2>&1 &
PID_PE_COUNT=$!
echo "  • [PID $PID_PE_COUNT] Parler-E COUNT 已启动"

run_task "parler-E" "sum" > "${LOG_DIR}/proj_abae_parler_e_sum.log" 2>&1 &
PID_PE_SUM=$!
echo "  • [PID $PID_PE_SUM] Parler-E SUM 已启动"

run_task "amazon" "count" > "${LOG_DIR}/proj_abae_amazon_count.log" 2>&1 &
PID_A_COUNT=$!
echo "  • [PID $PID_A_COUNT] Amazon COUNT 已启动"

run_task "amazon" "sum" > "${LOG_DIR}/proj_abae_amazon_sum.log" 2>&1 &
PID_A_SUM=$!
echo "  • [PID $PID_A_SUM] Amazon SUM 已启动"

echo -e "\n[*] 等待所有 Projection-ABae 任务完成 (可使用 'tail -f logs/proj_abae_*.log' 查看进度)..."

wait $PID_P_COUNT $PID_P_SUM $PID_PE_COUNT $PID_PE_SUM $PID_A_COUNT $PID_A_SUM

echo -e "\n=============================================================================="
echo "🎉 所有数据集的 Projection-ABae 基线实验执行完毕！"
echo "=============================================================================="