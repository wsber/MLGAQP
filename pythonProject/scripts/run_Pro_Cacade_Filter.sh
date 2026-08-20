#!/usr/bin/env bash
# ==============================================================================
# 脚本名称: run_all_proj_cascade_filter.sh
# 作用: 一键并行执行所有数据集的 PROJ-Cascade-Filter (Double Truncation) 基线实验 (COUNT & SUM)
# ==============================================================================

# 1. 自动定位项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "${SCRIPT_DIR}/../../pythonProject" ]; then
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
elif [ -d "${SCRIPT_DIR}/../pythonProject" ]; then
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    PROJECT_ROOT="${SCRIPT_DIR}"
fi

# 2. 动态获取外部激活的 Python 环境
PYTHON_EXEC=$(command -v python)
if [ -z "$PYTHON_EXEC" ]; then
    echo "❌ 错误: 未找到 Python，请确保执行脚本前已激活正确的虚拟环境 (如: conda activate proxy)！"
    exit 1
fi

set -e

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

if [ -n "$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
fi

LOG_DIR="${PROJECT_ROOT}/pythonProject/logs"
mkdir -p "${LOG_DIR}"

# Python 脚本路径
PYTHON_SCRIPT="${PROJECT_ROOT}/pythonProject/src/baseline/PROJ-Cascade-Filter.py"

# 全局运行参数
WORKERS=16        

echo "=============================================================================="
echo "🚀 启动 PROJ-Cascade-Filter (Double Truncation) 全数据集并发基线测试"
echo "项目根目录 : ${PROJECT_ROOT}"
echo "Python 脚本: ${PYTHON_SCRIPT}"
echo "日志目录   : ${LOG_DIR}"
echo "并行核心数 : ${WORKERS}"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# 2. 定义执行函数 (根据不同数据集配置专用参数)
# ------------------------------------------------------------------------------

run_task() {
    local dataset=$1
    local agg_mode=$2
    local ablation_csv="${PROJECT_ROOT}/datasets/${dataset}/results/efficiency/allocation_strategy_comparison_${agg_mode}.csv"
    local out_csv="Pro_Double_Truncation_${dataset}_${agg_mode}.csv"

    if [ "$dataset" == "amazon" ]; then
        "$PYTHON_EXEC" "${PYTHON_SCRIPT}" \
            --parent_dataset "${dataset}" \
            --agg-mode "${agg_mode}" \
            --ablation_csv "${ablation_csv}" \
            --table1 "product" \
            --table1_proxy "ML3_proxy2_probability" \
            --table1_oracle "ML3_oracle2_probability" \
            --t1_ids "post_id_list" \
            --t1_low 0.4 \
            --t1_high 0.6 \
            --table2 "review" \
            --table2_proxy "ML2_proxy2_probability" \
            --table2_oracle "ML2_oracle1_probability" \
            --t2_ids "comment_id_list" \
            --t2_low 0.2 \
            --t2_high 0.3 \
            --num_workers ${WORKERS} \
            --out_csv "${out_csv}"

    elif [ "$dataset" == "parler" ] || [ "$dataset" == "parler-E" ]; then
        # 【修改提示】: 已修正此处原脚本的 Bug (原版错误写成了 product/review)
        "$PYTHON_EXEC" "${PYTHON_SCRIPT}" \
            --parent_dataset "${dataset}" \
            --agg-mode "${agg_mode}" \
            --ablation_csv "${ablation_csv}" \
            --table1 "post" \
            --table1_proxy "ML1_proxy4b_probability" \
            --table1_oracle "ML1_oracle2_probability" \
            --t1_ids "post_id_list" \
            --t1_low 0.7 \
            --t1_high 0.9 \
            --table2 "comment" \
            --table2_proxy "ML2_proxy1_probability" \
            --table2_oracle "ML2_oracle2_probability" \
            --t2_ids "comment_id_list" \
            --t2_low 0.2 \
            --t2_high 0.3 \
            --num_workers ${WORKERS} \
            --out_csv "${out_csv}"
    else
        echo "[Error] 未知的数据集: ${dataset}"
        return 1
    fi
}

# ------------------------------------------------------------------------------
# 3. 后台并发调度与日志重定向
# ------------------------------------------------------------------------------
echo -e "\n[*] 正在启动三大数据集 (COUNT & SUM) 的后台并发任务..."

# Parler
run_task "parler" "count" > "${LOG_DIR}/proj_cascade_parler_count.log" 2>&1 &
PID_P_COUNT=$!
echo "  • [PID $PID_P_COUNT] Parler COUNT 已启动"

run_task "parler" "sum" > "${LOG_DIR}/proj_cascade_parler_sum.log" 2>&1 &
PID_P_SUM=$!
echo "  • [PID $PID_P_SUM] Parler SUM 已启动"

# Parler-E
run_task "parler-E" "count" > "${LOG_DIR}/proj_cascade_parler_e_count.log" 2>&1 &
PID_PE_COUNT=$!
echo "  • [PID $PID_PE_COUNT] Parler-E COUNT 已启动"

run_task "parler-E" "sum" > "${LOG_DIR}/proj_cascade_parler_e_sum.log" 2>&1 &
PID_PE_SUM=$!
echo "  • [PID $PID_PE_SUM] Parler-E SUM 已启动"

# Amazon
run_task "amazon" "count" > "${LOG_DIR}/proj_cascade_amazon_count.log" 2>&1 &
PID_A_COUNT=$!
echo "  • [PID $PID_A_COUNT] Amazon COUNT 已启动"

run_task "amazon" "sum" > "${LOG_DIR}/proj_cascade_amazon_sum.log" 2>&1 &
PID_A_SUM=$!
echo "  • [PID $PID_A_SUM] Amazon SUM 已启动"

echo -e "\n[*] 等待所有 PROJ-Cascade-Filter 任务完成 (可使用 'tail -f ${LOG_DIR}/proj_cascade_*.log' 查看进度)..."

wait $PID_P_COUNT $PID_P_SUM $PID_PE_COUNT $PID_PE_SUM $PID_A_COUNT $PID_A_SUM

echo -e "\n=============================================================================="
echo "🎉 所有数据集的 PROJ-Cascade-Filter (Double Truncation) 基线实验执行完毕！"
echo "=============================================================================="