#!/usr/bin/env bash
# ==============================================================================
# 脚本名称: run_all_proj_cascade_filter.sh
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "${SCRIPT_DIR}/../../pythonProject" ]; then
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
elif [ -d "${SCRIPT_DIR}/../pythonProject" ]; then
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    PROJECT_ROOT="${SCRIPT_DIR}"
fi

PYTHON_EXEC=$(command -v python)
set -e

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
LOG_DIR="${PROJECT_ROOT}/pythonProject/logs"
mkdir -p "${LOG_DIR}"

PYTHON_SCRIPT="${PROJECT_ROOT}/pythonProject/src/baseline/PROJ-Cascade-Filter.py"
WORKERS=16        

echo "=============================================================================="
echo "🚀 启动 PROJ-Cascade-Filter (Double Truncation) 全数据集并发基线测试"
echo "项目根目录 : ${PROJECT_ROOT}"
echo "=============================================================================="

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
            --table1 "product" --table1_proxy "ML3_proxy2_probability" --table1_oracle "ML3_oracle2_probability" \
            --t1_ids "post_id_list" --t1_low 0.3 --t1_high 0.4 \
            --table2 "review" --table2_proxy "ML2_proxy2_probability" --table2_oracle "ML2_oracle1_probability" \
            --t2_ids "comment_id_list" --t2_low 0.1 --t2_high 0.2 \
            --num_workers ${WORKERS} --out_csv "${out_csv}"

    elif [ "$dataset" == "parler" ]; then
        "$PYTHON_EXEC" "${PYTHON_SCRIPT}" \
            --parent_dataset "${dataset}" \
            --agg-mode "${agg_mode}" \
            --ablation_csv "${ablation_csv}" \
            --table1 "post" --table1_proxy "ML1_proxy4b_probability" --table1_oracle "ML1_oracle2_probability" \
            --t1_ids "post_id_list" --t1_low 0.7 --t1_high 0.9 \
            --table2 "comment" --table2_proxy "ML2_proxy1_probability" --table2_oracle "ML2_oracle2_probability" \
            --t2_ids "comment_id_list" --t2_low 0.2 --t2_high 0.3 \
            --num_workers ${WORKERS} --out_csv "${out_csv}"
            
    elif [ "$dataset" == "parler-E" ]; then
        "$PYTHON_EXEC" "${PYTHON_SCRIPT}" \
            --parent_dataset "${dataset}" \
            --agg-mode "${agg_mode}" \
            --ablation_csv "${ablation_csv}" \
            --table1 "post" --table1_proxy "ML1_proxy4b_probability" --table1_oracle "ML1_oracle2_probability" \
            --t1_ids "post_id_list" --t1_low 0.7 --t1_high 0.9 \
            --table2 "comment" --table2_proxy "ML2_proxy1_probability" --table2_oracle "ML2_oracle2_probability" \
            --t2_ids "comment_id_list" --t2_low 0.2 --t2_high 0.3 \
            --num_workers ${WORKERS} --out_csv "${out_csv}"
    fi
}

echo -e "\n[*] 正在启动三大数据集 (COUNT & SUM) 的后台并发任务..."

run_task "amazon" "count" > "${LOG_DIR}/proj_cascade_amazon_count.log" 2>&1 &
PID_A_COUNT=$!

run_task "amazon" "sum" > "${LOG_DIR}/proj_cascade_amazon_sum.log" 2>&1 &
PID_A_SUM=$!

run_task "parler" "count" > "${LOG_DIR}/proj_cascade_parler_count.log" 2>&1 &
PID_P_COUNT=$!

run_task "parler" "sum" > "${LOG_DIR}/proj_cascade_parler_sum.log" 2>&1 &
PID_P_SUM=$!

run_task "parler-E" "count" > "${LOG_DIR}/proj_cascade_parler_e_count.log" 2>&1 &
PID_PE_COUNT=$!

run_task "parler-E" "sum" > "${LOG_DIR}/proj_cascade_parler_e_sum.log" 2>&1 &
PID_PE_SUM=$!

wait $PID_A_SUM $PID_A_COUNT $PID_P_SUM $PID_P_COUNT $PID_PE_SUM $PID_PE_COUNT

echo -e "\n=============================================================================="
echo "🎉 所有数据集的 PROJ-Cascade-Filter 基线实验执行完毕！"
echo "=============================================================================="