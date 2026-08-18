#!/usr/bin/env bash
# ==============================================================================
# 脚本名称: run_RQ2.sh
# 作用: RQ2 核心性能评估总控脚本
#       1. 智能检测 aggregated_results_${agg} 目录，已存在则自动跳过 Step 1 提速
#       2. 并行运行 Parler-E (COUNT), Parler-E (SUM), Amazon (SUM)
#       3. 待 Parler-E 计数与求和完成后，自动离线合成 Parler-E (AVG)
#       4. 所有任务执行日志实时输出至 logs/ 目录
# ==============================================================================

# 1. 基础环境与全局路径
# ==============================================================================
PYTHON_EXEC="/home/wangshuo/software/anaconda3/envs/proxy/bin/python"
source /home/wangshuo/software/anaconda3/etc/profile.d/conda.sh
conda activate proxy

PROJECT_ROOT="/home/wangshuo/projects/PROXY"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

if [ -n "$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
fi

# 创建统一的日志存储目录
LOG_DIR="${PROJECT_ROOT}/pythonProject/logs"
mkdir -p "${LOG_DIR}"

# 2. 全局参数配置
# ==============================================================================
SAMPLE_BUDGET=60000
RUN_TIMES=5
MAX_WORKERS=16
TARGET_TICKS="0.01,0.05,0.075,0.1,0.125,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8"

# 脚本路径
RUNNER_STEP1="${PROJECT_ROOT}/pythonProject/src/runner/Projection_Sampling_and_Weight_Estimation_Runner.py"
RUNNER_STEP2="${PROJECT_ROOT}/pythonProject/src/runner/Proxy_Guided_Stratified_Importance_Sampling_Runner.py"
SYNTHESIZE_AVG="${PROJECT_ROOT}/pythonProject/src/runner/AVG_Runner.py"

echo "=============================================================================="
echo "🚀 开始 RQ2 核心性能评估流水线 (并发执行 + 智能跳过已物化数据)"
echo "工作目录   : ${PROJECT_ROOT}"
echo "采样率梯度 : ${TARGET_TICKS}"
echo "日志目录   : ${LOG_DIR}"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# 任务定义函数
# ------------------------------------------------------------------------------

# 任务 1: Parler-E COUNT
run_parler_e_count() {
    local dataset_name="parler-E"
    local dataset_id=1
    local agg_mode="count"
    local agg_dir="${PROJECT_ROOT}/datasets/${dataset_name}/results/aggregated_results_${agg_mode}"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Task 1] 启动 Parler-E COUNT 评估..."
    
    # 智能检查：如果已经存在物化的 aggregated_results_count 目录且不为空，则跳过 Step 1
    if [ -d "${agg_dir}" ] && [ -n "$(find "${agg_dir}" -maxdepth 1 -name '*.csv' -print -quit 2>/dev/null)" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Task 1] ⚡ 检测到已存在物化目录: ${agg_dir}，跳过 Step 1 权重物化！"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Task 1] [Step 1/2] 未检测到物化目录，开始调用 C++ 引擎生成投影权重..."
        "$PYTHON_EXEC" "${RUNNER_STEP1}" \
            --base_dir "${PROJECT_ROOT}" \
            --dataset "${dataset_name}" \
            --sample_budget ${SAMPLE_BUDGET} \
            --agg_func "${agg_mode}" \
            --table1 post \
            --table2 comment \
            --workers ${MAX_WORKERS} \
            --run_cpp
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Task 1] [Step 1/2] 权重物化完成！"
    fi
        
    # Step 2: POSSA 采样
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Task 1] [Step 2/2] 开始执行 COUNT 模式分层重要性采样..."
    "$PYTHON_EXEC" "${RUNNER_STEP2}" \
        -d ${dataset_id} \
        --agg_mode "${agg_mode}" \
        --target_ticks "${TARGET_TICKS}" \
        --run_times ${RUN_TIMES} \
        --max_workers ${MAX_WORKERS} \
        --base_dir "${PROJECT_ROOT}"
        
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Task 1] ✅ Parler-E COUNT 执行完毕！"
}

# 任务 2: Parler-E SUM
run_parler_e_sum() {
    local dataset_name="parler-E"
    local dataset_id=1
    local agg_mode="sum"
    local agg_dir="${PROJECT_ROOT}/datasets/${dataset_name}/results/aggregated_results_${agg_mode}"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Task 2] 启动 Parler-E SUM 评估..."
    
    # 智能检查：如果已经存在物化的 aggregated_results_sum 目录且不为空，则跳过 Step 1
    if [ -d "${agg_dir}" ] && [ -n "$(find "${agg_dir}" -maxdepth 1 -name '*.csv' -print -quit 2>/dev/null)" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Task 2] ⚡ 检测到已存在物化目录: ${agg_dir}，跳过 Step 1 权重物化！"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Task 2] [Step 1/2] 未检测到物化目录，开始调用 C++ 引擎生成投影权重..."
        "$PYTHON_EXEC" "${RUNNER_STEP1}" \
            --base_dir "${PROJECT_ROOT}" \
            --dataset "${dataset_name}" \
            --sample_budget ${SAMPLE_BUDGET} \
            --agg_func "${agg_mode}" \
            --sum_table post \
            --sum_col upvotes \
            --sum_label 2 \
            --table1 post \
            --table2 comment \
            --workers ${MAX_WORKERS} \
            --run_cpp
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Task 2] [Step 1/2] 权重物化完成！"
    fi
        
    # Step 2: POSSA 采样
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Task 2] [Step 2/2] 开始执行 SUM 模式分层重要性采样..."
    "$PYTHON_EXEC" "${RUNNER_STEP2}" \
        -d ${dataset_id} \
        --agg_mode "${agg_mode}" \
        --target_ticks "${TARGET_TICKS}" \
        --run_times ${RUN_TIMES} \
        --max_workers ${MAX_WORKERS} \
        --base_dir "${PROJECT_ROOT}"
        
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Task 2] ✅ Parler-E SUM 执行完毕！"
}

# 任务 3: Amazon SUM
run_amazon_sum() {
    local dataset_name="amazon"
    local dataset_id=2
    local agg_mode="sum"
    local agg_dir="${PROJECT_ROOT}/datasets/${dataset_name}/results/aggregated_results_${agg_mode}"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Task 3] 启动 Amazon SUM 评估..."
    
    # 智能检查：如果已经存在物化的 aggregated_results_sum 目录且不为空，则跳过 Step 1
    if [ -d "${agg_dir}" ] && [ -n "$(find "${agg_dir}" -maxdepth 1 -name '*.csv' -print -quit 2>/dev/null)" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Task 3] ⚡ 检测到已存在物化目录: ${agg_dir}，跳过 Step 1 权重物化！"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Task 3] [Step 1/2] 未检测到物化目录，开始调用 C++ 引擎生成投影权重..."
        "$PYTHON_EXEC" "${RUNNER_STEP1}" \
            --base_dir "${PROJECT_ROOT}" \
            --dataset "${dataset_name}" \
            --sample_budget ${SAMPLE_BUDGET} \
            --agg_func "${agg_mode}" \
            --sum_table product \
            --sum_col price \
            --sum_label 12 \
            --table1 product \
            --table2 review \
            --workers ${MAX_WORKERS} \
            --run_cpp
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Task 3] [Step 1/2] 权重物化完成！"
    fi
        
    # Step 2: POSSA 采样
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Task 3] [Step 2/2] 开始执行 SUM 模式分层重要性采样..."
    "$PYTHON_EXEC" "${RUNNER_STEP2}" \
        -d ${dataset_id} \
        --agg_mode "${agg_mode}" \
        --target_ticks "${TARGET_TICKS}" \
        --run_times ${RUN_TIMES} \
        --max_workers ${MAX_WORKERS} \
        --base_dir "${PROJECT_ROOT}"
        
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Task 3] ✅ Amazon SUM 执行完毕！"
}

# ------------------------------------------------------------------------------
# 3. 并发调度执行 (Background Execution & Logging)
# ------------------------------------------------------------------------------
echo -e "\n[*] 正在启动后台并发任务..."

run_parler_e_count > "${LOG_DIR}/RQ2_parler_e_count.log" 2>&1 &
PID_PARLER_E_COUNT=$!
echo "  • [PID $PID_PARLER_E_COUNT] Parler-E COUNT 任务已启动 -> logs/RQ2_parler_e_count.log"

run_parler_e_sum > "${LOG_DIR}/RQ2_parler_e_sum.log" 2>&1 &
PID_PARLER_E_SUM=$!
echo "  • [PID $PID_PARLER_E_SUM] Parler-E SUM 任务已启动 -> logs/RQ2_parler_e_sum.log"

run_amazon_sum > "${LOG_DIR}/RQ2_amazon_sum.log" 2>&1 &
PID_AMAZON_SUM=$!
echo "  • [PID $PID_AMAZON_SUM] Amazon SUM 任务已启动 -> logs/RQ2_amazon_sum.log"

echo -e "\n[*] 等待后台任务运行完成..."

wait $PID_PARLER_E_COUNT
EXIT_CODE_COUNT=$?

wait $PID_PARLER_E_SUM
EXIT_CODE_SUM=$?

wait $PID_AMAZON_SUM
EXIT_CODE_AMAZON=$?

# 检查子任务退出状态
if [ $EXIT_CODE_COUNT -ne 0 ] || [ $EXIT_CODE_SUM -ne 0 ]; then
    echo "❌ 错误: Parler-E 的 COUNT 或 SUM 阶段执行异常失败，终止后续 AVG 合成！"
    echo "请查看对应日志: ${LOG_DIR}/parler_e_count.log 或 ${LOG_DIR}/parler_e_sum.log"
    exit 1
fi

if [ $EXIT_CODE_AMAZON -ne 0 ]; then
    echo "⚠️ 警告: Amazon SUM 执行出现异常，请检查日志: ${LOG_DIR}/amazon_sum.log"
fi

echo -e "\n[$(date '+%Y-%m-%d %H:%M:%S')] ✅ 所有并行任务已完成！"

# ------------------------------------------------------------------------------
# 4. 后续任务: 触发 Parler-E AVG 离线合成 (依赖 COUNT 与 SUM)
# ------------------------------------------------------------------------------
echo -e "\n[Step 4] 正在根据定理 6 合成 Parler-E [AVG] 结果..."
"$PYTHON_EXEC" "${SYNTHESIZE_AVG}" \
    --parent_data parler-E \
    --dataset parler-E \
    --t1_oracle ML1_oracle2_probability \
    --t2_oracle ML2_oracle2_probability \
    > "${LOG_DIR}/parler_e_avg.log" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ [Step 4] Parler-E AVG 合成完毕 -> logs/parler_e_avg.log"

# ------------------------------------------------------------------------------
# 5. 完成提示
# ------------------------------------------------------------------------------
echo -e "\n=============================================================================="
echo "🎉 RQ2 全部实验与合成流程圆满完成！"
echo "各任务结果已保存至各数据集的 results/efficiency/ 目录:"
echo "  1. Parler-E COUNT: datasets/parler-E/results/efficiency/allocation_strategy_comparison_count.csv"
echo "  2. Parler-E SUM  : datasets/parler-E/results/efficiency/allocation_strategy_comparison_sum.csv"
echo "  3. Parler-E AVG  : datasets/parler-E/results/efficiency/allocation_strategy_comparison_avg.csv"
echo "  4. Amazon SUM    : datasets/amazon/results/efficiency/allocation_strategy_comparison_sum.csv"
echo "=============================================================================="