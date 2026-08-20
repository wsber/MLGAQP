#!/usr/bin/env bash
# ==============================================================================
# 脚本名称: run_RQ1.sh
# 作用: 真正全并发执行 Parler / Parler-E / Amazon 的 COUNT & SUM，最后自动合成 AVG
# ==============================================================================

set -e

# 动态定位脚本本身所在目录及项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 与 RQ2 保持一致，日志统一存放至 pythonProject/logs
LOG_DIR="${PROJECT_ROOT}/pythonProject/logs"
mkdir -p "${LOG_DIR}"

echo "=============================================================================="
echo "🚀 启动【全并行并发】全局实验流水线 (Parallel Master Pipeline)"
echo "📂 所有日志将实时输出至: ${LOG_DIR}/"
echo "=============================================================================="

# 后台并发执行辅助函数 (末尾加 & 转入后台)
run_bg() {
    SCRIPT_NAME=$1
    MODULE_DESC=$2
    LOG_FILE="${LOG_DIR}/${SCRIPT_NAME%.sh}.log"
    
    # 动态定位子脚本路径
    SCRIPT_PATH="${SCRIPT_DIR}/${SCRIPT_NAME}"
    
    chmod +x "${SCRIPT_PATH}"
    echo "[$(date +'%H:%M:%S')] ⚡ [后台启动] ${MODULE_DESC}"
    echo "   -> 脚本: ${SCRIPT_PATH}"
    echo "   -> 日志: ${LOG_FILE}"
    
    # 核心：使用 & 符号将任务打入后台并发执行
    bash "${SCRIPT_PATH}" > "${LOG_FILE}" 2>&1 &
}

# ------------------------------------------------------------------------------
# 阶段 1: 同时并发启动所有 6 个独立实验 (6 进程组并行)
# ------------------------------------------------------------------------------
echo -e "\n🔥 >>> [阶段 1/2] 正在同时触发所有数据集的 COUNT 和 SUM 实验... <<<"

run_bg "run_parler_count.sh"   "Parler   [COUNT] 评估"
run_bg "run_parler_sum.sh"     "Parler   [SUM]   评估"
run_bg "run_parler_e_count.sh" "Parler-E [COUNT] 评估"
run_bg "run_parler_e_sum.sh"   "Parler-E [SUM]   评估"
run_bg "run_amazon_count.sh"   "Amazon   [COUNT] 评估"
run_bg "run_amazon_sum.sh"     "Amazon   [SUM]   评估"

echo -e "\n⏳ 6 个任务已全部转入后台并行运行中！"
echo "💡 提示：你可以打开新终端输入 'htop' 或 'tail -f ${LOG_DIR}/run_amazon_sum.log' 查看实时动态。"
echo "⏳ 正在等待这 6 个实验全部并发结束 (通过 wait 阻塞等待)..."

# 关键：wait 会阻塞等待上面 6 个后台任务全部完成！
wait

echo -e "\n✅ [阶段 1/2] 恭喜！所有 6 个 COUNT 与 SUM 实验已全部并行完成！"

# ------------------------------------------------------------------------------
# 阶段 2: 执行最终的 AVG 比率离线合成
# ------------------------------------------------------------------------------
echo -e "\n🚀 >>> [阶段 2/2] 开始执行全数据集的 AVG 比率离线合成... <<<"

AVG_SCRIPT="${SCRIPT_DIR}/run_all_avg.sh"
AVG_LOG="${LOG_DIR}/run_all_avg.log"

if [ -f "${AVG_SCRIPT}" ]; then
    chmod +x "${AVG_SCRIPT}"
    bash "${AVG_SCRIPT}" > "${AVG_LOG}" 2>&1
    echo "✅ [阶段 2/2] 全数据集 AVG 离线合成已顺利完成！日志: ${AVG_LOG}"
else
    echo "⚠️ 警告: 未找到合成脚本 ${AVG_SCRIPT}，跳过此步。"
fi

echo -e "\n=============================================================================="
echo "🎉 所有实验全线告捷！全部数据已安全落盘至 datasets/*/results/efficiency/ 目录！"
echo "=============================================================================="