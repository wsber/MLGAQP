#!/usr/bin/env bash
# ==============================================================================
# 脚本名称: run_ground_truth_matching.sh
# 作用: 交互式执行精确子图匹配 (生成 Ground Truth 与复现 ENUM 基线)
# 特性: 由于精确匹配极其消耗 CPU 和内存，每次严格限制只选择运行一个数据集！
# ==============================================================================

PYTHON_EXEC="/home/wangshuo/software/anaconda3/envs/proxy/bin/python"
source /home/wangshuo/software/anaconda3/etc/profile.d/conda.sh
conda activate proxy

set -e

# 定位项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "${SCRIPT_DIR}/../../pythonProject" ]; then
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
elif [ -d "${SCRIPT_DIR}/../pythonProject" ]; then
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    PROJECT_ROOT="${SCRIPT_DIR}"
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

MATCHER_SCRIPT="${PROJECT_ROOT}/pythonProject/src/algorithms/precision_submatching.py"

echo "=============================================================================="
echo "⚠️  【重要提示】精确子图匹配 (Exact Subgraph Matching / ENUM 基线)"
echo "   由于精确回溯枚举复杂度极高（耗时数小时且极度占用内存），系统不支持全量并发。"
echo "   请在下方菜单中选择【本次仅运行的一个数据集】进行真值生成："
echo "=============================================================================="
echo "  1) Parler    (单谓词基准 - 统一查询集)"
echo "  2) Parler-E  (多谓词扩展 - 统一查询集)"
echo "  3) Amazon    (COUNT 模式 - query_graph_count)"
echo "  4) Amazon    (SUM 模式 - query_graph_sum)"
echo "  q) 退出"
echo "=============================================================================="

read -p "请输入选项 [1-4 或 q]: " CHOICE

case "$CHOICE" in
    1)
        echo -e "\n[*] 开始执行: Parler 精确子图匹配..."
        "$PYTHON_EXEC" "${MATCHER_SCRIPT}" --base_dir "${PROJECT_ROOT}" --dataset "parler" --agg_mode count
        ;;
    2)
        echo -e "\n[*] 开始执行: Parler-E 精确子图匹配..."
        "$PYTHON_EXEC" "${MATCHER_SCRIPT}" --base_dir "${PROJECT_ROOT}" --dataset "parler-E" --agg_mode count
        ;;
    3)
        echo -e "\n[*] 开始执行: Amazon (COUNT 模式) 精确子图匹配..."
        "$PYTHON_EXEC" "${MATCHER_SCRIPT}" --base_dir "${PROJECT_ROOT}" --dataset "amazon" --agg_mode count
        ;;
    4)
        echo -e "\n[*] 开始执行: Amazon (SUM 模式) 精确子图匹配..."
        "$PYTHON_EXEC" "${MATCHER_SCRIPT}" --base_dir "${PROJECT_ROOT}" --dataset "amazon" --agg_mode sum
        ;;
    q|Q)
        echo "已取消操作。"
        exit 0
        ;;
    *)
        echo "❌ 无效选项，退出。"
        exit 1
        ;;
esac

echo -e "\n=============================================================================="
echo "✅ 所选数据集的精确匹配已完成，Ground Truth 文件已更新！"
echo "=============================================================================="