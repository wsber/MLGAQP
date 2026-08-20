#!/usr/bin/env bash
# ==============================================================================
# 脚本名称: run_get_all_structural_matching.sh
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
MATCHER_SCRIPT="${PROJECT_ROOT}/pythonProject/src/algorithms/precision_submatching.py"

echo "=============================================================================="
echo "⚠️  【精确子图匹配 (ENUM / Ground Truth 生产引擎)】"
echo "   精确子图匹配为纯拓扑搜索，与 COUNT/SUM 聚合模式无关。"
echo "   Amazon 仅需运行一次全量查询匹配即可覆盖全部求和子图！"
echo "=============================================================================="
echo "  1) Parler    (单谓词基准 -  查询)"
echo "  2) Parler-E  (多谓词扩展 -  查询)"
echo "  3) Amazon    (异构全量子图 - 一次性匹配全部查询)"
echo "  q) 退出"
echo "=============================================================================="

read -p "请选择要执行的数据集 [1-3 或 q]: " CHOICE

case "$CHOICE" in
    1)
        "$PYTHON_EXEC" "${MATCHER_SCRIPT}" --base_dir "${PROJECT_ROOT}" --dataset "parler"
        ;;
    2)
        "$PYTHON_EXEC" "${MATCHER_SCRIPT}" --base_dir "${PROJECT_ROOT}" --dataset "parler-E"
        ;;
    3)
        "$PYTHON_EXEC" "${MATCHER_SCRIPT}" --base_dir "${PROJECT_ROOT}" --dataset "amazon"
        ;;
    q|Q)
        echo "已退出。"
        exit 0
        ;;
    *)
        echo "❌ 无效选项。"
        exit 1
        ;;
esac