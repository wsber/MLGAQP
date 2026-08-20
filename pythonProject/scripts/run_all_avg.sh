#!/usr/bin/env bash
# ==============================================================================
# 脚本名称: run_all_avg.sh
# 作用: 一键合成 Parler, Parler-E, Amazon 三个数据集的 AVG 实验结果与真值
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_EXEC=$(command -v python)
if [ -z "$PYTHON_EXEC" ]; then
    echo "❌ 错误: 未找到 Python，请确保执行脚本前已激活虚拟环境！"
    exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

SCRIPT_PATH="${PROJECT_ROOT}/pythonProject/src/runner/AVG_Runner.py"

echo "========================================================================="
echo "🚀 开始执行所有数据集的 AVG 离线合成流水线..."
echo "工作目录: ${PROJECT_ROOT}"
echo "========================================================================="

# 1. Parler 数据集
echo -e "\n>>> 1. Processing Parler (dataset_three)..."
"$PYTHON_EXEC" "${SCRIPT_PATH}" \
    --parent_data parler \
    --dataset parler \
    --t1_oracle ML1_oracle2_probability \
    --t2_oracle ML2_oracle2_probability

# 2. Parler-E 数据集
echo -e "\n>>> 2. Processing Parler-E (dataset_test)..."
"$PYTHON_EXEC" "${SCRIPT_PATH}" \
    --parent_data parler \
    --dataset parler-E \
    --t1_oracle ML1_oracle2_probability \
    --t2_oracle ML2_oracle2_probability

# 3. Amazon 数据集
echo -e "\n>>> 3. Processing Amazon (amazon_extend)..."
"$PYTHON_EXEC" "${SCRIPT_PATH}" \
    --parent_data amazon \
    --dataset amazon \
    --t1_oracle ML3_oracle2_probability \
    --t2_oracle ML2_oracle1_probability

echo -e "\n========================================================================="
echo "🎉 全线完成！所有 AVG 数据已成功生成！"
echo "========================================================================="