#!/usr/bin/env bash
set -euo pipefail

# ============ 默认参数（可直接改这里） ============
FASTEST_BIN="/home/wangshuo/projects/FaSTest-main/build/Fastest"
DATASET="dataset_three"
ROOT_LABEL="2"
SAMPLE_BUDGET="20000"

POST_ORACLE_COL="ML1_oracle2_probability"
COMMENT_ORACLE_COL="ML2_oracle2_probability"

AGG_FUNC="sum"
SUM_TABLE="post"
SUM_COL="upvotes"
SUM_LABEL="2"

MULTI_PROXY_PROB="ML1_proxy4b_probability"

BUDGET_CURVE_IN="/home/wangshuo/resource/datasets/parler_data/dataset_three/results/efficiency/allocation_strategy_comparison.csv"
FASTESTO_RUNS="2"
BUDGET_CURVE_OUT="/home/wangshuo/resource/datasets/parler_data/dataset_three/results/efficiency/FastestO_budget_curve.csv"

# 开关项（1=开启，0=关闭）
ESTIMATE_WITH_PREDICATE="1"
FASTESTO_BUDGET_CURVE="1"

usage() {
  cat << 'EOF'
用法:
  ./run_fastesto.sh [可选参数]

可选参数:
  --bin PATH
  --dataset NAME
  --root-label INT
  --sample-budget INT
  --post-oracle-col NAME
  --comment-oracle-col NAME
  --agg-func NAME
  --sum-table NAME
  --sum-col NAME
  --sum-label INT
  --multi-proxy-prob NAME
  --budget-curve-in PATH
  --fastesto-runs INT
  --budget-curve-out PATH

  --estimate-with-predicate       开启该flag
  --no-estimate-with-predicate    关闭该flag
  --fastesto-budget-curve         开启该flag
  --no-fastesto-budget-curve      关闭该flag

  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bin) FASTEST_BIN="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --root-label) ROOT_LABEL="$2"; shift 2 ;;
    --sample-budget) SAMPLE_BUDGET="$2"; shift 2 ;;
    --post-oracle-col) POST_ORACLE_COL="$2"; shift 2 ;;
    --comment-oracle-col) COMMENT_ORACLE_COL="$2"; shift 2 ;;
    --agg-func) AGG_FUNC="$2"; shift 2 ;;
    --sum-table) SUM_TABLE="$2"; shift 2 ;;
    --sum-col) SUM_COL="$2"; shift 2 ;;
    --sum-label) SUM_LABEL="$2"; shift 2 ;;
    --multi-proxy-prob) MULTI_PROXY_PROB="$2"; shift 2 ;;
    --budget-curve-in) BUDGET_CURVE_IN="$2"; shift 2 ;;
    --fastesto-runs) FASTESTO_RUNS="$2"; shift 2 ;;
    --budget-curve-out) BUDGET_CURVE_OUT="$2"; shift 2 ;;

    --estimate-with-predicate) ESTIMATE_WITH_PREDICATE="1"; shift ;;
    --no-estimate-with-predicate) ESTIMATE_WITH_PREDICATE="0"; shift ;;
    --fastesto-budget-curve) FASTESTO_BUDGET_CURVE="1"; shift ;;
    --no-fastesto-budget-curve) FASTESTO_BUDGET_CURVE="0"; shift ;;

    -h|--help) usage; exit 0 ;;
    *) echo "未知参数: $1"; usage; exit 1 ;;
  esac
done

if [[ ! -x "$FASTEST_BIN" ]]; then
  echo "错误: 可执行文件不存在或无执行权限: $FASTEST_BIN"
  exit 1
fi

if [[ ! -f "$BUDGET_CURVE_IN" ]]; then
  echo "错误: 输入预算曲线文件不存在: $BUDGET_CURVE_IN"
  exit 1
fi

mkdir -p "$(dirname "$BUDGET_CURVE_OUT")"

cmd=(
  "$FASTEST_BIN"
  -d "$DATASET"
  --ROOT_LABEL "$ROOT_LABEL"
  --SAMPLE_BUDGET "$SAMPLE_BUDGET"
  --POST_ORACLE_COL "$POST_ORACLE_COL"
  --COMMENT_ORACLE_COL "$COMMENT_ORACLE_COL"
  --AGG_FUNC "$AGG_FUNC"
  --SUM_TABLE "$SUM_TABLE"
  --SUM_COL "$SUM_COL"
  --SUM_LABEL "$SUM_LABEL"
  --MULTI_PROXY_PROB "$MULTI_PROXY_PROB"
  --BUDGET_CURVE_IN "$BUDGET_CURVE_IN"
  --FASTESTO_RUNS "$FASTESTO_RUNS"
  --FASTESTO_BUDGET_CURVE_OUT "$BUDGET_CURVE_OUT"
)

if [[ "$ESTIMATE_WITH_PREDICATE" == "1" ]]; then
  cmd+=(--ESTIMATE_WITH_PREDICATE)
fi

if [[ "$FASTESTO_BUDGET_CURVE" == "1" ]]; then
  cmd+=(--FASTESTO_BUDGET_CURVE)
fi

echo "即将执行命令:"
printf '%q ' "${cmd[@]}"
echo

"${cmd[@]}"

echo "完成，输出文件: $BUDGET_CURVE_OUT"