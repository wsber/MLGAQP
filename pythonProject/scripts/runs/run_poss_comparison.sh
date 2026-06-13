#!/usr/bin/env bash
set -euo pipefail

# 默认参数（可改）
DATASET_NAME="${1:-dataset_test}"
RUN_TIMES="${2:-3}"
TARGET_TICKS="${3:-0.01,0.05,0.075,0.1,0.125,0.15,0.2}"
MAX_WORKERS="${4:-}"

PY_FILE="/home/wangshuo/projects/Neo4j_Exp/pythonProject/src/paper_exp/POSS_comparation.py"

if [[ ! -f "$PY_FILE" ]]; then
  echo "Python file not found: $PY_FILE"
  exit 1
fi

CMD=(
  python "$PY_FILE"
  --dataset_name "$DATASET_NAME"
  --run_times "$RUN_TIMES"
  --target_ticks "$TARGET_TICKS"
)

if [[ -n "$MAX_WORKERS" ]]; then
  CMD+=(--max_workers "$MAX_WORKERS")
fi

echo "Running:"
printf '%q ' "${CMD[@]}"
echo
"${CMD[@]}"