#!/bin/bash

# 定义项目根目录
PROJECT_ROOT="/home/wangshuo/projects/Neo4j_Exp"
SCRIPT_PATH="$PROJECT_ROOT/pythonProject/src/Structure_first/single_predicate/method.py"
DATASET="dataset_three"
# DATASET="dataset_test"
# 激活环境 (如果有 conda)
# source activate your_env_name

# 定义任务列表
# 格式: "dataset proxy_model oracle_model"
tasks=(
    # "Dist_SkewHigh_proxy4 Dist_SkewHigh_oracle_prob"
    # "Dist_SkewLow_proxy4 Dist_SkewLow_oracle_prob"

    "ML1_proxy2b_probability ML1_oracle2_probability"
    # "Dist_Extreme_Mix_proxy4 Dist_Extreme_Mix_oracle_prob"
    

    # "Dist_Bimodal_origin_proxy2 Dist_Bimodal_origin_oracle_prob"
    # "Dist_Bimodal_left_proxy4 Dist_Bimodal_left_oracle_prob"
    # "Dist_Bimodal_right_proxy4 Dist_Bimodal_right_oracle_prob"

    # "Dist_Beta_U_proxy4 Dist_Beta_U_oracle_prob"
    # "Dist_Asym_LeftHigh_proxy4 Dist_Asym_LeftHigh_oracle_prob"
    # "Dist_Asym_RightHigh_proxy4 Dist_Asym_RightHigh_oracle_prob"
    
    
    # "Dist_Normal_proxy2 Dist_Normal_oracle_prob"
    # "Dist_Uniform_proxy2 Dist_Uniform_oracle_prob"
)

# 循环并在后台运行
for task in "${tasks[@]}"; do
    set -- $task # 将字符串解析为参数 $1, $2
    echo "🚀 Starting task: Dataset=$DATASET, Proxy=$1"
    
    # 使用 nohup 在后台运行，日志输出到 logs 文件夹
    mkdir -p logs
    nohup python "$SCRIPT_PATH" \
        --dataset "$DATASET" \
        --proxy_model "$1" \
        --oracle_model "$2" \
        --run_times 20 \
        > "logs/${DATASET}_${1}.log" 2>&1 &
done

echo "✅ All tasks started in background. Check 'logs/' folder for output."