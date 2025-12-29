import pandas as pd

datadir = '/home/wangshuo/resource/datasets/parler_data/30W_valid_user/10-10-5/'
# filename = 'comment_test'
filename = 'comment'
file = datadir + f"{filename}.csv"
# 指标列名
oracle_probability = 'ML2_oracle2_probability'
proxy_probability = 'ML2_proxy2d2_probability'


def find_optimle_proxy_threshold(file, oracle_probability, proxy_probability, config_threshold=0.3):
    # 1. 读取文件（替换为实际路径）
    df = pd.read_csv(file)
    # 定义阈值列表
    delta1 = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
    delta2 = [0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.65, 0.7, 0.8, 0.84, 0.85, 0.86, 0.87, 0.88, 0.9, 0.95]
    results = []
    for d1 in delta1:
        TP = df[df[oracle_probability] > d1].shape[0]
        for d2 in delta2:
            R = df[df[proxy_probability] > d2].shape[0]
            RTP = df[
                (df[oracle_probability] > d1) &
                (df[proxy_probability] > d2)
                ].shape[0]

            precision = RTP / R if R > 0 else 0.0
            recall = RTP / TP if TP > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            results.append({
                oracle_probability: d1,
                proxy_probability: d2,
                'precision': precision,
                'recall': recall,
                'F1': f1,
                'res': R,
                'ture': RTP,
                'all_ture': TP
            })
    # 转成 DataFrame
    res_df = pd.DataFrame(results)

    # 1) F1 降序（不加额外筛选）
    f1_sorted = res_df.sort_values(by='F1', ascending=False).reset_index(drop=True)
    print("\n=== 按 F1 值降序排名（前 10 条） ===")
    print(f1_sorted.head(10).to_string(index=False, float_format='%.4f'))

    # 2) Precision 降序，但先剔除 recall < 0.3 的组合
    pre_threshold = config_threshold
    prec_filtered = res_df[res_df['recall'] >= pre_threshold]
    prec_sorted = prec_filtered.sort_values(by='precision', ascending=False).reset_index(drop=True)
    print(f"\n=== 按 Precision 值降序排名（recall >= {config_threshold}，前 10 条） ===")
    print(prec_sorted.head(10).to_string(index=False, float_format='%.4f'))

    # 3) Recall 降序，但先剔除 precision < 0.3 的组合
    rec_threshold = config_threshold
    rec_filtered = res_df[res_df['precision'] >= rec_threshold]
    rec_sorted = rec_filtered.sort_values(by='recall', ascending=False).reset_index(drop=True)
    print(f"\n=== 按 Recall 值降序排名（precision >= {config_threshold}，前 10 条） ===")
    print(rec_sorted.head(10).to_string(index=False, float_format='%.4f'))


find_optimle_proxy_threshold(file, oracle_probability, proxy_probability, config_threshold=0.3)