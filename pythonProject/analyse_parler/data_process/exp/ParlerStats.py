import pandas as pd
from transformers import AutoTokenizer
from typing import List, Tuple
import os

class ParlerStats:
    """
    Parler 数据统计分析类：用于统计用户发帖评论数量、bio信息筛选、字段频次分析、
    以及概率值阈值下的 F1、Precision、Recall 计算等。
    """

    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def user_post_comment_stats(self, user_path: str, post_path: str, comment_path: str, output_path: str = None):
        """
        统计每个用户发帖(post)和评论(comment)的数量，按评论数和发帖数排序
        """
        users_df = pd.read_csv(user_path)
        post_df = pd.read_csv(post_path)
        comment_df = pd.read_csv(comment_path)

        post_count = post_df['creator'].value_counts().reset_index()
        post_count.columns = ['id', 'post_sum']

        comment_count = comment_df['creator'].value_counts().reset_index()
        comment_count.columns = ['id', 'comment_sum']

        merged = pd.merge(post_count, comment_count, on='id', how='outer').fillna(0)
        final_df = pd.merge(users_df, merged, on='id', how='left')
        final_df_sorted = final_df.sort_values(by=['comment_sum', 'post_sum'], ascending=False)

        print(final_df_sorted[['id', 'post_sum', 'comment_sum']])

        if output_path:
            final_df_sorted.to_csv(output_path, index=False)

        return final_df_sorted

    def filter_bio(self, user_csv: str, output_path: str = None) -> pd.DataFrame:
        """
        筛选 bio 非空且单词数 ≥ 3 的用户
        """
        df = pd.read_csv(user_csv, dtype=str)
        empty_mask = df['bio'].isna() | df['bio'].str.strip().eq('')
        word_counts = df['bio'].fillna('').str.split().apply(len)
        keep_mask = (~empty_mask) & (word_counts >= 3)
        df_filtered = df[keep_mask].copy()

        print(f"原始行数：{len(df)}")
        print(f"保留行数：{len(df_filtered)}")

        if output_path:
            df_filtered.to_csv(output_path, index=False)

        return df_filtered

    def file_lengths(self, file_paths: List[str]):
        """
        输出多个 CSV 文件的记录数
        """
        for path in file_paths:
            df = pd.read_csv(path)
            print(f"{os.path.basename(path)} 文件包含 {len(df)} 行数据")

    def value_counts_for_columns(self, csv_file: str, columns: List[str]):
        """
        分析指定列的值频次
        """
        df = pd.read_csv(csv_file)
        for col in columns:
            print(f"\n列 {col} 的频次分布：")
            print(df[col].value_counts())

    def threshold_f1_analysis(self, file: str, oracle_col: str, proxy_col: str, delta1: List[float], delta2: List[float], config_threshold: float = 0.3):
        """
        计算 precision, recall, F1 分数，并排序输出
        """
        df = pd.read_csv(file)
        results = []

        for d1 in delta1:
            TP = df[df[oracle_col] > d1].shape[0]
            for d2 in delta2:
                R = df[df[proxy_col] > d2].shape[0]
                RTP = df[(df[oracle_col] > d1) & (df[proxy_col] > d2)].shape[0]

                precision = RTP / R if R > 0 else 0.0
                recall = RTP / TP if TP > 0 else 0.0
                f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

                results.append({
                    oracle_col: d1,
                    proxy_col: d2,
                    'precision': precision,
                    'recall': recall,
                    'F1': f1,
                    'res': R,
                    'true': RTP,
                    'all_true': TP
                })

        res_df = pd.DataFrame(results)

        print("\n=== 按 F1 值降序排名（前 10 条） ===")
        print(res_df.sort_values(by='F1', ascending=False).head(10).to_string(index=False, float_format='%.4f'))

        print(f"\n=== Precision 值降序（Recall >= {config_threshold}） ===")
        filtered = res_df[res_df['recall'] >= config_threshold]
        print(filtered.sort_values(by='precision', ascending=False).head(10).to_string(index=False, float_format='%.4f'))

        print(f"\n=== Recall 值降序（Precision >= {config_threshold}） ===")
        filtered = res_df[res_df['precision'] >= config_threshold]
        print(filtered.sort_values(by='recall', ascending=False).head(10).to_string(index=False, float_format='%.4f'))

        return res_df

    def analyze_dual_threshold_precision_recall(self, datadir: str):
        """
        读取指定路径下的 post.csv，按双阈值 (ML1_oracle1_probability > d1 且 ML1_proxy1_probability > d2)
        计算同时满足的样本数量，并输出对应的 precision 和 recall。
        """
        import pandas as pd

        oracle_pobability = 'ML1_oracle1_probability'
        proxy_pobability = 'ML1_proxy1_probability'
        delta1 = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
        delta2 = [0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.65, 0.7, 0.8, 0.84, 0.85, 0.86, 0.87, 0.88, 0.9]

        df = pd.read_csv(os.path.join(datadir, "post.csv"))

        print(f"\n=== 双阈值组合统计 ({oracle_pobability} > d1 且 {proxy_pobability} > d2) ===")
        for d1 in delta1:
            TP = df[df[oracle_pobability] > d1].shape[0]
            print(f'd1={d1}: {oracle_pobability} > {d1} 的数量: {TP}')
            for d2 in delta2:
                R = df[df[proxy_pobability] > d2].shape[0]
                RTP = df[
                    (df[oracle_pobability] > d1) &
                    (df[proxy_pobability] > d2)
                    ].shape[0]
                if R != 0 and TP != 0 and RTP != 0:
                    precision = RTP / R
                    recall = RTP / TP
                else:
                    precision = 0
                    recall = 0
                print(
                    f" d2={d2}: {proxy_pobability} > {d2} 的数量: {R}, 同时满足两个条件的数量: {RTP}, precision={precision}, recall={recall}")

    def count_label_with_threshold(self, csv_file: str, prob_col: str, label_col: str, label_value: str, threshold: float):
        """
        统计在概率列大于 threshold 且 标签列等于指定值的文本条目
        """
        df = pd.read_csv(csv_file)
        filtered = df[(df[prob_col] > threshold) & (df[label_col] == label_value)]
        print(f"共 {len(filtered)} 条满足条件的记录，正文如下：")
        for i, text in enumerate(filtered['body'], start=1):
            print(f"{i:4d}: {text}")

    def count_tokens(self, text: str, model_path: str = 'facebook/bart-large-mnli') -> int:
        """
        计算输入文本的分词数量（默认使用 BART Tokenizer）
        """
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokens = tokenizer.tokenize(text)
        print(f"Token 数量: {len(tokens)}")
        return len(tokens)


# 使用示例
if __name__ == '__main__':
    stats = ParlerStats(base_dir='/home/wangshuo/resource/datasets/parler_data/30W_valid_user/10-10-5/')

    # # 示例：统计用户发帖和评论数
    # stats.user_post_comment_stats(
    #     user_path=f"{stats.base_dir}/../middle_daset/30W_valid_user/user.csv",
    #     post_path=f"{stats.base_dir}/../middle_daset/30W_valid_user/post.csv",
    #     comment_path=f"{stats.base_dir}/../middle_daset/30W_valid_user/comment.csv",
    #     output_path=f"{stats.base_dir}/../middle_daset/sorted_user.csv"
    # )
    #
    # # 示例：筛选 bio
    # stats.filter_bio(
    #     user_csv=f"{stats.base_dir}/../users_test1.csv",
    #     output_path=f"{stats.base_dir}/../users_test1_filtered.csv"
    # )
    #
    # # 示例：打印文件行数
    # stats.file_lengths([
    #     f"{stats.base_dir}/../middle_daset/30W_valid_user/comment.csv",
    #     f"{stats.base_dir}/../middle_daset/30W_valid_user/post.csv"
    # ])
    #
    # # 示例：列值统计
    # stats.value_counts_for_columns(f"{stats.base_dir}/comment_test.csv",
    #                                 ['ML2_oracle1_label', 'ML2_proxy1_label'])

    # 示例：F1 分析
    stats.threshold_f1_analysis(
        file=f"{stats.base_dir}comment.csv",
        oracle_col='ML2_oracle1_probability',
        proxy_col='ML2_proxy3dd1_probability',
        delta1=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9],
        delta2=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
    )
    stats.analyze_dual_threshold_precision_recall(datadir=stats.base_dir)
    # # 示例：统计某概率下的标签
    # stats.count_label_with_threshold(
    #     csv_file=f"{stats.base_dir}/comment_ML2_proxy1.csv",
    #     prob_col='ML1_oracle2_probability',
    #     label_col='LLM_label',
    #     label_value='deepseek_yes',
    #     threshold=0.5
    # )

    # 示例：统计 token 数量
    sample_text = "I believe in the Spirit of America!"
    stats.count_tokens(sample_text, model_path='/home/wangshuo/resource/AIModels/NLP/bart-large-mnli')
