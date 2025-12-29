import os
import re
import random
import pandas as pd
from tqdm import tqdm


class ParlerDataProcessor:
    """
    Parler 数据处理器：
    封装用户、帖子、评论等数据的预处理、排序、筛选、抽样等功能。
    """

    def __init__(self, base_dir: str):
        """
        初始化处理器
        :param base_dir: 根目录路径，下面会基于此构建各子路径
        """
        self.base_dir = '/home/wangshuo/resource/datasets/parler_data/'
        # 各数据集路径
        self.user_dir = os.path.join(base_dir, 'csv_data/user/')
        self.valid_user_dir = os.path.join(base_dir, 'csv_data/user/valid_users2/')
        self.posts_pc_dir = os.path.join(base_dir, 'csv_data/pc/posts/')
        self.valid_posts3 = os.path.join(self.posts_pc_dir, 'valid_posts3/')
        self.valid_posts4 = os.path.join(self.posts_pc_dir, 'valid_posts4')
        self.comments_pc_dir = os.path.join(base_dir, 'csv_data/pc/comments/')
        self.valid_comments2 = os.path.join(self.comments_pc_dir, 'valid_comments2')
        self.sub_data_dir = os.path.join(base_dir, 'csv_data/sub_data/middle_daset')
        os.makedirs(self.valid_user_dir, exist_ok=True)
        os.makedirs(self.valid_posts4, exist_ok=True)
        os.makedirs(self.sub_data_dir, exist_ok=True)

    def clean_text_column(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        清洗单列文本：去除换行、收敛多空格、剔除非ASCII字符
        """
        df[col] = df[col].fillna('')
        df[col] = df[col].str.replace(r'[\r\n]+', ' ', regex=True)  # 替换换行符
        df[col] = df[col].str.replace(r' {2,}', ' ', regex=True).str.strip()  # 合并多余空格
        df[col] = df[col].str.replace(r'[^\x00-\x7F]+', '', regex=True)  # 删除非ASCII字符
        return df

    def preprocess_user(self):
        """
        1. 预处理 user：清洗 bio 列
        """
        path = os.path.join(self.base_dir, f'sorted_user.csv')
        df = pd.read_csv(path, dtype=str)
        df = self.clean_text_column(df, 'bio')
        df.to_csv(path, index=False, encoding='utf-8')

    def preprocess_post(self):
        """
        2. 预处理 post/comment：清洗 body、bodywithurls、article、preview 列
        """
        path = os.path.join(self.base_dir, f'comment.csv')
        df = pd.read_csv(path, dtype=str)
        for col in tqdm(['body', 'bodywithurls', 'article', 'preview'], desc='Cleaning post text'):
            df = self.clean_text_column(df, col)
        df.to_csv(path, index=False, encoding='utf-8')

    def sort_by_comments(self, input_dir: str, posts_dir: str, comments_dir: str):
        """
        3. 按 comments 数量排序并拆分为 posts/comments
        """
        for d in [posts_dir, comments_dir]: os.makedirs(d, exist_ok=True)
        files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
        for f in tqdm(files, desc='Sorting files'):
            df = pd.read_csv(os.path.join(input_dir, f), dtype={'comments': str})
            df['comments'] = pd.to_numeric(df['comments'], errors='coerce').fillna(0).astype(int)
            for typ, out_dir in [('posts', posts_dir), ('comments', comments_dir)]:
                sub = df[df['datatype'] == typ]
                if not sub.empty:
                    sub = sub.sort_values('comments', ascending=False)
                    sub.to_csv(os.path.join(out_dir, f'parler_{typ}_{f}'), index=False)

    def filter_posts_min_comments(self, input_dir: str, output_dir: str, min_comments: int = 2):
        """
        4. 筛选 posts：评论数 >= min_comments
        """
        os.makedirs(output_dir, exist_ok=True)
        for f in tqdm(os.listdir(input_dir), desc='Filtering posts'):
            if not f.endswith('.csv'): continue
            df = pd.read_csv(os.path.join(input_dir, f))
            df = df[df['comments'] >= min_comments]
            if not df.empty:
                df.to_csv(os.path.join(output_dir, f), index=False)

    def sort_users(self, directory: str):
        """
        5. 用户排序：按 comments、posts 数量降序
        """
        files = [f for f in os.listdir(directory) if f.endswith('.csv')]
        for f in tqdm(files, desc='Sorting users'):
            df = pd.read_csv(os.path.join(directory, f))
            df = df.sort_values(by=['comments', 'posts'], ascending=[False, False])
            df.to_csv(os.path.join(directory, f), index=False)

    def filter_users_activity(self, input_dir: str, output_dir: str):
        """
        6. 筛选活跃用户：comments>=2 或 posts>=1
        """
        os.makedirs(output_dir, exist_ok=True)
        for f in tqdm(os.listdir(input_dir), desc='Filtering users'):
            if not f.endswith('.csv'): continue
            df = pd.read_csv(os.path.join(input_dir, f))
            df = df[(df['comments'] >= 2) | (df['posts'] >= 1)]
            df.to_csv(os.path.join(output_dir, f), index=False)

    def filter_posts_by_users(self, posts_dir: str, users_dir: str, output_dir: str):
        """
        7. 仅保留 creator 在有效用户集合中的 posts
        """
        os.makedirs(output_dir, exist_ok=True)
        ids = set()
        for f in tqdm(os.listdir(users_dir), desc='Collecting user IDs'):
            if f.endswith('.csv'):
                ids |= set(pd.read_csv(os.path.join(users_dir, f), usecols=['id'], dtype=str)['id'])
        for f in tqdm(os.listdir(posts_dir), desc='Filtering posts by user'):
            if not f.endswith('.csv'): continue
            df = pd.read_csv(os.path.join(posts_dir, f), dtype=str)
            df = df[df['creator'].isin(ids)]
            df.to_csv(os.path.join(output_dir, f), index=False)

    def sample_and_sort_posts(self, input_dir: str, n_files: int, m_tuples: int, output_file: str):
        """
        8. 随机抽样 n_files 个文件，每个取 m_tuples 条，合并后按 comments 排序
        """
        files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
        sampled = random.sample(files, n_files)
        frames = []
        for f in tqdm(sampled, desc='Sampling posts'):
            df = pd.read_csv(os.path.join(input_dir, f))
            frames.append(df.head(m_tuples))
        combined = pd.concat(frames)
        combined = combined.sort_values('comments', ascending=False)
        combined.to_csv(output_file, index=False)

    def count_sub_comments(self, post_file: str, comments_dir: str, output_file: str):
        """
        9. 为每条 post 统计关联评论数，保存到新列 sub_comment
        """
        post_df = pd.read_csv(post_file, dtype=str)
        post_ids = set(post_df['id'])
        counts = {pid: 0 for pid in post_ids}
        for f in tqdm(os.listdir(comments_dir), desc='Counting comments'):
            if not f.endswith('.csv'): continue
            df = pd.read_csv(os.path.join(comments_dir, f), dtype=str)
            df = df[df['post'].isin(post_ids)]
            for pid in df['post']:
                counts[pid] += 1
        post_df['sub_comment'] = post_df['id'].map(counts)
        post_df.to_csv(output_file, index=False)

    def extract_relevant_comments(self, post_file: str, comments_dir: str, output_file: str):
        """
        10. 提取与指定 posts 关联的所有评论并保存
        """
        post_ids = set(pd.read_csv(post_file, dtype=str)['id'])
        frames = []
        for f in tqdm(os.listdir(comments_dir), desc='Extract comments'):
            if not f.endswith('.csv'): continue
            df = pd.read_csv(os.path.join(comments_dir, f), dtype=str)
            df = df[df['post'].isin(post_ids)]
            if not df.empty:
                frames.append(df)
        if frames:
            pd.concat(frames).to_csv(output_file, index=False)

    def extract_child_comments(self, parent_file: str, children_dir: str, output_file: str):
        """
        11. 根据 parent 列提取所有子评论
        """
        parent_ids = set(pd.read_csv(parent_file, usecols=['id'], dtype=str)['id'])
        frames = []
        for f in tqdm(os.listdir(children_dir), desc='Child comments'):
            if not f.endswith('.csv'): continue
            df = pd.read_csv(os.path.join(children_dir, f), usecols=['parent', 'id', 'body', 'creator', 'createdAt'],
                             dtype=str)
            df = df[df['parent'].isin(parent_ids)]
            if not df.empty:
                frames.append(df)
        if frames:
            pd.concat(frames).to_csv(output_file, index=False)

    def merge_parent_and_child_comments(self, parent_file: str, child_file: str):
        """
        12. 将子评论追加到父评论文件末尾
        """
        df_p = pd.read_csv(parent_file, dtype=str)
        df_c = pd.read_csv(child_file, dtype=str)
        pd.concat([df_p, df_c], ignore_index=True).to_csv(parent_file, index=False)

    def extract_unique_users(self, valid_users_dir: str, post_file: str, comment_file: str, output_file: str):
        """
        13. 从 post/comment 文件提取唯一 creator，对应用户信息保存到 users.csv
        """
        dfs = [pd.read_csv(os.path.join(valid_users_dir, f), dtype=str)
               for f in tqdm(os.listdir(valid_users_dir), desc='Reading users') if f.endswith('.csv')]
        users_df = pd.concat(dfs, ignore_index=True)
        post_df = pd.read_csv(post_file, dtype=str)
        comment_df = pd.read_csv(comment_file, dtype=str)
        creators = set(post_df['creator']).union(set(comment_df['creator']))
        result = users_df[users_df['id'].isin(creators)]
        result.to_csv(output_file, index=False)
        print(f"用户元组数量: {len(result)}")
