"""
从posts.csv文件中查找body属性中含有'trump'的1000个元组，这1000个元组的id必须和filtered_posts中的所有post的id不同，并将这些元组保存到新文件中，只保存下面字段
id:ID,:LABEL,comments,body,createdAt,creator,score,sensitive,upvotes,username,post,pcNum,entailment_probability,nli_label
"""
import pandas as pd

# 读取 posts.csv 和 filtered_posts.csv 文件
root = '/home/wangshuo/resource/datasets/IOGS/many_predicates/independent/dataset_2/'
posts_df = pd.read_csv(root + 'parler_posts_parler_data2.csv')
filtered_posts_df = pd.read_csv(root + 'posts.csv')

# 获取 filtered_posts 中的所有 post 的 id
filtered_ids = filtered_posts_df['id:ID'].tolist()

# 查找 body 字段中包含 'trump' 的元组，并且 id 不在 filtered_ids 中
filtered_trump_posts = posts_df[posts_df['body'].str.contains('trump', case=False, na=False)]
filtered_trump_posts = filtered_trump_posts[~filtered_trump_posts['id'].isin(filtered_ids)]

# 选取前 1000 个符合条件的元组
trump_posts_to_save = filtered_trump_posts.head(1000)

# 只保留指定字段
columns_to_save = ['id', 'comments', 'body', 'createdAt', 'creator', 'score', 'sensitive', 'upvotes', 'username', 'post']
trump_posts_to_save = trump_posts_to_save[columns_to_save]

# 保存到新文件
trump_posts_to_save.to_csv(root + 'trump_posts_filtered.csv', index=False)

print(f"已保存 {len(trump_posts_to_save)} 个符合条件的元组到 'trump_posts_filtered.csv'")
