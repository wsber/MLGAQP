import json
from transformers import pipeline
from tqdm import tqdm  # 导入 tqdm

# 设置本地模型路径
# model_dir = "/home/wangshuo/ws/AI_models/huggingface/distilbart-mnli-12-6"
model_dir = "/home/wangshuo/ws/AI_models/huggingface/electra-small-discriminator"

# 加载 zero-shot-classification pipeline 并指定模型路径
classifier = pipeline("zero-shot-classification", model=model_dir, tokenizer=model_dir,device=0)

# 预定义分类标签
categories = ["political discussion", "social issues", "economics and free markets", "culture and entertainment"]

def check_nli(post):
    # 使用零样本分类模型判断 post 的类别
    result = classifier(post, categories, multi_label=False)
    # 返回最相关的标签和得分
    return result['labels'][0], result['scores'][0]

# 统计每个分类的数量
category_count = {category: 0 for category in categories}

# 打开文件并处理数据
input_file = '../data0_posts.ndjson'
output_file = '/home/wangshuo/home/wangshuo/ws/python_project/OutData/analyse_parler/data0_posts_classified_electra-small-discriminator.ndjson'

# 计算总行数，方便设置进度条
with open(input_file, 'r', encoding='utf-8') as f_in:
    total_lines = sum(1 for line in f_in)

with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
    # 使用 tqdm 显示进度条，total 为总行数
    for line in tqdm(f_in, total=total_lines, desc="Processing posts", unit="post"):
        post_data = json.loads(line)
        post_body = post_data.get('body', '')

        if post_body:
            # 判断 post 的类别
            category, score = check_nli(post_body)
            post_data['class'] = category  # 添加类别字段

            # 统计该分类的数量
            category_count[category] += 1

            # 写入更新后的数据
            f_out.write(json.dumps(post_data, ensure_ascii=False) + '\n')

# 输出每个类别的 post 数量
print("\n分类统计结果:")
for category, count in category_count.items():
    print(f"{category}: {count} posts")
