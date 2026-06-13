import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from scipy.special import softmax
from tqdm import tqdm

# 1. 模型与 tokenizer 路径或名称
MODEL = "/home/wangshuo/resource/AIModels/NLP/TE/twitter-roberta-base-sentiment"

# 2. 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3. 下载并读取标签映射（0: negative, 1: neutral, 2: positive）
#    如果不希望运行时下载，也可以手动提前下载 mapping.txt 并改为本地路径
import csv
mapping_file = "/home/wangshuo/resource/AIModels/NLP/TE/twitter-roberta-base-sentiment/mapping.txt"  # 或者写成绝对路径 "/home/wangshuo/.../mapping.txt"
labels = []
with open(mapping_file, encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        if len(row) >= 2:
            labels.append(row[1])

# 4. 定义预处理函数（将 @user、链接 等替换为占位符）
def preprocess(text: str) -> str:
    tokens = []
    for t in text.split():
        if t.startswith("@") and len(t) > 1:
            tokens.append("@user")
        elif t.startswith("http"):
            tokens.append("http")
        else:
            tokens.append(t)
    return " ".join(tokens)

# 5. 读取你的 CSV，假设有一列叫 'body'
datadir = '/home/wangshuo/resource/datasets/parler_data/30W_valid_user/10-10-5'
in_csv = f"{datadir}/comment_test.csv"    # 替换为你的文件名
out_csv = f"{datadir}/comment_test.csv"
df = pd.read_csv(in_csv)
all_texts = df['body'].fillna("").astype(str).tolist()

# 6. 批量推理函数
def sentiment_batch(batch_texts):
    # 6.1 预处理
    texts = [preprocess(t) for t in batch_texts]
    # 6.2 编码
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)
    # 6.3 推理
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        scores = outputs.logits.cpu().numpy()
    # 6.4 软最大化
    probs = softmax(scores, axis=1)
    # 6.5 取最高概率及对应标签
    max_ids = probs.argmax(axis=1)
    max_probs = probs.max(axis=1)
    labels_out = [labels[i] for i in max_ids]
    return labels_out, max_probs

# 7. 循环处理并记录结果
batch_size = 64
sent_labels = []
sent_scores = []

for i in tqdm(range(0, len(all_texts), batch_size), desc="Sentiment Analysis"):
    batch = all_texts[i:i+batch_size]
    lbs, scs = sentiment_batch(batch)
    sent_labels.extend(lbs)
    sent_scores.extend(scs)

# 8. 写回 DataFrame 并保存
df['ML2_oracle1_label'] = sent_labels
df['ML2_oracle1_probability'] = sent_scores
df.to_csv(out_csv, index=False)
print(f"情感分析结果已保存至：{out_csv}")
