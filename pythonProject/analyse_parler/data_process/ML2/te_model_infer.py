import time

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from scipy.special import softmax
from tqdm import tqdm

# 1. 模型路径
model_name = 'distill_oracle2_roberta-base-SST-2_epoch5'
# model_name = 'roberta-large-sst2'
proxy_model_dir_pre = '/home/wangshuo/resource/AIModels/Finetune/TE/sst2/'
model_dir_pre = '/home/wangshuo/resource/AIModels/NLP/TE/'
MODEL_PATH = proxy_model_dir_pre + model_name
# MODEL_PATH = model_dir_pre + model_name

# 2. 手动加载 tokenizer 和 model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.half()
model.to(device).eval()

# 3. 手动写标签映射（sst2 两分类）
id2label = {
    0: "negative",
    1: "positive"
}


# 4. （可选）预处理函数
def preprocess(text: str) -> str:
    # 这里 SST-2 不需要特殊替换，简单 strip 即可
    return text.strip()


# 5. 读取 CSV
# datadir = "/home/wangshuo/resource/datasets/parler_data/30W_valid_user/10-10-5"
datadir = "/home/wangshuo/resource/datasets/parler_data/dataset_test/csv_data"
file_name = "comment_new.csv"
in_csv = f"{datadir}/{file_name}"
out_csv = f"{datadir}/{file_name}"
df = pd.read_csv(in_csv)
texts = df['body'].fillna("").astype(str).tolist()


# 6. 批量推理函数
def infer_batch(batch_texts):
    # 6.1 预处理 + 分词
    inputs = tokenizer(
        [preprocess(t) for t in batch_texts],
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    ).to(device)

    # 6.2 模型推理
    with torch.no_grad():
        logits = model(**inputs).logits.cpu().numpy()

    probs = softmax(logits, axis=1)
    # 返回每条文本的 positive 类概率（索引 1）
    return probs[:, 1]


# 7. 遍历所有文本
batch_size = 32
all_scores = []
batch_times = []
for i in tqdm(range(0, len(texts), batch_size), desc=f"{model_name} 推理"):
    batch = texts[i: i + batch_size]
    t0 = time.perf_counter()
    scs = infer_batch(batch)
    t1 = time.perf_counter()
    all_scores.extend(scs.tolist())
    batch_times.append(t1 - t0)

min_time = min(batch_times)
max_time = max(batch_times)
avg_time = sum(batch_times) / len(batch_times)

# 吞吐量 = batch_size / 时间
max_throughput = 1 / min_time  # 最快时的最大吞吐
min_throughput = 1 / max_time  # 最慢时的最小吞吐
avg_throughput = 1 / avg_time  # 平均吞吐

print(f"每个 batch 推理耗时 (s)：最短={min_time:.4f}, 最长={max_time:.4f}, 平均={avg_time:.4f}")
print(f"吞吐量 (batch/s)：最大={max_throughput:.2f}, 最小={min_throughput:.2f}, 平均={avg_throughput:.2f}")

# 8. 写回 CSV
# model_label = 'proxy2d2'
model_label = 'proxy1d1'
# model_label = 'proxy1'
df[f'ML2_{model_label}_probability'] = all_scores
df.to_csv(out_csv, index=False, encoding="utf-8-sig")

print(f"情感分析结果已保存到：{out_csv}")
