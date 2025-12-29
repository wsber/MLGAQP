import os
import time
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm

# —————— 配置 ——————
# model_name = 'deberta-v3-xsmall-binary-epoch20'
# model_name = 'distilbert-base-uncased-finetuned-mnli'
# model_name = 'deberta-v3-base-binary'
model_name = 'bert-mini-binary-epoch20'
# MODEL_DIR = f"/home/wangshuo/resource/AIModels/NLP/NLI/{model_name}"  # 你的模型与 tokenizer 保存目录
MODEL_DIR = f"/home/wangshuo/resource/AIModels/Finetune/NLI/base/{model_name}"  # 你的模型与 tokenizer 保存目录
INPUT_CSV = "/home/wangshuo/resource/datasets/parler_data/dataset_one/csv_data/post_t.csv"  # 待推理的文件
OUTPUT_CSV = "/home/wangshuo/resource/datasets/parler_data/dataset_one/csv_data/post_t.csv"  # 覆盖写回
BATCH_SIZE = 32
MAX_LEN = 256
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# —————— 加载模型 & tokenizer ——————
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
model.half()
model.eval()

# —————— 读取数据 ——————
df = pd.read_csv(INPUT_CSV)
posts = df['body'].fillna("").astype(str).tolist()


# —————— 推理函数 ——————
def infer_entail_over_contra(posts_batch):
    enc = tokenizer(
        posts_batch,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        logits = model(**enc).logits  # [B,2]
    # 只取 contra(0) 和 entail(1) 两列，softmax 归一化
    two_logits = logits[:, [0, 1]]
    probs = two_logits.softmax(dim=1)
    # 返回 entail 概率
    return probs[:, 1].cpu().numpy()


# —————— 批量推理 ——————
proxy_probs = []
batch_times = []
for i in tqdm(range(0, len(posts), BATCH_SIZE), desc="Inferencing"):
    batch = posts[i: i + BATCH_SIZE]
    t0 = time.perf_counter()
    proxy_probs.extend(infer_entail_over_contra(batch))
    t1 = time.perf_counter()
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
# —————— 写回结果 ——————
df['ML1_proxy1b_probability'] = proxy_probs
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ 推理完成，结果保存到 {OUTPUT_CSV}")
