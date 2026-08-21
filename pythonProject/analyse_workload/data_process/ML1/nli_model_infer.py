import time

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

# ————————————————
# 1. 配置部分
# ————————————————
# model_name = 'deberta-v3-base-binary'
model_dir = '/home/wangshuo/resource/AIModels/NLP/NLI/'
model_name = 'deberta-v3-xsmall-binary-epoch20'
proxy_model_dir_pre = '/home/wangshuo/resource/AIModels/Finetune/NLI/base/'

model_dir = proxy_model_dir_pre + model_name

# data_dir = '/home/wangshuo/resource/datasets/parler_data/30W_valid_user/10-10-5'
data_dir = '/home/wangshuo/resource/datasets/parler_data/dataset_three/csv_data'
file_name = 'post.csv'
input_csv = data_dir + f'/{file_name}'
output_csv = data_dir + f'/{file_name}'

THRESHOLD = 0.3
BATCH_SIZE = 1
topic = "I support Trump"

# ————————————————
# 2. 加载模型和 tokenizer
# ————————————————
print(f'loading {model_name}')
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model.half()  # 使用半精度
model.to(device)
model.eval()

# ————————————————
# 3. 定义批量推理函数，使用 “This text is about {topic}.”
# ————————————————
def check_nli_batch(topic, posts):
    """
    对一批 posts 做 NLI 推理，假设句为 "This text is about {topic}."
    返回：每条 post 的 entailment 概率（numpy array）
    """
    # 构造 hypothesis 列表
    hypotheses = [f"This text is about {topic}." for _ in posts]

    # 使用 text_pair，将 posts 作为 premise、hypotheses 作为 hypothesis
    encoding = tokenizer(
        posts,
        hypotheses,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    ).to(device)

    with torch.no_grad():
        logits = model(**encoding).logits

    # 只取 contradiction(0) 和 entailment(2) 的 logits
    ec_probs = logits[:, [0, 2]].softmax(dim=1)
    # 返回 entailment（index=1）的概率
    return ec_probs[:, 0].cpu().numpy()


# ————————————————
# 4. 读取数据并执行批量推理
# ————————————————
df = pd.read_csv(input_csv)
posts = df['body'].fillna("").astype(str).tolist()

results = []
batch_times = []
for i in tqdm(range(0, len(posts), BATCH_SIZE), desc="Processing Posts"):
    batch = posts[i:i + BATCH_SIZE]
    t0 = time.perf_counter()
    probs = check_nli_batch(topic, batch)
    t1 = time.perf_counter()
    batch_times.append(t1 - t0)
    results.extend(probs)

# ————————————————
# 5. 计算耗时与吞吐量
# ————————————————
min_time = min(batch_times)
max_time = max(batch_times)
avg_time = sum(batch_times) / len(batch_times)

# 吞吐量 = batch_size / 时间
max_throughput = 1 / min_time  # 最快时的最大吞吐
min_throughput = 1 / max_time  # 最慢时的最小吞吐
avg_throughput = 1 / avg_time  # 平均吞吐

print(f"每个 batch 推理耗时 (s)：最短={min_time:.4f}, 最长={max_time:.4f}, 平均={avg_time:.4f}")
print(f"吞吐量 (batch/s)：最大={max_throughput:.2f}, 最小={min_throughput:.2f}, 平均={avg_throughput:.2f}")

# ————————————————
# 6. 保存结果到 CSV
# ————————————————
df['ML1_proxy6b_probability'] = results
df.to_csv(output_csv, index=False)
print(f"推理结果已保存到 {output_csv}")
