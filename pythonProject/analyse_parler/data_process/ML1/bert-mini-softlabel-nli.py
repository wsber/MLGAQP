from transformers import AutoConfig
# #### 0.使用deberta-v2-xlarge-mnli对comment/post进行文本推理

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

# ————————————————
# 1. 配置部分
# ————————————————

model_dir = '/home/wangshuo/resource/AIModels/Finetune/bert-mini-softlabel-nli'
data_dir = '/home/wangshuo/resource/datasets/parler_data/dataset_one/csv_data'
input_csv = data_dir + '/post_t.csv'
output_csv = data_dir + '/post_t.csv'
# 先加载配置，明确 num_labels=2
config = AutoConfig.from_pretrained(model_dir, num_labels=2)

THRESHOLD = 0.3
BATCH_SIZE = 64
topic = "I support Trump"

# ————————————————
# 2. 加载模型和 tokenizer
# ————————————————
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir, config=config,
                                                           ignore_mismatched_sizes=True)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)
# model.half()
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
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**encoding).logits
    if model.config.num_labels == 3:
        # 3-way MNLI：取 [contradiction, entailment]
        ec = logits[:, [0, 2]]
        probs = ec.softmax(dim=1)  # [batch, 2]
        # 取第二个分量（原来 index=2，对应 entailment）
        return probs[:, 1].cpu().numpy()

    elif model.config.num_labels == 2:
        # 2-way 二分类：直接在两个 logits 上 softmax
        probs = logits.softmax(dim=1)  # [batch, 2]
        # 还是取第二个分量（index=1，对应 entailment）
        return probs[:, 1].cpu().numpy()

    else:
        raise ValueError(f"Unexpected num_labels={model.config.num_labels}")


# ————————————————
# 4. 读取数据并执行批量推理
# ————————————————
df = pd.read_csv(input_csv)
posts = df['body'].fillna("").astype(str).tolist()

results = []
for i in tqdm(range(0, len(posts), BATCH_SIZE), desc="Processing Posts"):
    batch = posts[i:i + BATCH_SIZE]
    probs = check_nli_batch(topic, batch)
    results.extend(probs)

# ————————————————
# 5. 保存结果到 CSV
# ————————————————
df['ML1_proxy4_probability'] = results
df['ML1_proxy4_label'] = df['ML1_proxy4_probability'].apply(
    lambda p: "ML1_proxy4_entailment" if p > THRESHOLD else "ML1_proxy4_not"
)
df.to_csv(output_csv, index=False)
print(f"推理结果已保存到 {output_csv}")
