# #### 0.使用deberta-v2-xlarge-mnli对comment/post进行文本推理
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

# ————————————————
# 1. 配置部分
# ————————————————
model_dir = '/home/wangshuo/resource/AIModels/NLP/NLI/deberta-v2-xxlarge-mnli'
data_dir = '/home/wangshuo/resource/datasets/parler_data/dataset_one/csv_data'
input_csv = data_dir + '/post_t.csv'
output_csv = data_dir + '/post_t.csv'

THRESHOLD = 0.3
BATCH_SIZE = 32
topic = "I support Trump"

# ————————————————
# 2. 加载模型和 tokenizer
# ————————————————
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model.to(device)
# model.half()
model.eval()
print(model.config.id2label)

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
        max_length=512,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**encoding).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs


# ————————————————
# 4. 读取数据并执行批量推理
# ————————————————
df = pd.read_csv(input_csv)
posts = df['body'].fillna("").astype(str).tolist()

all_probs = []
for i in tqdm(range(0, len(posts), BATCH_SIZE), desc="Processing Posts"):
    batch = posts[i:i + BATCH_SIZE]
    probs = check_nli_batch(topic, batch)
    all_probs.append(probs)

# 合并为 (N,3)
all_probs = np.vstack(all_probs)
# ————————————————
# 5. 保存结果到 CSV
# ————————————————
# 三分类概率
# df['ML1_oracle2_contra']  = all_probs[:, 0]
# df['ML1_oracle2_neutral'] = all_probs[:, 1]
# df['ML1_oracle2_entail']  = all_probs[:, 2]

# 二元条件概率：只考虑 contra & entail 两项，归一化后的 entail 概率
contra_plus_entail = all_probs[:, 0] + all_probs[:, 2]
# 避免除以零
df['ML1_oracle2_probability'] = np.where(
    contra_plus_entail > 0,
    all_probs[:, 2] / contra_plus_entail,
    0.0
)
df.to_csv(output_csv, index=False)
print(f"推理结果已保存到 {output_csv}")
