import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

# 1. 配置
model_dir = '/home/wangshuo/resource/AIModels/NLP/NLI/deberta-v2-xxlarge-mnli'
input_csv = "/home/wangshuo/resource/datasets/parler_data/30W_valid_user/10-10-5/meta_classified.csv"
output_csv = './post_origin.csv'
THRESHOLD = 0.3
BATCH_SIZE = 32
topic = "This product is a household electrical appliance."

# 2. 初始化
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
# if device.type == "cuda":
#     model.half()
# model.eval()

# 3. 推理函数
def check_nli_batch(topic, posts):
    hypotheses = [f"This text is about {topic}." for _ in posts]
    encoding = tokenizer(
        posts, hypotheses,
        padding=True, truncation=True, max_length=512,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        logits = model(**encoding).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs

# 4. 读取数据
df = pd.read_csv(input_csv)
posts = df['input_text'].fillna("").astype(str).tolist()

# 5. 批量推理
all_probs = []
for i in tqdm(range(0, len(posts), BATCH_SIZE), desc="Processing input_text"):
    batch = posts[i:i + BATCH_SIZE]
    probs = check_nli_batch(topic, batch)
    all_probs.append(probs)
    torch.cuda.empty_cache()

all_probs = np.vstack(all_probs)

# 6. 保存结果
df['ML1_proxy1_contra']  = all_probs[:, 0]
df['ML1_proxy1_neutral'] = all_probs[:, 1]
df['ML1_proxy1_entail']  = all_probs[:, 2]

contra_plus_entail = all_probs[:, 0] + all_probs[:, 2]
df['ML1_proxy1_probability'] = np.where(
    contra_plus_entail > 0,
    all_probs[:, 2] / contra_plus_entail,
    0.0
)

df.to_csv(output_csv, index=False)
print(f"✅ 推理结果已保存到 {output_csv}")
