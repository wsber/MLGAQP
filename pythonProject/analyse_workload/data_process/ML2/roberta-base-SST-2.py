import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from scipy.special import softmax
from tqdm import tqdm

# —— 1. 模型路径或名称 ——
MODEL_NAME = "/home/wangshuo/resource/AIModels/NLP/TE/roberta-base-SST-2"

# —— 2. 手动加载 tokenizer 和 model ——
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model.half()  # 使用半精度加速
model.to(device).eval()

# —— 3. 从 config 中读取 id2label ——
#     DistilBERT 的 config 中自带 {"0":"NEGATIVE","1":"POSITIVE"}
id2label = model.config.id2label

# —— 4. （可选）预处理函数 ——
def preprocess(text: str) -> str:
    return text.strip()

# —— 5. 读取 CSV ——
datadir  = "/home/wangshuo/resource/datasets/parler_data/30W_valid_user/10-10-5"
in_csv   = f"{datadir}/comment_test.csv"
out_csv  = f"{datadir}/comment_test.csv"
df       = pd.read_csv(in_csv)
texts    = df['body'].fillna("").astype(str).tolist()

# —— 6. 批量推理函数 ——
def infer_batch(batch_texts):
    # 分词 + 编码
    inputs = tokenizer(
        [preprocess(t) for t in batch_texts],
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    ).to(device)

    # 推理
    with torch.no_grad():
        logits = model(**inputs).logits.cpu().numpy()

    # softmax → 概率
    probs = softmax(logits, axis=1)

    # 取最优类别及其概率
    pred_ids    = probs.argmax(axis=1)
    pred_scores = probs.max(axis=1)
    pred_labels = [id2label[i] for i in pred_ids]

    return pred_labels, pred_scores

# —— 7. 分批处理所有文本 ——
batch_size = 256
all_labels = []
all_scores = []

for i in tqdm(range(0, len(texts), batch_size), desc="DistilBERT 推理"):
    batch = texts[i : i + batch_size]
    lbs, scs = infer_batch(batch)
    all_labels.extend(lbs)
    all_scores.extend(scs)

# —— 8. 写回 CSV ——
df['ML2_proxy1_label'] = all_labels
df['ML2_proxy3_probability'] = all_scores
df.to_csv(out_csv, index=False, encoding="utf-8-sig")

print(f"roberta-base-SST-2 情感分析结果已保存到：{out_csv}")
