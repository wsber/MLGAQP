import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from scipy.special import softmax
from tqdm import tqdm

# 1. 模型路径
MODEL_PATH = "/home/wangshuo/resource/AIModels/NLP/TE/bert-mini-finetuned-sst2"

# 2. 手动加载 tokenizer 和 model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
device    = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
datadir  = "/home/wangshuo/resource/datasets/parler_data/30W_valid_user/10-10-5"
in_csv   = f"{datadir}/comment_test.csv"
out_csv  = f"{datadir}/comment_test.csv"
df       = pd.read_csv(in_csv)
texts    = df['body'].fillna("").astype(str).tolist()

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

    # 6.3 softmax → 概率
    probs = softmax(logits, axis=1)

    # 6.4 拿到最优
    pred_ids   = probs.argmax(axis=1)
    pred_scores = probs.max(axis=1)
    pred_labels = [ id2label[i] for i in pred_ids ]

    return pred_labels, pred_scores

# 7. 遍历所有文本
batch_size   = 32
all_labels   = []
all_scores   = []

for i in tqdm(range(0, len(texts), batch_size), desc="BERT‑Mini 推理"):
    batch = texts[i : i + batch_size]
    lbs, scs = infer_batch(batch)
    all_labels.extend(lbs)
    all_scores.extend(scs)

# 8. 写回 CSV
df['ML2_proxy2_label'] = all_labels
df['ML2_proxy2_probability'] = all_scores
df.to_csv(out_csv, index=False, encoding="utf-8-sig")

print(f"情感分析结果已保存到：{out_csv}")
