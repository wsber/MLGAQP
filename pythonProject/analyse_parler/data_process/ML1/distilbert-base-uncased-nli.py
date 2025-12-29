import pandas as pd
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm.auto import tqdm

# ————————————————
# 1. 配置部分
# ————————————————

# model_dir = '/home/wangshuo/resource/AIModels/Finetune/distil-proxy/electra_entailment_final'
model_dir = '/home/wangshuo/resource/AIModels/NLP/NLI/distilbert-base-uncased-finetuned-mnli'
data_dir = '/home/wangshuo/resource/datasets/parler_data/dataset_one/csv_data'
input_csv = f'{data_dir}/post_t.csv'
output_csv = f'{data_dir}/post_t.csv'

# 先加载配置，明确 num_labels=2
config = AutoConfig.from_pretrained(model_dir, num_labels=2)

# 推理阈值与批次大小
THRESHOLD = 0.5
BATCH_SIZE = 64
topic = "I support Trump"

# ————————————————
# 2. 加载模型和 tokenizer
# ————————————————

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(
    model_dir,
    config=config,
    ignore_mismatched_sizes=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ————————————————
# 3. 定义批量推理函数
# ————————————————

def proxy_entailment_batch(topic, texts):
    """
    对一批文本做代理模型推理，返回 entailment 概率
    """
    hypotheses = [topic] * len(texts)
    encoding = tokenizer(
        texts,
        hypotheses,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**encoding).logits

    if config.num_labels == 2:
        probs = logits.softmax(dim=1)[:, 1]
    else:
        # 假如是三分类, 则取 index=2
        ec = logits[:, [0, 2]]
        probs = ec.softmax(dim=1)[:, 1]

    return probs.cpu().numpy()

# ————————————————
# 4. 批量读取并推理
# ————————————————

df = pd.read_csv(input_csv)
# 确保 body 列无缺失
texts = df['body'].fillna('').astype(str).tolist()

results = []
for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Processing Texts"):
    batch_texts = texts[i: i + BATCH_SIZE]
    batch_probs = proxy_entailment_batch(topic, batch_texts)
    results.extend(batch_probs)

# ————————————————
# 5. 保存结果到 CSV
# ————————————————

df['ML1_proxy3_probability'] = results


df.to_csv(output_csv, index=False)
print(f"推理完成，结果已保存至 {output_csv}")
