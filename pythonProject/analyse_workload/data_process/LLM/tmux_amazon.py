import os
import torch
import pandas as pd
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# —— 1. 读取并准备 prompts ——
# 假设 mini_meta.csv 在当前工作目录，且包含 'title' 和 'features' 两列
df = pd.read_csv("/home/wangshuo/resource/datasets/amazon_data/meta_kcore.csv")

# —— 2. 加载模型与分词器 ——
MODEL_DIR = "/home/wangshuo/resource/AIModels/NLP/deepseek/DeepSeek-R1-Distill-Qwen-14B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
tokenizer.padding_side = 'left'
torch.backends.cuda.matmul.allow_tf32 = True
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="auto", trust_remote_code=True)
model.eval()
model = torch.compile(model)
# 定义所有可能的分类
CATEGORIES = [
    "Furniture",
    "Home Decor",
    "Kitchenware & Dining",
    "Bedding",
    "Storage & Organization",
    "Cleaning & Household",
    "Gardening",
    "Bath",
    "Appliances"
]

prompts = []
for _, row in df.iterrows():
    title = row["title"]
    features = row["features"]
    prompt = (
        f"Title: {title}\n"
        f"Features: {features}\n\n"
        f"Which category does the above item belong to? "
        f"Choose one from: {', '.join(CATEGORIES)}.\n"
        f"Answer with exactly one category name, no explanations,no think."
    )
    prompts.append(prompt)
OUTPUT_CSV = "/home/wangshuo/resource/datasets/amazon_data/meta_kcore_with_preds.csv"
# —— 3. 定义答案解析函数 ——
def match_category(answer_line):
    cleaned = answer_line.strip()
    print('output: ', cleaned)
    return cleaned
    # 直接看是否和某个分类完全匹配
    # for cat in CATEGORIES:
    #     if cleaned.lower() == cat.lower():
    #         return cat
    # return None

# —— 4. 批量推理 & 写回 DataFrame ——
BATCH_SIZE = 8
row_idx = 0
results = []
for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Processing"):
    batch_prompts = prompts[i : i + BATCH_SIZE]
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )
    # 解析当前 batch 的结果
    batch_rows = []
    for j, output in enumerate(outputs):
        text = tokenizer.decode(output, skip_special_tokens=True)
        last_line = text.strip().splitlines()[-1]
        pred  = match_category(last_line)
         # 立刻打印这一行的结果
        row_idx = i + j
        print(f"Row {row_idx}: predicted_category = {pred}")
        batch_rows.append({
            "title": df.loc[row_idx, "title"],
            "features": df.loc[row_idx, "features"],
            "predicted_category": pred
        })
    # 将当前 batch 的结果写入 CSV，第一批写 header，后续 append
    mode = 'w' if i == 0 else 'a'
    header = True if i == 0 else False
    pd.DataFrame(batch_rows).to_csv(OUTPUT_CSV, index=False, mode=mode, header=header)
