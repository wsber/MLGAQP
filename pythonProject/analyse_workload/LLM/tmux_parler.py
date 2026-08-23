import os
import torch
import pandas as pd
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# data_dir = '/home/wangshuo/resource/datasets/IOGS/many_predicates/independent/dataset_3/valid_data/'
# data_dir = '/home/wangshuo/resource/datasets/IOGS/many_predicates/independent/dataset_4/'
data_dir = '/home/wangshuo/resource/datasets/parler_data/30W_valid_user/10-10-5/'

# 定义答案解析函数，匹配答案行格式（严格要求： "yes,0.85" 或 "no,0.85"）
def match_answer_ex(answer_line):
    # 清理前后空格，并剔除前缀 "Answer:"（若存在）
    cleaned = answer_line.strip()
    if cleaned.lower().startswith("answer:"):
        cleaned = cleaned[len("answer:"):].strip()
    print(cleaned)
    # 使用严格的正则表达式匹配格式
    pattern = r"^(?i)(yes|no)\s*,\s*([0-9]*\.[0-9]+)$"
    match = re.search(pattern, cleaned)
    if match:
        return {
            "oracle_label": match.group(1).lower(),
            "oracle_confidence_probability": float(match.group(2)),
            "answer": cleaned
        }
    return {"oracle_label": None, "oracle_confidence_probability": None, "answer": cleaned}


# 加载分词器和模型
MODEL_DIR = "/home/wangshuo/resource/AIModels/NLP/deepseek/DeepSeek-R1-Distill-Qwen-14B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
tokenizer.padding_side = 'left'  # 关键修改

# 设备优化
torch.backends.cuda.matmul.allow_tf32 = True  # 允许 TF32 加速
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="auto", trust_remote_code=True)
model.half()
model.eval()
model = torch.compile(model)  # 进一步优化计算图
INPUT_CSV = data_dir + "post_ML1_albert_mnli.csv"      # 输入文件
OUTPUT_CSV = data_dir + "post_LLM.csv"    # 输出文件
# TOPIC = "Trump will win and save America"   # 主题
TOPIC = "I support Trump"   # 主题
BATCH_SIZE = 24                             # 批处理大小
START_ROW = 1  # 指定起始行号（不包括 header），例如：从第1000行开始读取
N = 30000         # 读取 N 行数据

# 读取 CSV 文件，仅从指定行开始读取 N 行，保留表头
# 注意：header 默认在第一行 (index 0)，因此 skiprows 从 1 开始跳过
df = pd.read_csv(INPUT_CSV, skiprows=range(1, START_ROW), nrows=N)
if "body" not in df.columns:
    raise ValueError("CSV 文件缺少 'body' 列")

# 构造每一行的提示文本，将 'body' 列作为输入文本
prompts = [
    f"Text: {text}.\n\n Given the topic '{TOPIC}', does the following text imply or infer this topic? Only answer 'yes' or 'no'. No explanation just the answer.\n"
    for text in df["body"].astype(str)
]

# 如果输出文件已存在，先删除以避免追加旧数据
if os.path.exists(OUTPUT_CSV):
    os.remove(OUTPUT_CSV)
# 批量处理：逐批推理并写入结果
results = []  # 存储所有结果
for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Processing"):
    batch_prompts = prompts[i:i + BATCH_SIZE]

    # 编码当前批次的提示文本
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    # 模型推理
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=600, do_sample=False, temperature=0.0,
                                 top_p=0.0, )

    batch_results = []
    # 对当前批次的每一条生成结果进行解码和后处理
    for j, output in enumerate(outputs):
        # 解码完整生成的文本
        full_text = tokenizer.decode(output, skip_special_tokens=True)
        # 分割文本为多行，并取最后一行（答案所在行）
        lines = full_text.strip().splitlines()
        if lines:
            answer_line = lines[-1].strip()
        else:
            answer_line = ""
        # 使用 match_answer_ex 函数解析答案行
        batch_results.append(match_answer_ex(answer_line))

    # 将当前批次的结果添加到对应的 DataFrame 行中
    batch_df = df.iloc[i:i + BATCH_SIZE].copy()
    batch_df["oracle_label"] = [r["oracle_label"] for r in batch_results]
    # batch_df["oracle_confidence_probability"] = [r["oracle_confidence_probability"] for r in batch_results]
    batch_df["answer"] = [r["answer"] for r in batch_results]

    # 追加写入 CSV 文件：第一个批次写入表头，其余批次追加写入
    if i == 0:
        batch_df.to_csv(OUTPUT_CSV, index=False, mode='w', header=True)
    else:
        batch_df.to_csv(OUTPUT_CSV, index=False, mode='a', header=False)

    results.extend(batch_results)

print(f"处理完成，结果已保存到 {OUTPUT_CSV}")
