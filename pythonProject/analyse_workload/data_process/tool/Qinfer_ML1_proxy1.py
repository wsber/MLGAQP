import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import deepspeed
from tqdm import tqdm

# —— 1. 模型路径 & 设备 ——
model_dir = '/home/wangshuo/resource/AIModels/NLP/distilbart-mnli-12-6'
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# —— 2. 加载 tokenizer 和原始模型 ——
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model     = AutoModelForSequenceClassification.from_pretrained(model_dir)

# —— 3. 半精度（FP16）并转到 GPU ——
model.half()
model.to(device)
model.eval()

# —— 4. DeepSpeed-Inference 注入 ——
#    安装 bitsandbytes / deepspeed 后使用：
model = deepspeed.init_inference(
    model,                                  # 原始模型
    mp_size=1,                              # 单卡推理
    # dtype=torch.float16,                    # 推理时用半精度
    replace_method="auto",                  # 自动替换支持的层
    replace_with_kernel_inject=True         # 注入 DeepSpeed-kernels
)

# —— 5. 读取数据 ——
datadir  = '/home/wangshuo/resource/datasets/parler_data/30W_valid_user/10-10-5'
in_csv   = f"{datadir}/post_ML1.csv"
df       = pd.read_csv(in_csv)
all_posts = df['body'].fillna("").astype(str).tolist()
print(f"加载完毕，共 {len(all_posts)} 条帖子")

# —— 6. 定义推理函数 ——
def check_nli_batch(topic, batch_posts):
    texts = [f"{post} [SEP] {topic}" for post in batch_posts]
    enc = tokenizer(texts, padding=True, truncation=True,
                    return_tensors="pt", max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs.logits
        # 只取 “entailment vs contradiction”
        entail_contra = logits[:, [0, 2]]
        probs = entail_contra.softmax(dim=1)
        return probs[:, 1].cpu().numpy()

# —— 7. 批量推理 ——
batch_size = 32
results = []
for i in tqdm(range(0, len(all_posts), batch_size), desc="DeepSpeed 推理"):
    batch = all_posts[i:i+batch_size]
    probs = check_nli_batch("I support Trump", batch)
    results.extend(probs)

# —— 8. 保存结果 ——
df['ML1_proxy1_probability'] = results
df['ML1_proxy1_label'] = df['ML1_proxy1_probability'].apply(
    lambda x: "ML1_proxy1_entailment" if x > 0.5 else "ML1_proxy1_not"
)
out_csv = f"{datadir}/post_ML1_deepspeed.csv"
df.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"推理结果已保存到：{out_csv}")
