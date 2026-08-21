import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

# ————————————————
# 1. 配置部分
# ————————————————
model_name = 'distilled_bert-mini-finetuned-mnli50'
model_dir = f'/home/wangshuo/resource/AIModels/Finetune/NLI/{model_name}'
data_dir = '/home/wangshuo/resource/datasets/parler_data/30W_valid_user/10-10-5'
input_csv = data_dir + '/post_origin.csv'
output_csv = data_dir + '/post_origin.csv'

THRESHOLD = 0.3
BATCH_SIZE = 32
topic = "I support Trump"

# ————————————————
# 2. 加载模型和 tokenizer
# ————————————————
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


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
    return ec_probs[:, 1].cpu().numpy()


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
df['ML1_proxy2ff_probability'] = results
df.to_csv(output_csv, index=False)
print(f"推理结果已保存到 {output_csv}")
