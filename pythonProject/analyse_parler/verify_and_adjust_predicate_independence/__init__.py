import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm  # 导入 tqdm

# 设置模型文件路径
model_dir = '/home/wangshuo/resource/AIModels/NLP/bart-large-mnli'

# 加载模型和 tokenizer（从本地加载）
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# 将模型转移到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(device)

# 静态赋值给 df
data = {
    'user_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'bio': [
        "I am a doctor working in a hospital and I specialize in cardiology.",
        "I teach mathematics at a high school and love helping students learn.",
        "I work as a software engineer in a tech company.",
        "I am a professional soccer player and train every day.",
        "I am an artist who paints landscapes and exhibits my work in galleries.",
        "I work as a lawyer and deal with criminal defense cases.",
        "I am a police officer and work for public safety.",
        "I serve in the military as a lieutenant.",
        "I manage financial portfolios for a private equity firm.",
        "I am a teacher in a primary school and love working with young children."
    ]
}

df = pd.DataFrame(data)

# 获取所有的 "bio" 字段内容
all_bios = df['bio'].tolist()
print(f"数据加载完毕，共有 {len(all_bios)} 个用户")

# 职业分类列表
professions = [
    "Healthcare profession", "Public service profession", "Public administration and government profession",
    "Arts profession", "Legal profession", "Education profession", "Business and finance profession",
    "Military profession", "Sports and fitness profession"
]


# 定义推理函数
def check_nli_batch(topic, batch_bios):
    """
    对每一批次的 bio 进行 NLI 推理，判断其是否与给定的职业类别匹配
    """
    # 创建每个输入样本的 "bio + profession" 对
    batch_inputs = [f"{bio} [SEP] {topic}" for bio in batch_bios]

    # 编码批量输入
    encoding = tokenizer(batch_inputs, padding=True, truncation=True, return_tensors="pt")
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # 模型推理
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # 选择 "entailment" 和 "contradiction" 的 logits
        entail_contradiction_logits = logits[:, [0, 2]]

        # 计算 softmax 概率
        probs = entail_contradiction_logits.softmax(dim=1)

        # 获取 "entailment" 类别的概率（标签为真时的概率）
        prob_label_is_true = probs[:, 1].cpu().numpy()

        return prob_label_is_true


# 假设每批次处理 32 个文本
batch_size = 32
results = []

# 使用 tqdm 显示进度条
for i in tqdm(range(0, len(all_bios), batch_size), desc="Processing Bios"):
    batch_bios = all_bios[i:i + batch_size]
    # 为每个职业分类进行推理
    profession_probs = []
    for profession in professions:
        probabilities = check_nli_batch(profession, batch_bios)
        profession_probs.append(probabilities)

    # 计算每个职业类别的最大概率，并选择相应的职业类别
    for i in range(len(batch_bios)):
        max_prob_idx = max(range(len(profession_probs)), key=lambda x: profession_probs[x][i])
        if profession_probs[max_prob_idx][i] > 0.5:
            # 若最大概率大于0.5，则选择对应职业
            results.append(professions[max_prob_idx])
        else:
            # 否则分类为其他职业
            results.append("Other profession")

# 将结果添加到 DataFrame
df['profession'] = results

# 输出 DataFrame 验证
print(df)

# # 保存包含新列的 CSV 文件
# output_csv_file = '/home/wangshuo/resource/datasets/workload_20w/users_with_profession.csv'
# df.to_csv(output_csv_file, index=False)
#
# print(f"推理结果已保存到 {output_csv_file}")
