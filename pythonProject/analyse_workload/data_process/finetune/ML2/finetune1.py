import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from tqdm.auto import tqdm

# — 0. 参数设置 —
MODEL_TEACHER = "/home/wangshuo/resource/AIModels/NLP/TE/bert-large-uncased-sst2"
MODEL_STUDENT = "/home/wangshuo/resource/AIModels/NLP/TE/TinyBERT-4L-312D-SST-2"  # TinyBERT-4L-312D, 2-class head
CSV_PATH = "/home/wangshuo/resource/datasets/parler_data/30W_valid_user/10-10-5/comment_test.csv"
OUTPUT_DIR = "/home/wangshuo/resource/AIModels/Finetune/TE/distill_tinybert/"
BATCH_SIZE = 64
EPOCHS = 10
LR = 5e-5
TEMPERATURE = 2.0
ALPHA = 0.5  # 蒸馏损失 vs. 交叉熵损失的权重

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# — 1. 数据集定义 — 包含文本 和 “软标签” teacher_probs
class SST2DistillDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.texts = df["body"].fillna("").astype(str).tolist()
        # teacher 只给了正类概率 p; 构造 [neg_prob, pos_prob]
        p_pos = df["ML2_oracle1_probability"].values.astype(float)
        p_neg = 1.0 - p_pos
        self.teacher_probs = torch.tensor(
            list(zip(p_neg, p_pos)), dtype=torch.float32
        )
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["teacher_probs"] = self.teacher_probs[idx]
        return item


# — 2. 加载 tokenizer 和模型 —
tokenizer_t = AutoTokenizer.from_pretrained(MODEL_TEACHER)
teacher = AutoModelForSequenceClassification.from_pretrained(
    MODEL_TEACHER
).to(device)
teacher.eval()  # 只取过来的软标签，本例不在训练中调用 teacher.forward

tokenizer_s = AutoTokenizer.from_pretrained(MODEL_STUDENT)
student = AutoModelForSequenceClassification.from_pretrained(
    MODEL_STUDENT, num_labels=2
).to(device)

# — 3. 读取并切分数据 —
# — 3. 读取、抽样并切分数据 —
df_full = pd.read_csv(CSV_PATH)
# 随机抽取 20% 的样本
df = df_full.sample(frac=0.2, random_state=42).reset_index(drop=True)
dataset = SST2DistillDataset(df, tokenizer_s)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# — 4. 损失函数 & 优化器 & 学习率调度器 —
# 交叉熵（硬标签从 teacher_probs 二值化而来）
ce_loss = nn.CrossEntropyLoss()
# KL 散度用于软标签
kl_loss = nn.KLDivLoss(reduction="batchmean")
optimizer = AdamW(student.parameters(), lr=LR)
total_steps = len(loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps // 10,
    num_training_steps=total_steps,
)

# — 5. 训练循环 —
student.train()
for epoch in range(EPOCHS):
    loop = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
    running_loss = 0.0
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        teacher_p = batch["teacher_probs"].to(device)  # shape [B,2]

        # Student 前向
        outputs = student(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits_s = outputs.logits  # shape [B,2]

        # 1) 硬标签：从 teacher_p 选 max
        hard_labels = teacher_p.argmax(dim=1)

        # 2) 交叉熵 Loss
        loss_ce = ce_loss(logits_s, hard_labels)

        # 3) 软标签 Distillation Loss (温度缩放后 KLDiv)
        #    log_softmax(student_logits / T)
        log_p_s = nn.functional.log_softmax(logits_s / TEMPERATURE, dim=1)
        p_t = nn.functional.softmax(teacher_p / TEMPERATURE, dim=1)
        loss_kd = kl_loss(log_p_s, p_t) * (TEMPERATURE ** 2)

        # 4) 总损失
        loss = ALPHA * loss_kd + (1 - ALPHA) * loss_ce
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        loop.set_postfix(loss=running_loss / (loop.n + 1))

    print(f"→ Epoch {epoch + 1} avg loss: {running_loss / len(loader):.4f}")

# — 6. 保存微调后的 student 模型 —
student.save_pretrained(OUTPUT_DIR)
tokenizer_s.save_pretrained(OUTPUT_DIR)
print(f"✅ Distilled TinyBERT saved to {OUTPUT_DIR}")
