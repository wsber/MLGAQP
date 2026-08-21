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

# ——————————————
# 0. 全局超参与路径
# ——————————————
MODEL_STUDENT = "/home/wangshuo/resource/AIModels/Finetune/NLI/distilled_distilbart_mnli"
DATA_CSV = "/home/wangshuo/resource/datasets/parler_data/30W_valid_user/10-10-5/post_origin.csv"
OUTPUT_DIR = "/home/wangshuo/resource/AIModels/Finetune/NLI/distilled_distilbart_mnli_epoch10/"
BATCH_SIZE = 16
EPOCHS = 7
LR = 3e-5
TEMPERATURE = 2.0
ALPHA = 0.6  # soft vs hard loss 权重
MAX_LEN = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ——————————————
# 1. Dataset：读取 body, 构造 hypothesis, 加载教师三分类软标签
# ——————————————
class NLIDistillDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 512):
        self.premises = df["body"].fillna("").astype(str).tolist()
        # 固定 hypothesis
        self.hypotheses = ["This text is about I support Trump." for _ in self.premises]
        # 教师在 CSV 中已存的三分类概率列
        probs = df[[
            "ML1_oracle1_contra",
            "ML1_oracle1_neutral",
            "ML1_oracle1_entail"
        ]].values.astype(float)
        self.teacher_probs = torch.tensor(probs, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.premises)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.premises[idx],
            self.hypotheses[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["teacher_probs"] = self.teacher_probs[idx]
        return item


# ——————————————
# 2. 加载学生模型 & tokenizer
# ——————————————
tokenizer_s = AutoTokenizer.from_pretrained(MODEL_STUDENT)
student = AutoModelForSequenceClassification.from_pretrained(
    MODEL_STUDENT,
    num_labels=3,  # 三分类头
    ignore_mismatched_sizes=True  # in case original head !=3
).to(device)
student.train()

# ——————————————
# 3. 构建 DataLoader
# ——————————————
df_full = pd.read_csv(DATA_CSV)
# 可选：抽样加速实验
df = df_full.sample(frac=0.2, random_state=42).reset_index(drop=True)
dataset = NLIDistillDataset(df, tokenizer_s, max_len=MAX_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# ——————————————
# 4. 损失 / 优化器 / Scheduler
# ——————————————
ce_loss = nn.CrossEntropyLoss()
kl_loss = nn.KLDivLoss(reduction="batchmean")
optimizer = AdamW(student.parameters(), lr=LR)
total_steps = len(loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps // 10,
    num_training_steps=total_steps
)

# ——————————————
# 5. 训练循环：混合软硬标签
# ——————————————
for epoch in range(EPOCHS):
    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
    epoch_loss = 0.0

    for batch in pbar:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        teacher_p = batch["teacher_probs"].to(device)  # [B,3]

        # 学生前向
        outputs = student(input_ids=input_ids, attention_mask=attention_mask)
        logits_s = outputs.logits  # [B,3]

        # —— 1) 硬标签 CE Loss ——
        hard_labels = teacher_p.argmax(dim=1)  # [B]
        loss_ce = ce_loss(logits_s, hard_labels)

        # —— 2) 软标签 KD Loss ——
        log_p_s = nn.functional.log_softmax(logits_s / TEMPERATURE, dim=1)
        p_t = nn.functional.softmax(teacher_p / TEMPERATURE, dim=1)
        loss_kd = kl_loss(log_p_s, p_t) * (TEMPERATURE ** 2)

        # —— 3) 混合总损失 ——
        loss = ALPHA * loss_kd + (1 - ALPHA) * loss_ce
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        pbar.set_postfix(avg_loss=epoch_loss / (pbar.n + 1))

    print(f"→ Epoch {epoch + 1} avg loss: {epoch_loss / len(loader):.4f}")

# ——————————————
# 6. 保存微调好的学生模型
# ——————————————
student.save_pretrained(OUTPUT_DIR)
tokenizer_s.save_pretrained(OUTPUT_DIR)
print(f"✅ Distilled model saved to {OUTPUT_DIR}")
