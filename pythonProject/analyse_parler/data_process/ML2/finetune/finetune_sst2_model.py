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

teacher_model_name = 'bert-large-uncased-sst2'
student_model_name = 'TinyBERT-4L-312D-SST-2'  # TinyBERT-4L-312D, 2-class head
dataset_name = 'comment'
oralce_label = 'oracle1'

# — 0. 参数设置 —
BATCH_SIZE = 64
EPOCHS = 15
LR = 4e-5
TEMPERATURE = 2.0
ALPHA = 0.8  # 蒸馏损失 vs. 交叉熵损失的权重
MODEL_TEACHER = f"/home/wangshuo/resource/AIModels/NLP/TE/{teacher_model_name}"
MODEL_STUDENT = f"/home/wangshuo/resource/AIModels/NLP/TE/{student_model_name}"  # TinyBERT-4L-312D, 2-class head
CSV_PATH = f"/home/wangshuo/resource/datasets/parler_data/30W_valid_user/10-10-5/{dataset_name}.csv"
OUTPUT_DIR = f"/home/wangshuo/resource/AIModels/Finetune/TE/sst2/distill_{oralce_label}_{student_model_name}_epoch{EPOCHS}/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 打开日志文件，写表头
log_path = os.path.join(OUTPUT_DIR, "log.txt")
with open(log_path, "w") as log_f:
    log_f.write("Epoch\tTrainLoss\tValLoss\tValAcc\n")


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
df = df_full.sample(frac=0.11, random_state=42).reset_index(drop=True)
dataset = SST2DistillDataset(df, tokenizer_s)
# 这里用 95% 训练，5% 验证
n_val = int(0.05 * len(dataset))
n_train = len(dataset) - n_val
train_ds, val_ds = torch.utils.data.random_split(
    dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,num_workers=0)

# — 4. 损失函数 & 优化器 & 学习率调度器 —
# 交叉熵（硬标签从 teacher_probs 二值化而来）
ce_loss = nn.CrossEntropyLoss()
# KL 散度用于软标签
kl_loss = nn.KLDivLoss(reduction="batchmean")
optimizer = AdamW(student.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps // 10,
    num_training_steps=total_steps,
)

# — 5. 训练循环 —
student.train()
for epoch in range(EPOCHS):
    loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
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
    train_loss = running_loss / len(train_loader)
    # —— 验证
    student.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} Val"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            teacher_p = batch["teacher_probs"].to(device)

            logits = student(input_ids=input_ids, attention_mask=attention_mask).logits
            log_p_s = nn.functional.log_softmax(logits / TEMPERATURE, dim=1)
            val_loss += kl_loss(log_p_s, teacher_p) * (TEMPERATURE ** 2)

            preds = logits.argmax(dim=1)
            correct += (preds == teacher_p.argmax(dim=1)).sum().item()
            total += preds.size(0)

    val_loss = (val_loss / len(val_loader)).item()
    val_acc = correct / total
    # — 5.3 将每个 epoch 的指标写入日志 —
    line = f"{epoch}\t{train_loss:.4f}\t{val_loss:.4f}\t{val_acc:.4f}\n"
    with open(log_path, "a") as log_f:
        log_f.write(line)
    print(f"Epoch {epoch}: TrainLoss={train_loss:.4f}, ValLoss={val_loss:.4f}, ValAcc={val_acc:.4f}")

# — 6. 保存微调后的 student 模型 —
student.save_pretrained(OUTPUT_DIR)
tokenizer_s.save_pretrained(OUTPUT_DIR)
print(f"✅ Distilled {student_model_name} saved to {OUTPUT_DIR}")
