import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from tqdm.auto import tqdm

# ——————————————
# 0. 全局超参 & 路径
# ——————————————
BATCH_SIZE = 16
EPOCHS = 4
LR = 4e-5
TEMPERATURE = 2.0
ALPHA = 0.8  # soft vs hard loss 权重
MAX_LEN = 512

model_name = 'DeBERTa-v3-base-mnli-fever-anli'
proxy_model_dir = '/home/wangshuo/resource/AIModels/Finetune/NLI/'
model_dir = '/home/wangshuo/resource/AIModels/NLP/NLI/'

MODEL_STUDENT = model_dir + model_name
DATA_CSV = "/home/wangshuo/resource/datasets/parler_data/30W_valid_user/10-10-5/post_origin.csv"
OUTPUT_DIR = proxy_model_dir + f"proxy_{model_name}_epoch{EPOCHS}_reverse/"

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ——————————————
# 1. Dataset
# ——————————————
class NLIDistillDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 512):
        self.premises = df["body"].fillna("").astype(str).tolist()
        self.hypotheses = ["This text is about I support Trump."] * len(self.premises)
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
# 2. 加载模型 & tokenizer
# ——————————————
tokenizer_s = AutoTokenizer.from_pretrained(MODEL_STUDENT)
student = AutoModelForSequenceClassification.from_pretrained(
    MODEL_STUDENT,
    num_labels=3,
    ignore_mismatched_sizes=True
).to(device)

# ——————————————
# 3. 准备数据：读取、切分、DataLoader
# ——————————————
df_full = pd.read_csv(DATA_CSV)

# 3.1 抽样加速实验（可选）
df_sample = df_full.sample(frac=0.2, random_state=42).reset_index(drop=True)

# 3.2 划分 train / val（80% / 20%）
n_total = len(df_sample)
n_val = int(0.1 * n_total)
n_train = n_total - n_val
df_train, df_val = random_split(df_sample, [n_train, n_val], generator=torch.Generator().manual_seed(42))

# 3.3 DataLoader
train_ds = NLIDistillDataset(pd.DataFrame(df_train.dataset.iloc[df_train.indices]), tokenizer_s, max_len=MAX_LEN)
val_ds = NLIDistillDataset(pd.DataFrame(df_val.dataset.iloc[df_val.indices]), tokenizer_s, max_len=MAX_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ——————————————
# 4. 损失 / 优化器 / Scheduler
# ——————————————
ce_loss = nn.CrossEntropyLoss()
kl_loss = nn.KLDivLoss(reduction="batchmean")
optimizer = AdamW(student.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps // 10,
    num_training_steps=total_steps
)

# ——————————————
# 5. 训练 + 验证循环
# ——————————————
for epoch in range(1, EPOCHS + 1):
    # ——— 5.1 训练
    student.train()
    train_loss = 0.0
    pbar = tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{EPOCHS}")
    for batch in pbar:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        teacher_p = batch["teacher_probs"].to(device)

        outputs = student(input_ids=input_ids, attention_mask=attention_mask)
        logits_s = outputs.logits

        # 硬标签 CE
        hard_labels = teacher_p.argmax(dim=1)
        loss_ce = ce_loss(logits_s, hard_labels)

        # 软标签 KD
        log_p_s = nn.functional.log_softmax(logits_s / TEMPERATURE, dim=1)
        p_t = nn.functional.softmax(teacher_p / TEMPERATURE, dim=1)
        loss_kd = kl_loss(log_p_s, p_t) * (TEMPERATURE ** 2)

        loss = ALPHA * loss_kd + (1 - ALPHA) * loss_ce
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        pbar.set_postfix(avg_loss=train_loss / (pbar.n + 1))

    # ——— 5.2 验证
    student.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            teacher_p = batch["teacher_probs"].to(device)

            logits_s = student(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = logits_s.argmax(dim=1)
            labels = teacher_p.argmax(dim=1)  # 用教师硬标签当“真值”

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total if total > 0 else 0.0
    print(f"→ Epoch {epoch} Train Loss: {train_loss / len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

# ——————————————
# 6. 保存模型
# ——————————————
student.save_pretrained(OUTPUT_DIR)
tokenizer_s.save_pretrained(OUTPUT_DIR)
print(f"✅ Model saved to {OUTPUT_DIR}")
