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
from torch.cuda.amp import autocast, GradScaler  # Use cuda.amp for default device_type

# Configuration
teacher_model_name = 'roberta-large-sst2'
student_model_name = 'roberta-base-SST-2'
dataset_name = 'comment'
oracle_label = 'oracle2'

BATCH_SIZE = 64
EPOCHS = 10
LR = 5e-5
TEMPERATURE = 2.0
ALPHA = 0.7  # weight for distillation vs. CE
MODEL_TEACHER = f"/home/wangshuo/resource/AIModels/NLP/TE/{teacher_model_name}"
MODEL_STUDENT = f"/home/wangshuo/resource/AIModels/NLP/TE/{student_model_name}"
CSV_PATH = f"/home/wangshuo/resource/datasets/parler_data/30W_valid_user/10-10-5/{dataset_name}.csv"
OUTPUT_DIR = f"/home/wangshuo/resource/AIModels/Finetune/TE/distil/distill_{oracle_label}_{student_model_name}_epoch{EPOCHS}/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Log setup
log_path = os.path.join(OUTPUT_DIR, "log.txt")
with open(log_path, "w") as log_f:
    log_f.write("Epoch\tTrainLoss\tValLoss\tValAcc\n")

# Dataset
torch.manual_seed(42)
class SST2DistillDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.texts = df["body"].fillna("").astype(str).tolist()
        p_pos = df[f"ML2_{oracle_label}_probability"].astype(float).values
        p_neg = 1.0 - p_pos
        self.teacher_probs = torch.tensor(list(zip(p_neg, p_pos)), dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['teacher_probs'] = self.teacher_probs[idx]
        return item

# Load models & tokenizers
tokenizer_s = AutoTokenizer.from_pretrained(MODEL_STUDENT)
student = AutoModelForSequenceClassification.from_pretrained(
    MODEL_STUDENT, num_labels=2
).to(device)
student.train()

# Prepare data
df_full = pd.read_csv(CSV_PATH)
df_sampled = df_full.sample(frac=0.11, random_state=42).reset_index(drop=True)
dataset = SST2DistillDataset(df_sampled, tokenizer_s)
n_val = int(0.05 * len(dataset))
n_train = len(dataset) - n_val
train_ds, val_ds = torch.utils.data.random_split(
    dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# Losses, optimizer, scheduler, scaler
ce_loss = nn.CrossEntropyLoss()
kl_loss = nn.KLDivLoss(reduction='batchmean')
optimizer = AdamW(student.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps // 10,
    num_training_steps=total_steps
)
scaler = GradScaler()

# Training loop with AMP
for epoch in range(1, EPOCHS + 1):
    running_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        teacher_p = batch['teacher_probs'].to(device)

        with autocast():  # Now device_type defaults to 'cuda'
            outputs = student(input_ids=input_ids, attention_mask=attention_mask)
            logits_s = outputs.logits
            hard_labels = teacher_p.argmax(dim=1)
            loss_ce = ce_loss(logits_s, hard_labels)
            log_p_s = nn.functional.log_softmax(logits_s / TEMPERATURE, dim=1)
            p_t = nn.functional.softmax(teacher_p / TEMPERATURE, dim=1)
            loss_kd = kl_loss(log_p_s, p_t) * (TEMPERATURE ** 2)
            loss = ALPHA * loss_kd + (1 - ALPHA) * loss_ce

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item()
        loop.set_postfix(loss=running_loss / (loop.n + 1))

    # Validation
    student.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            teacher_p = batch['teacher_probs'].to(device)
            with autocast():
                logits = student(input_ids=input_ids, attention_mask=attention_mask).logits
                log_p_s = nn.functional.log_softmax(logits / TEMPERATURE, dim=1)
                val_loss += kl_loss(log_p_s, nn.functional.softmax(teacher_p / TEMPERATURE, dim=1)) * (TEMPERATURE ** 2)
            preds = logits.argmax(dim=1)
            correct += (preds == teacher_p.argmax(dim=1)).sum().item()
            total += preds.size(0)
    val_loss = (val_loss / len(val_loader)).item()
    val_acc = correct / total
    with open(log_path, 'a') as log_f:
        log_f.write(f"{epoch}\t{running_loss/len(train_loader):.4f}\t{val_loss:.4f}\t{val_acc:.4f}\n")
    print(f"Epoch {epoch}: TrainLoss={running_loss/len(train_loader):.4f}, ValLoss={val_loss:.4f}, ValAcc={val_acc:.4f}")

# Save student
student.save_pretrained(OUTPUT_DIR)
tokenizer_s.save_pretrained(OUTPUT_DIR)
print(f"✅ Distilled model saved to {OUTPUT_DIR}")
