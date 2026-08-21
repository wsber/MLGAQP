import os
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import deepspeed
from tqdm.auto import tqdm

# ——————————————
# 0. 超参 & 路径设置
# ——————————————
BATCH_SIZE = 16  # 增加 batch，用梯度累积控制显存
ACCUM_STEPS = 2
EPOCHS = 5
LR = 3e-5
TEMPERATURE = 2.0
ALPHA = 0.7
MAX_LEN = 512

MODEL_NAME = 'deberta-base-mnli'
MODEL_STUDENT = f'/home/wangshuo/resource/AIModels/NLP/NLI/{MODEL_NAME}'
DATA_CSV = '/home/wangshuo/resource/datasets/parler_data/30W_valid_user/10-10-5/post_origin.csv'
OUTPUT_DIR = f'/home/wangshuo/resource/AIModels/Finetune/NLI/proxy_{MODEL_NAME}_deepspeed/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ——————————————
# 1. DeepSpeed config
# ——————————————
ds_config = {
    "train_batch_size": BATCH_SIZE * ACCUM_STEPS,
    "train_micro_batch_size_per_gpu": BATCH_SIZE,
    "gradient_accumulation_steps": ACCUM_STEPS,
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": LR, "betas": [0.9, 0.999], "eps": 1e-8}
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {"warmup_min_lr": 0, "warmup_max_lr": LR, "warmup_num_steps": 100}
    },
    "fp16": {"enabled": True},
    "zero_optimization": {"stage": 2},
}
with open(os.path.join(OUTPUT_DIR, 'ds_config.json'), 'w') as fp:
    json.dump(ds_config, fp, indent=2)


# ——————————————
# 2. Dataset 定义
# ——————————————
class NLIDistillDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 512):
        self.premises = df['body'].fillna('').astype(str).tolist()
        self.hypotheses = ["This text is about I support Trump." for _ in self.premises]
        probs = df[['ML1_oracle1_contra', 'ML1_oracle1_neutral', 'ML1_oracle1_entail']].values
        self.teacher_probs = torch.tensor(probs, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.premises)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.premises[idx],
            self.hypotheses[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['teacher_probs'] = self.teacher_probs[idx]
        return item


# ——————————————
# 3. 加载模型 & tokenizer
# ——————————————
tokenizer = AutoTokenizer.from_pretrained(MODEL_STUDENT)
student = AutoModelForSequenceClassification.from_pretrained(
    MODEL_STUDENT, num_labels=3, ignore_mismatched_sizes=True
)

# DeepSpeed init
model_engine, optimizer, _, scheduler = deepspeed.initialize(
    model=student,
    config=os.path.join(OUTPUT_DIR, 'ds_config.json')
)

# ——————————————
# 4. DataLoader
# ——————————————
df = pd.read_csv(DATA_CSV).sample(frac=0.2, random_state=42)
dataset = NLIDistillDataset(df, tokenizer, max_len=MAX_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# 损失函数
ce_loss = nn.CrossEntropyLoss()
kl_loss = nn.KLDivLoss(reduction='batchmean')

# ——————————————
# 5. 训练循环
# ——————————————
for epoch in range(EPOCHS):
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(model_engine.local_rank)
        attention_mask = batch['attention_mask'].to(model_engine.local_rank)
        teacher_p = batch['teacher_probs'].to(model_engine.local_rank)

        logits = model_engine(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits
        hard_labels = teacher_p.argmax(dim=1)
        loss_ce = ce_loss(logits, hard_labels)
        log_p_s = nn.functional.log_softmax(logits / TEMPERATURE, dim=1)
        p_t = nn.functional.softmax(teacher_p / TEMPERATURE, dim=1)
        loss_kd = kl_loss(log_p_s, p_t) * (TEMPERATURE ** 2)
        loss = ALPHA * loss_kd + (1 - ALPHA) * loss_ce

        model_engine.backward(loss)
        model_engine.step()
        total_loss += loss.item()
        pbar.set_postfix(avg_loss=total_loss / (step + 1))

# ——————————————
# 6. 保存模型
# ——————————————
model_engine.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Model and tokenizer saved to {OUTPUT_DIR}")
