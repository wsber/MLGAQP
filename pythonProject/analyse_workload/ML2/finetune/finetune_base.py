import os
import random
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


def main():
    # 0. 超参 & 路径
    SEED = 42
    BATCH_SIZE = 16
    EPOCHS = 20
    LR = 4e-5
    MAX_LEN = 256
    model_name = 'TinyBERT_General_4L_312D'
    MODEL_BACKBONE = f"/home/wangshuo/resource/AIModels/NLP/base-uncased/{model_name}"
    OUTPUT_DIR = f"/home/wangshuo/resource/AIModels/Finetune/TE/base/{model_name}-binary-epoch{EPOCHS}/"
    DATA_CSV = "/home/wangshuo/resource/datasets/parler_data/30W_valid_user/10-10-5/comment_test.csv"

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 固定随机
    torch.manual_seed(SEED)
    random.seed(SEED)
    if device.type.startswith("cuda:3"):
        torch.cuda.manual_seed_all(SEED)

    # Dataset
    class BinaryNLIDataset(Dataset):
        def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = MAX_LEN):
            self.texts = df["body"].fillna("").astype(str).tolist()
            # 只取 contra 和 entail，并归一化
            p_pos = df["ML2_oracle1_probability"].values.astype(float)
            p_neg = 1.0 - p_pos
            # two-class soft labels: [p_contra, p_entail]
            self.soft_labels = torch.tensor(
                list(zip(p_neg, p_pos)), dtype=torch.float32
            )
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = self.tokenizer(
                self.texts[idx],
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            )
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item["soft_labels"] = self.soft_labels[idx]
            return item

    # 加载模型 & tokenizer（二分类）
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BACKBONE)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_BACKBONE,
        num_labels=2,
        ignore_mismatched_sizes=True
    ).to(device)

    # 数据准备
    df = pd.read_csv(DATA_CSV).sample(frac=0.2, random_state=SEED).reset_index(drop=True)
    n_val = int(0.1 * len(df))
    n_train = len(df) - n_val
    df_train, df_val = random_split(
        df, [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
    )
    train_ds = BinaryNLIDataset(
        pd.DataFrame(df_train.dataset.iloc[df_train.indices]), tokenizer
    )
    val_ds = BinaryNLIDataset(
        pd.DataFrame(df_val.dataset.iloc[df_val.indices]), tokenizer
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 损失 / 优化 / 调度
    ce = nn.CrossEntropyLoss()
    kl = nn.KLDivLoss(reduction="batchmean")
    opt = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    sched = get_linear_schedule_with_warmup(opt, total_steps//10, total_steps)

    # 训练循环
    best_val = float('inf')
    ALPHA, T = 0.8, 2.0

    for epoch in range(1, EPOCHS+1):
        model.train(); running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for i, batch in enumerate(pbar, 1):
            opt.zero_grad()
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['soft_labels'].to(device)

            logits = model(input_ids=ids, attention_mask=mask).logits
            # CE uses hard labels
            hard = labels.argmax(dim=1)
            loss_ce = ce(logits, hard)
            # KD on two classes
            logp = nn.functional.log_softmax(logits/ T, dim=1)
            loss_kd = kl(logp, labels)*(T*T)
            loss = ALPHA*loss_kd + (1-ALPHA)*loss_ce

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

            running += loss.item()
            pbar.set_postfix(loss=f"{running / i:.4f}")

        # 验证
        model.eval(); vloss=0.0; corr=0; tot=0
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labels = batch['soft_labels'].to(device)
                logits = model(input_ids=ids, attention_mask=mask).logits
                logp = nn.functional.log_softmax(logits/ T, dim=1)
                vloss += kl(logp, labels)*(T*T)
                pred = logits.argmax(dim=1)
                corr += (pred == labels.argmax(dim=1)).sum().item()
                tot += pred.size(0)
        vloss = vloss / len(val_loader)
        vacc = corr / tot
        print(f"Val Loss={vloss:.4f}, Val Acc={vacc:.4f}")

        if vloss < best_val:
            best_val = vloss
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"Best model saved, Val Loss={vloss:.4f}")

    # 保存 final
    model.save_pretrained(os.path.join(OUTPUT_DIR, 'final'))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, 'final'))
    print(f'model save to :{OUTPUT_DIR}')

if __name__ == '__main__':
    main()
