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
    # ——————————————
    # 0. 全局超参 & 路径
    # ——————————————
    SEED           = 42
    BATCH_SIZE     = 32
    EPOCHS         = 8
    LR             = 3e-5
    MAX_LEN        = 256
    MODEL_BACKBONE = "/home/wangshuo/resource/AIModels/NLP/base-uncased/deberta-v3-base"
    OUTPUT_DIR     = "/home/wangshuo/resource/AIModels/Finetune/NLI/deberta-v3-base-p/"
    DATA_CSV       = "/home/wangshuo/resource/datasets/parler_data/30W_valid_user/10-10-5/post_origin.csv"
    TOPIC          = "I support Trump"

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 固定随机种子
    torch.manual_seed(SEED)
    random.seed(SEED)
    if device.type == "cuda:2":
        torch.cuda.manual_seed_all(SEED)

    # ——————————————
    # 1. Dataset 定义 & 加载
    # ——————————————
    class NLITopicDataset(Dataset):
        def __init__(self, df: pd.DataFrame, tokenizer, topic: str, max_len: int = MAX_LEN):
            self.premises   = df["body"].fillna("").astype(str).tolist()
            self.hypotheses = [f"This text is about {topic}." for _ in self.premises]
            proba_cols = ["ML1_oracle1_contra", "ML1_oracle1_neutral", "ML1_oracle1_entail"]
            self.soft_labels = torch.tensor(df[proba_cols].values, dtype=torch.float32)
            self.tokenizer    = tokenizer
            self.max_len      = max_len

        def __len__(self):
            return len(self.premises)

        def __getitem__(self, idx):
            enc = self.tokenizer(
                self.premises[idx],
                self.hypotheses[idx],
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item["soft_labels"] = self.soft_labels[idx]
            return item

    # ——————————————
    # 2. 加载 tokenizer & 模型
    # ——————————————
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BACKBONE)
    model     = AutoModelForSequenceClassification.from_pretrained(
        MODEL_BACKBONE,
        num_labels=3,
        ignore_mismatched_sizes=True
    ).to(device)

    # 设置 label 映射
    model.config.id2label = {0: "contra", 1: "neutral", 2: "entail"}
    model.config.label2id = {v: k for k, v in model.config.id2label.items()}

    # ——————————————
    # 3. 读数据 → 划分 → DataLoader
    # ——————————————
    df = pd.read_csv(DATA_CSV).sample(frac=0.2, random_state=SEED).reset_index(drop=True)
    n_val = int(0.1 * len(df))
    n_train = len(df) - n_val
    df_train, df_val = random_split(
        df, [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
    )

    train_ds = NLITopicDataset(
        pd.DataFrame(df_train.dataset.iloc[df_train.indices]), tokenizer, TOPIC
    )
    val_ds = NLITopicDataset(
        pd.DataFrame(df_val.dataset.iloc[df_val.indices]), tokenizer, TOPIC
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ——————————————
    # 4. 损失/优化/调度
    # ——————————————
    ce_loss   = nn.CrossEntropyLoss()
    kl_loss   = nn.KLDivLoss(reduction="batchmean")
    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = total_steps // 10,
        num_training_steps = total_steps
    )

    # ——————————————
    # 5. 训练 + 验证循环
    # ——————————————
    best_val_loss = float('inf')
    ALPHA = 0.8
    T     = 2.0

    for epoch in range(1, EPOCHS + 1):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{EPOCHS}"):
            optimizer.zero_grad()
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            soft_labels    = batch["soft_labels"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).logits

            # 硬标签 + 软标签混合
            hard_labels = soft_labels.argmax(dim=1)
            loss_ce     = ce_loss(logits, hard_labels)
            log_p_s     = torch.log_softmax(logits / T, dim=1)
            p_t         = torch.softmax(soft_labels / T, dim=1)
            loss_kd     = kl_loss(log_p_s, p_t) * (T * T)
            loss        = ALPHA * loss_kd + (1 - ALPHA) * loss_ce

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total   = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                soft_labels    = batch["soft_labels"].to(device)

                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).logits
                log_p_s = torch.log_softmax(logits / T, dim=1)
                p_t     = torch.softmax(soft_labels / T, dim=1)
                val_loss += kl_loss(log_p_s, p_t) * (T * T)

                # 计算准确率
                preds  = logits.argmax(dim=1)
                labels = soft_labels.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc      = correct / total if total > 0 else 0.0
        print(f"→ Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")

        # 保存最优模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"✅ Best model saved at epoch {epoch} (Val Loss={best_val_loss:.4f}, Val Acc={val_acc:.4f})")

    print("🎉 训练完成")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
