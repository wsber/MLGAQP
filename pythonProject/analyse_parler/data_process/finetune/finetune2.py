from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import os

# 禁用多线程警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 模型与 tokenizer 路径
model_dir = '/home/wangshuo/resource/AIModels/NLP/NLI/bert-mini-finetuned-mnli'
proxy_dir = '/home/wangshuo/resource/AIModels/Finetune/'

def main():
    # 1. 加载预训练模型和 tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # 2. 修改为二分类模型（仅保留 entailment vs contradiction）
    model.num_labels = 2
    model.classifier = torch.nn.Linear(model.classifier.in_features, 2)

    # 3. 定义数据集类
    datadir = '/home/wangshuo/resource/datasets/parler_data/30W_valid_user/10-10-5/'
    filename = 'post_LLM_cleaned_6.csv'

    class NLIDataset(Dataset):
        def __init__(self, dataframe, tokenizer):
            self.data = dataframe
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            row = self.data.iloc[idx]
            premise = row['body']
            hypothesis = "This text is about I support Trump."
            inputs = self.tokenizer(
                premise,
                hypothesis,
                truncation=True,
                padding='max_length',
                max_length=256,
                return_tensors='pt'
            )

            # 使用硬标签：根据概率阈值 (>0.5 为 1，否则为 0)
            prob = row['ML1_oracle2_probability']
            label = torch.tensor(int(prob > 0.5), dtype=torch.long)

            return {
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'labels': label
            }

    # 4. 读取数据并抽样
    df = pd.read_csv(os.path.join(datadir, filename))
    df_sample = df.sample(frac=0.2, random_state=42).reset_index(drop=True)

    # 5. 构造数据集
    train_dataset = NLIDataset(df_sample, tokenizer)

    # 6. 自定义 Trainer（可选）以显式使用 CrossEntropyLoss
    class HardLabelTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
            labels = inputs.pop("labels")        # shape: (batch,)
            outputs = model(**inputs)
            logits = outputs.logits              # shape: (batch, 2)
            loss_fn = CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

    # 7. TrainingArguments 设置
    training_args = TrainingArguments(
        output_dir=os.path.join(proxy_dir, "results"),
        num_train_epochs=5,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        save_steps=500,
        evaluation_strategy="no",
        disable_tqdm=False,
        report_to="none"
    )

    # 8. 初始化 Trainer 并开始微调
    trainer = HardLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer)
    )

    trainer.train()

    # 9. 保存微调后模型和 tokenizer
    out_dir = os.path.join(proxy_dir, "bert-mini-hardlabel-nli")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"模型已保存到: {out_dir}")

if __name__ == "__main__":
    main()
