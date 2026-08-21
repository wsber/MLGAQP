import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
    ElectraTokenizerFast, ElectraForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
import torch

# 1. 加载数据
csv_path = '/home/wangshuo/resource/datasets/parler_data/30W_valid_user/10-10-5/post_LLM_cleaned_6.csv'
df = pd.read_csv(csv_path)

# 2. 随机抽样 20% 用于训练与验证，剩余 80% 未使用
sampled_df = df.sample(frac=0.3, random_state=42).reset_index(drop=True)

# 3. 构造标签：ML1_oracle2_probability > 0.5 为正例
sampled_df['label'] = (sampled_df['ML1_oracle2_probability'] > 0.5).astype(int)

# 4. 在抽样数据上划分训练/验证集（例如 80% 训练，20% 验证）
train_df, val_df = train_test_split(
    sampled_df,
    test_size=0.1,
    stratify=sampled_df['label'],
    random_state=42
)

# 5. 选择模型与分词器
# model_name = '/home/wangshuo/resource/AIModels/NLP/base-uncased/distilbert-base-uncased'
model_name = '/home/wangshuo/resource/AIModels/Finetune/distil-proxy/electra_entailment_final'
tokenizer = ElectraTokenizerFast.from_pretrained(model_name)

# 6. 数据集类
class EntailmentDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, topic_text="I support Trump", max_length=128):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.topic = topic_text
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        body = self.df.iloc[idx]['body']
        inputs = self.tokenizer(
            self.topic,
            body,
            truncation=True,
            max_length=self.max_length
        )
        item = {k: torch.tensor(v) for k, v in inputs.items()}
        item['labels'] = torch.tensor(self.df.iloc[idx]['label'], dtype=torch.long)
        return item

# 7. 初始化数据集
train_dataset = EntailmentDataset(train_df, tokenizer)
val_dataset = EntailmentDataset(val_df, tokenizer)

# 8. 数据整理器
data_collator = DataCollatorWithPadding(tokenizer)

# 9. 定义 Trainer
output_dir = '/home/wangshuo/resource/AIModels/Finetune/distil-proxy'
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=20,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='roc_auc'
)

# 自定义评估指标
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

def compute_metrics(pred):
    logits, labels = pred
    probs = torch.softmax(torch.from_numpy(logits), dim=1)[:, 1].numpy()
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds),
        'roc_auc': roc_auc_score(labels, probs)
    }

model = ElectraForSequenceClassification.from_pretrained(model_name, num_labels=2)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 10. 开始训练
if __name__ == '__main__':
    trainer.train()
    trainer.save_model(output_dir + '/electra_entailment_final')
