from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from torch.utils.data import Dataset
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# 模型与tokenizer路径
model_dir = '/home/wangshuo/resource/AIModels/NLP/NLI/bert-mini-finetuned-mnli'
proxy_dir = '/home/wangshuo/resource/AIModels/Finetune/'
# model_dir = proxy_dir + 'bert-mini-softlabel-nli'

# 1. 加载预训练模型和 tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# 2. 修改为二分类模型（仅保留 entailment vs contradiction）
model.num_labels = 2
model.classifier = torch.nn.Linear(model.classifier.in_features, 2)

# 3. 定义数据集类
datadir = '/home/wangshuo/resource/datasets/parler_data/30W_valid_user/10-10-5/'
filename = datadir + 'post_LLM_cleaned_6.csv'


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
        inputs = self.tokenizer(premise, hypothesis, truncation=True, padding='max_length', max_length=256,
                                return_tensors='pt')

        # 使用 soft label
        prob = row['ML1_oracle2_probability']
        label = torch.tensor([1 - prob, prob], dtype=torch.float)

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': label
        }


# 4. 读取数据（假设只使用 1/5）
df = pd.read_csv(os.path.join(datadir, filename))
df_sample = df.sample(frac=0.3, random_state=42).reset_index(drop=True)

# 5. 构造数据集
train_dataset = NLIDataset(df_sample, tokenizer)

# 6. 自定义 Trainer 以支持 soft labels
from transformers import Trainer


class SoftLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        loss = loss_fn(log_probs, labels)
        return (loss, outputs) if return_outputs else loss


# 7. TrainingArguments 设置
training_args = TrainingArguments(
    output_dir=proxy_dir + "results",
    num_train_epochs=20,
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

# 8. 开始微调
trainer = SoftLabelTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer)
)

trainer.train()

# 可保存模型
model.save_pretrained(proxy_dir + "bert-mini-softlabel-nli")
tokenizer.save_pretrained(proxy_dir + "bert-mini-softlabel-nli")
