import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from transformers import AutoTokenizer, AutoModel
import torch
import joblib
from tqdm.auto import tqdm

# ─── 参数 & 设备 ──────────────────────────────────────────
CSV_PATH     = '/home/wangshuo/resource/datasets/parler_data/30W_valid_user/10-10-5/post_LLM_cleaned_6.csv'
MODEL_NAME   = '/home/wangshuo/resource/AIModels/NLP/base-uncased/distilbert-base-uncased'
PROXY_DIR    = '/home/wangshuo/resource/AIModels/Finetune/proxy_linear/'
TOPIC        = "I support Trump"
TEST_SIZE    = 0.1
RANDOM_STATE = 42

os.makedirs(PROXY_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── 读入 & 抽样（25%）& 划分 ─────────────────────────────────
full_df    = pd.read_csv(CSV_PATH)[['body','ML1_oracle2_probability']].dropna().reset_index(drop=True)
sampled_df = full_df.sample(frac=0.25, random_state=RANDOM_STATE).reset_index(drop=True)
train_df, val_df = train_test_split(sampled_df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# ─── 加载 DistilBERT encoder & tokenizer ──────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
encoder   = AutoModel.from_pretrained(MODEL_NAME).to(device)
encoder.eval()

def embed_texts(texts, max_length=256, batch_size=32):
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc   = tokenizer(batch,
                          padding=True,
                          truncation=True,
                          max_length=max_length,
                          return_tensors='pt').to(device)
        with torch.no_grad():
            hidden = encoder(**enc).last_hidden_state  # [B, L, H]
        vecs = hidden.mean(dim=1).cpu().numpy()       # mean pooling
        all_vecs.append(vecs)
    return np.vstack(all_vecs)

def make_features(df_subset):
    bodies    = df_subset['body'].astype(str).tolist()
    emb_topic = embed_texts([TOPIC] * len(bodies))
    emb_body  = embed_texts(bodies)
    return np.hstack([
        emb_topic,
        emb_body,
        emb_topic * emb_body,
        np.abs(emb_topic - emb_body)
    ])

# ─── 生成训练 & 验证特征 ─────────────────────────────────
print("Embedding train set...")
X_train = make_features(train_df)
y_train = train_df['ML1_oracle2_probability'].values
print("Embedding val   set...")
X_val   = make_features(val_df)
y_val   = val_df['ML1_oracle2_probability'].values

# ─── 训练线性回归模型 ─────────────────────────────────────
lr = LinearRegression()
print("Training LinearRegression proxy...")
lr.fit(X_train, y_train)

# ─── 验证 & 保存 ─────────────────────────────────────────
y_pred = lr.predict(X_val)
mse    = mean_squared_error(y_val, y_pred)
print(f"Validation MSE: {mse:.6f}")

# 保存模型与 tokenizer
joblib.dump(lr, os.path.join(PROXY_DIR, 'proxy_linear.pkl'))
tokenizer.save_pretrained(os.path.join(PROXY_DIR, 'tokenizer'))
print(f"Linear proxy model saved to {PROXY_DIR}")
