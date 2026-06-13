import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm.auto import tqdm

# ─── 1. 参数 & 设备 ──────────────────────────────────────────
CSV_PATH = '/home/wangshuo/resource/datasets/parler_data/30W_valid_user/10-10-5/post_LLM_cleaned_6.csv'
MODEL_NAME = '/home/wangshuo/resource/AIModels/NLP/base-uncased/distilbert-base-uncased'
TOPIC = "I support Trump"
TEST_SIZE = 0.2
RANDOM_STATE = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── 2. 读入 CSV & 划分 ─────────────────────────────────────
df = pd.read_csv(CSV_PATH)[['body', 'ML1_oracle2_probability']].dropna().reset_index(drop=True)
train_df, val_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# ─── 3. 加载 DistilBERT encoder ────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
encoder = AutoModel.from_pretrained(MODEL_NAME).to(device)
encoder.eval()


def embed_texts(texts, max_length=128):
    """返回 numpy array of shape (len(texts), hidden_size)"""
    all_vecs = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=max_length, return_tensors='pt').to(device)
        with torch.no_grad():
            out = encoder(**enc).last_hidden_state  # [B, L, H]
        # mean pooling
        vecs = out.mean(dim=1).cpu().numpy()  # [B, H]
        all_vecs.append(vecs)
    return np.vstack(all_vecs)


# ─── 4. 生成训练/验证集特征 ─────────────────────────────────
def make_features(df_subset):
    bodies = df_subset['body'].tolist()
    # 两路编码
    emb_topic = embed_texts([TOPIC] * len(bodies))
    emb_body = embed_texts(bodies)
    # 特征拼接：[t, b, t*b, |t-b|]
    feats = np.hstack([
        emb_topic,
        emb_body,
        emb_topic * emb_body,
        np.abs(emb_topic - emb_body)
    ])
    return feats


print("Embedding train set...")
X_train = make_features(train_df)
y_train = train_df['ML1_oracle2_probability'].values
print("Embedding val   set...")
X_val = make_features(val_df)
y_val = val_df['ML1_oracle2_probability'].values

# ─── 5. 训练轻量 MLP 回归器 ─────────────────────────────────
mlp = MLPRegressor(
    hidden_layer_sizes=(512, 256),
    activation='relu',
    solver='adam',
    max_iter=200,
    random_state=RANDOM_STATE,
    verbose=True
)
print("Training proxy MLP regressor...")
mlp.fit(X_train, y_train)

# ─── 6. 验证 & 保存 ────────────────────────────────────────
y_pred = mlp.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
print(f"Validation MSE: {mse:.4f}")

# 保存模型与 tokenizer（示例用 joblib）
import joblib
prosy_dir = '/home/wangshuo/resource/AIModels/Finetune/prosy_mlp/'
joblib.dump(mlp, prosy_dir +'mlp_proxy.pkl')
tokenizer.save_pretrained(prosy_dir + 'tokenizer')

print("代理模型已保存到 proxy_model/ 目录。")
