import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm.auto import trange
import joblib

# ─── 参数 & 设备 ──────────────────────────────────────────
CSV_PATH = '/home/wangshuo/resource/datasets/parler_data/30W_valid_user/10-10-5/post_LLM_cleaned_6.csv'
MODEL_NAME = '/home/wangshuo/resource/AIModels/NLP/base-uncased/distilbert-base-uncased'
PROXY_DIR = '/home/wangshuo/resource/AIModels/Finetune/proxy_mlp/'
TOPIC = "I support Trump"
TEST_SIZE = 0.05
RANDOM_STATE = 42

os.makedirs(PROXY_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── 读入 & 抽样（20%）& 划分 ───────────────────────────────────────
full_df = pd.read_csv(CSV_PATH)[['body', 'ML1_oracle2_probability']].dropna().reset_index(drop=True)
sampled_df = full_df.sample(frac=0.25, random_state=RANDOM_STATE).reset_index(drop=True)
train_df, val_df = train_test_split(sampled_df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# ─── 加载 Encoder ──────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
encoder = AutoModel.from_pretrained(MODEL_NAME).to(device)
encoder.eval()


def embed_texts(texts, max_length=128):
    all_vecs = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=max_length, return_tensors='pt').to(device)
        with torch.no_grad():
            out = encoder(**enc).last_hidden_state  # [B, L, H]
        vecs = out.mean(dim=1).cpu().numpy()  # mean pooling
        all_vecs.append(vecs)
    return np.vstack(all_vecs)


def make_features(df_subset):
    bodies = df_subset['body'].tolist()
    emb_topic = embed_texts([TOPIC] * len(bodies))
    emb_body = embed_texts(bodies)
    return np.hstack([
        emb_topic,
        emb_body,
        emb_topic * emb_body,
        np.abs(emb_topic - emb_body)
    ])


# ─── 生成训练 & 验证特征 ─────────────────────────────────────
print("Embedding train set...")
X_train = make_features(train_df)
y_train = train_df['ML1_oracle2_probability'].values
print("Embedding val   set...")
X_val = make_features(val_df)
y_val = val_df['ML1_oracle2_probability'].values

# ─── 初始化 MLP with warm_start ───────────────────────────
mlp = joblib.load('/home/wangshuo/resource/AIModels/Finetune/prosy_mlp/mlp_proxy_epoch200.pkl')
# mlp = MLPRegressor(
#     hidden_layer_sizes=(512, 256),
#     activation='relu',
#     solver='adam',
#     max_iter=1,           # 每次 fit 只跑 1 轮
#     warm_start=True,      # 保留上一次迭代结果
#     random_state=RANDOM_STATE,
#     verbose=False
# )

# ─── 手动迭代 & 定期保存 ───────────────────────────────────
total_epochs = 400
save_every = 100

print("Training proxy MLP with periodic saves...")
for epoch in trange(1, total_epochs + 1, desc="MLP Epoch"):
    # 注意：这里依然用同一份 train_df，每次训练都在那 20% 数据上迭代
    mlp.fit(X_train, y_train)
    # loss_curve_[-1] 就是这一轮的训练损失
    epoch_loss = mlp.loss_curve_[-1]
    # 输出当前 epoch 的训练损失
    print(f"Epoch {epoch:3d} — train loss: {epoch_loss:.6f}")
    # 每隔 20 轮，或者最后一轮，保存一次模型
    if epoch % save_every == 0 or epoch == total_epochs:
        save_path = os.path.join(PROXY_DIR, f"mlp_proxy_epoch{epoch}.pkl")
        joblib.dump(mlp, save_path)
        tokenizer.save_pretrained(os.path.join(PROXY_DIR, 'tokenizer'))
        print(f"  ↳ Saved proxy model at epoch {epoch} to {save_path}")

# ─── 验证 ─────────────────────────────────────────────────
y_pred = mlp.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
print(f"Final Validation MSE: {mse:.4f}")
print("All done.")
