import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm.auto import tqdm
import joblib

# ─── 参数 & 设备 ──────────────────────────────────────────
CSV_PATH   = '/home/wangshuo/resource/datasets/parler_data/30W_valid_user/10-10-5/post_LLM_cleaned_6.csv'
MODEL_NAME = '/home/wangshuo/resource/AIModels/NLP/base-uncased/distilbert-base-uncased'
PROXY_DIR  = '/home/wangshuo/resource/AIModels/Finetune/proxy_svm/'
TOPIC      = "I support Trump"
TEST_SIZE  = 0.1
RANDOM_STATE = 42

os.makedirs(PROXY_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── 读入 & 抽样（20%）& 划分 ───────────────────────────────
full_df    = pd.read_csv(CSV_PATH)[['body','ML1_probability']].dropna().reset_index(drop=True)
sampled_df = full_df.sample(frac=0.2, random_state=RANDOM_STATE).reset_index(drop=True)
# 构造二分类硬标签
sampled_df['label_bin'] = (sampled_df['ML1_probability'] > 0.5).astype(int)
train_df, val_df = train_test_split(
    sampled_df, test_size=TEST_SIZE, stratify=sampled_df['label_bin'], random_state=RANDOM_STATE
)

# ─── 加载 Encoder ──────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
encoder   = AutoModel.from_pretrained(MODEL_NAME).to(device)
encoder.eval()

def embed_texts(texts, max_length=128, batch_size=32):
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=max_length, return_tensors='pt').to(device)
        with torch.no_grad():
            out = encoder(**enc).last_hidden_state  # [B,L,H]
        vecs = out.mean(dim=1).cpu().numpy()       # mean pooling
        all_vecs.append(vecs)
    return np.vstack(all_vecs)

def make_features(df_subset):
    bodies    = df_subset['body'].tolist()
    emb_topic = embed_texts([TOPIC]*len(bodies))
    emb_body  = embed_texts(bodies)
    return np.hstack([
        emb_topic,
        emb_body,
        emb_topic * emb_body,
        np.abs(emb_topic - emb_body)
    ])

# ─── 生成特征与标签 ───────────────────────────────────────
print("Embedding train set...")
X_train = make_features(train_df)
y_train = train_df['label_bin'].values
print("Embedding val   set...")
X_val   = make_features(val_df)
y_val   = val_df['label_bin'].values

# ─── 定义 Pipeline 与网格搜索参数 ──────────────────────────
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(probability=True, random_state=RANDOM_STATE))
])

param_grid = {
    'svc__kernel':       ['linear', 'rbf', 'poly'],
    'svc__C':            [0.1, 1, 10, 100],
    'svc__gamma':        ['scale', 'auto', 0.1, 1, 10],
    'svc__class_weight': ['balanced', None]
}

grid = GridSearchCV(
    pipe,
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2
)

# ─── 训练 & 调参 ─────────────────────────────────────────
print("Starting Grid Search for SVM...")
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
print(f"Best CV AUC: {grid.best_score_:.4f}")

# ─── 验证 & 保存 ─────────────────────────────────────────
best_svc = grid.best_estimator_
val_probs = best_svc.predict_proba(X_val)[:,1]
val_preds = (val_probs > 0.5).astype(int)

auc = roc_auc_score(y_val, val_probs)
acc = accuracy_score(y_val, val_preds)
print(f"Validation AUC: {auc:.4f}, Accuracy: {acc:.4f}")

# 保存最优模型与 tokenizer
joblib.dump(best_svc, os.path.join(PROXY_DIR, 'proxy_svm_best.pkl'))
tokenizer.save_pretrained(os.path.join(PROXY_DIR, 'tokenizer'))
print(f"Best proxy SVM saved to {PROXY_DIR}")
