import os
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch

# ==================== 1. 模型配置区域 (Oracle) ====================

# Oracle 模型 (Ground Truth) - 极度准确，显存占用稍大
model_path = "/home/wangshuo/resource/AIModels/CV/siglip-so400m-patch14-384"
batch_size = 16 
num_workers = 8  # 核心提速点：开启并行的后台进程读取图片
target_column_name = "ML3_oracle2_probability"

# ==================== 2. 全局路径配置 ====================
product_csv_path = "/home/wangshuo/resource/datasets/amazon_data/amazon_extend/csv_data/product.csv"
image_folder = "/home/fuyiding/dataset/AmazonReviews/homeAndkitchen/pic/images_kcore"
output_csv_path = "/home/wangshuo/resource/datasets/amazon_data/amazon_extend/csv_data/product.csv"

# ==================== 3. 优化后的分类标签 ====================
candidate_labels = [
    "a photo of a plastic product",     # 0
    "a photo of a metal object",        # 1
    "a photo of a wooden item",         # 2 <-- 木头特征 1
    "a photo of wooden furniture",      # 3 <-- 木头特征 2
    "a product made of solid wood",     # 4 <-- 木头特征 3
    "a photo of a fabric textile",      # 5
    "a photo of a glass material"       # 6
]
wooden_indices = [2, 3, 4]

# ==================== 4. 读取 CSV 文件 ====================
print("Reading CSV data...")
try:
    df = pd.read_csv(product_csv_path, encoding="utf-8-sig")
except UnicodeDecodeError:
    df = pd.read_csv(product_csv_path)

df.columns = df.columns.str.strip()

if "id:ID" not in df.columns:
    raise ValueError(f"CSV文件中找不到 'id:ID' 列，当前列名为: {df.columns.tolist()}")

# ==================== 5. 加载 SigLIP 模型与【预计算文本特征】 ====================
print(f"Loading model from {model_path} ...")
device = "cuda:2" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 按要求不使用半精度加速，保持默认 float32
model = AutoModel.from_pretrained(model_path).to(device).eval()
processor = AutoProcessor.from_pretrained(model_path)

print("Pre-computing text features (Optimization 1)...")
with torch.no_grad():
    # 极速优化 1：文本特征只算这 1 次！
    # SigLIP 必须使用 padding="max_length"
    text_inputs = processor(text=candidate_labels, return_tensors="pt", padding="max_length").to(device)
    text_features = model.get_text_features(**text_inputs)
    # L2 归一化
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

# ==================== 6. 【异步数据加载器构建】 ====================
class ImageDataset(Dataset):
    def __init__(self, df, image_folder, processor):
        self.item_ids = df["id:ID"].tolist()
        self.image_folder = image_folder
        self.processor = processor

    def __len__(self):
        return len(self.item_ids)

    def __getitem__(self, idx):
        item_id = self.item_ids[idx]
        path = os.path.join(self.image_folder, f"0_{item_id}_img1.jpg")
        try:
            # 极速优化 2：多进程并行读取图像并进行预处理
            image = Image.open(path).convert("RGB")
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
            return {"pixel_values": pixel_values, "valid": True, "idx": idx}
        except Exception:
            # 遇到损坏图片不阻断进程，给一个空的占位符并标记无效。动态获取图像所需的边长尺寸（通常SigLIP为384）
            img_size = self.processor.image_processor.size.get("height", 384)
            return {"pixel_values": torch.zeros((3, img_size, img_size)), "valid": False, "idx": idx}

print("Preparing DataLoader (Optimization 2)...")
dataset = ImageDataset(df, image_folder, processor)
# num_workers 让 CPU 并行预处理图像
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

# ==================== 7. 极致推理循环 ====================
print("Starting ultra-fast inference...")
batch_results = [None] * len(df)

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Processing Images"):
        valid_mask = batch["valid"]
        if not valid_mask.any(): 
            continue
        
        # 仅将有效的图像丢入 GPU
        pixel_values = batch["pixel_values"][valid_mask].to(device)
        indices = batch["idx"][valid_mask].numpy()

        # 计算图像特征并进行 L2 归一化
        image_features = model.get_image_features(pixel_values=pixel_values)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
        # ！！！核心区别：SigLIP 的相似度不仅有缩放系数，还要加上偏置（bias）！！！
        logit_scale = model.logit_scale.exp()
        logit_bias = model.logit_bias
        logits_per_image = logit_scale * image_features @ text_features.T + logit_bias
        
        # SigLIP 的 Loss 是 Sigmoid，绝对不能用 Softmax
        probs = torch.sigmoid(logits_per_image).cpu().numpy()

        # 概率提取与归一化逻辑
        for i, df_idx in enumerate(indices):
            # 1. 提取所有负向材质（非木头）的绝对概率
            plastic_prob = probs[i][0]
            metal_prob   = probs[i][1]
            fabric_prob  = probs[i][5]
            glass_prob   = probs[i][6]
            
            # 2. 提取正向材质（木头）的绝对概率（3个木头特征的平均值）
            wood_raw_prob = probs[i][wooden_indices].mean()
            
            # 3. 将所有可能材质的概率加总，作为归一化基准
            total_material_score = plastic_prob + metal_prob + fabric_prob + glass_prob + wood_raw_prob
            
            # 4. 计算木头材质的“相对纯二分类概率”
            if total_material_score > 0:
                final_wood_prob = wood_raw_prob / total_material_score
            else:
                final_wood_prob = 0.0
                
            batch_results[df_idx] = float(final_wood_prob)

# ==================== 8. 结果保存 ====================
df[target_column_name] = batch_results
df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

print(f"推理完成，木头概率结果已保存到：{output_csv_path}，新增列名为：{target_column_name}")