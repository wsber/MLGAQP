import os
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm
import torch

# ==================== 1. 模型配置区域 (切换代理/Oracle) ====================

# 【选项 A】：代理模型 (Proxy) - 速度快，显存占用极小
model_path = "/home/wangshuo/resource/AIModels/CV/siglip-base"
batch_size = 64  # 修改为半精度后，如果显存有富余，可以尝试将这里的 batch_size 调大 (如 128)
target_column_name = "ML3_proxy2_probability"

# ==================== 2. 全局路径配置 ====================
product_csv_path = "/home/wangshuo/resource/datasets/amazon_data/amazon_extend/csv_data/product.csv"
image_folder = "/home/fuyiding/dataset/AmazonReviews/homeAndkitchen/pic/images_kcore"
output_csv_path = "/home/wangshuo/resource/datasets/amazon_data/amazon_extend/csv_data/product.csv"

# ==================== 3. 加载 SigLIP 模型 ====================
print(f"Loading model from {model_path} ...")
# 添加 torch_dtype=torch.float16，直接以半精度加载模型权重，大幅降低显存占用
model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16)
processor = AutoProcessor.from_pretrained(model_path)
device = "cuda:2" if torch.cuda.is_available() else "cpu"
# 提取具体的硬件类型供 autocast 使用
device_type = "cuda" if "cuda" in device else "cpu"
print(f"Using device: {device}")
model.to(device).eval()

# ==================== 4. 读取 CSV 文件 ====================
print("Reading CSV data...")
try:
    df = pd.read_csv(product_csv_path, encoding="utf-8-sig")
except UnicodeDecodeError:
    df = pd.read_csv(product_csv_path)

df.columns = df.columns.str.strip()

if "id:ID" not in df.columns:
    raise ValueError(f"CSV文件中找不到 'id:ID' 列，当前列名为: {df.columns.tolist()}")

# ==================== 5. 优化后的分类标签 ====================
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

# ==================== 6. 批量推理函数 ====================
def infer_images_batch(image_paths):
    valid_images = []
    valid_indices = []
    
    for idx, path in enumerate(image_paths):
        try:
            if not os.path.exists(path):
                raise FileNotFoundError("File not found")
            image = Image.open(path).convert("RGB")
            valid_images.append(image)
            valid_indices.append(idx)
        except Exception:
            pass

    batch_results = [None] * len(image_paths)

    if not valid_images:
        return batch_results

    inputs = processor(
        text=candidate_labels,
        images=valid_images,
        return_tensors="pt",
        padding="max_length"
    ).to(device)

    with torch.no_grad():
        # 添加 autocast (自动混合精度) 上下文管理器
        with torch.autocast(device_type=device_type, dtype=torch.float16):
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
        
        # Sigmoid 限制绝对输出在 0-1 之间，转换回 float32 保证精度
        probs = torch.sigmoid(logits_per_image).to(torch.float32).cpu().numpy()

    # 提取木头的概率并转换为标准的二分类概率
    for i, valid_idx in enumerate(valid_indices):
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
            # 只有当木头得分 > 其他所有材质得分之和时，final_wood_prob 才会 > 0.5
            final_wood_prob = wood_raw_prob / total_material_score
        else:
            final_wood_prob = 0.0
            
        batch_results[valid_idx] = float(final_wood_prob)

    return batch_results

# ==================== 7. 执行批量推理 ====================
print("Preparing image paths...")
image_paths = []
for _, row in df.iterrows():
    item_id = row["id:ID"]
    image_path = os.path.join(image_folder, f"0_{item_id}_img1.jpg")
    image_paths.append(image_path)

probs_list = []

for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images in batches"):
    batch_paths = image_paths[i:i + batch_size]
    batch_probs = infer_images_batch(batch_paths)
    probs_list.extend(batch_probs)

# ==================== 8. 保存结果 ====================
df[target_column_name] = probs_list
df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

print(f"推理完成，木头相对二分类概率(支持>0.5判定)已保存到：{output_csv_path}，新增列名为：{target_column_name}")