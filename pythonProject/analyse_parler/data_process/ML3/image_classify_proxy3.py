import os
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch

# ==================== 1. 模型与全局配置区域 ====================

# 如果你需要测试纯CNN的高速模型，可以将这里换成 "openai/clip-resnet-50"
model_path = "wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M" 
batch_size = 256
num_workers = 8  # 核心提速点：开启 8 个并行的后台进程，疯狂读取硬盘上的图片
target_column_name = "ML3_proxy3_probability"

product_csv_path = "/home/wangshuo/resource/datasets/amazon_data/amazon_extend/csv_data/product.csv"
image_folder = "/home/fuyiding/dataset/AmazonReviews/homeAndkitchen/pic/images_kcore"
output_csv_path = "/home/wangshuo/resource/datasets/amazon_data/amazon_extend/csv_data/product.csv"

# ==================== 2. 分类标签定义 ====================
candidate_labels = [
    "a photo of a plastic product",     # 0
    "a photo of a metal object",        # 1
    "a photo of a wooden item",         # 2 <-- 木头
    "a photo of wooden furniture",      # 3 <-- 木头
    "a product made of solid wood",     # 4 <-- 木头
    "a photo of a fabric textile",      # 5
    "a photo of a glass material"       # 6
]
wooden_indices = [2, 3, 4]

# ==================== 3. 读取 CSV ====================
print("Reading CSV data...")
try:
    df = pd.read_csv(product_csv_path, encoding="utf-8-sig")
except UnicodeDecodeError:
    df = pd.read_csv(product_csv_path)

df.columns = df.columns.str.strip()
if "id:ID" not in df.columns:
    raise ValueError("CSV文件中找不到 'id:ID' 列")

# ==================== 4. 加载模型与【预计算文本特征】 ====================
print(f"Loading model from {model_path} ...")
device = "cuda:2" if torch.cuda.is_available() else "cpu"
device_type = "cuda" if "cuda" in device else "cpu"

# 权重半精度加载，静态显存砍半
model = CLIPModel.from_pretrained(model_path, torch_dtype=torch.float16).to(device).eval()
processor = CLIPProcessor.from_pretrained(model_path)

print("Pre-computing text features (Optimization 1)...")
with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.float16):
    # 极速优化 1：在这整个脚本生命周期中，文本特征只算这 1 次！绝对不放进循环！
    text_inputs = processor(text=candidate_labels, return_tensors="pt", padding=True).to(device)
    text_features = model.get_text_features(**text_inputs)
    # 提前做 L2 归一化，留作给图像算相似度时用
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

# ==================== 5. 【异步数据加载器构建】 ====================
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
            # 极速优化 2：图像的 I/O 读取和 Transformer 预处理被下放到了多核 CPU 子进程
            image = Image.open(path).convert("RGB")
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
            return {"pixel_values": pixel_values, "valid": True, "idx": idx}
        except Exception:
            # 万一遇到损坏图片，不阻断进程，给一个空的占位符并标记无效
            return {"pixel_values": torch.zeros((3, 224, 224)), "valid": False, "idx": idx}

print("Preparing DataLoader (Optimization 2)...")
dataset = ImageDataset(df, image_folder, processor)
# num_workers 让 CPU 并行打工，GPU 只管吃现成的饭
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

# ==================== 6. 极致推理循环 ====================
print("Starting ultra-fast inference...")
# 初始化结果列表，默认 0.0
batch_results = [0.0] * len(df)

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Processing Images"):
        valid_mask = batch["valid"]
        if not valid_mask.any(): 
            continue  # 如果这整个批次的图片碰巧都是坏的，跳过
        
        # 将张量挪到 GPU。这里拿到的仅仅是有效图片
        pixel_values = batch["pixel_values"][valid_mask].to(device)
        indices = batch["idx"][valid_mask].numpy()

        # 开启自动混合半精度，利用 Tensor Core 加速
        with torch.autocast(device_type=device_type, dtype=torch.float16):
            # 只需要让模型过一下【图像编码器】即可，跳过所有多余的 wrapper 代码
            image_features = model.get_image_features(pixel_values=pixel_values)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            # 手动执行：余弦相似度 × 缩放系数
            logit_scale = model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.T
            
            # 7分类 Softmax 相互挤压，结果落回 CPU 内存以策安全
            probs = logits_per_image.softmax(dim=1).to(torch.float32).cpu().numpy()

        # 提取目标材质（木头）的概率
        for i, df_idx in enumerate(indices):
            # 因为经过 softmax 处理过了，这 7 个类的概率和已经是 1，直接加起来即可
            wood_prob = probs[i][wooden_indices].sum()
            batch_results[df_idx] = float(wood_prob)

# ==================== 7. 结果保存 ====================
df[target_column_name] = batch_results
df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

print(f"极致加速推理完成！结果已保存到：{output_csv_path}，新增列名为：{target_column_name}")