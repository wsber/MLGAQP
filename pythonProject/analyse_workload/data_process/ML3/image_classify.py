import os
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import torch

# 1. 配置路径
product_csv_path = "/home/wangshuo/resource/datasets/amazon_data/amazon_extend/csv_data/product_init.csv"
image_folder = "/home/fuyiding/dataset/AmazonReviews/homeAndkitchen/pic/images_kcore"
# model_path = "/home/wangshuo/resource/AIModels/CV/clip-vit-large-patch14"
model_path = "/home/wangshuo/resource/AIModels/CV/clip-vit-base-patch32"
output_csv_path = "/home/wangshuo/resource/datasets/amazon_data/amazon_extend/csv_data/product_init.csv"

# 2. 加载模型
print("Loading model...")
model = CLIPModel.from_pretrained(model_path)
processor = CLIPProcessor.from_pretrained(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
# model.half()
model.to(device).eval()

# 3. 读取 CSV 文件
print("Reading CSV data...")
df = pd.read_csv(product_csv_path)

# 4. 定义分类标签 (添加提示词模板 "a photo of a" 可以显著提高 CLIP 的准确率)
candidate_labels = [
    "a photo of a plastic product", # index 0
    "a photo of a metal object",    # index 1
    "a photo of a wooden item",     # index 2  <-- 这是我们需要提取的目标
    "a photo of a fabric textile",  # index 3
    "a photo of a glass material"   # index 4
]
target_index = 2 # "wooden item" 在 candidate_labels 中的索引位置

# 5. 批量推理函数 (修复了由于包含 None 导致 processor 崩溃的问题)
def infer_images_batch(image_paths):
    valid_images = []
    valid_indices = []
    
    # 过滤无效图片
    for idx, path in enumerate(image_paths):
        try:
            if not os.path.exists(path):
                raise FileNotFoundError("File not found")
            image = Image.open(path).convert("RGB")
            valid_images.append(image)
            valid_indices.append(idx)
        except Exception as e:
            # 静默处理或打印错误
            # print(f"Error loading image {path}: {e}")
            pass

    # 初始化该批次的所有返回值为 None
    batch_results = [None] * len(image_paths)

    # 如果整个批次都没有有效图片，直接返回
    if not valid_images:
        return batch_results

    # 将有效图片送入模型
    inputs = processor(
        text=candidate_labels,
        images=valid_images,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        # dim=1 表示对 5 个类别进行 Softmax 得到 0~1 的概率分布
        probs = logits_per_image.softmax(dim=1).cpu().numpy()

    # 提取 "wooden item" (index=2) 的概率，并填回对应的原位置
    for i, valid_idx in enumerate(valid_indices):
        batch_results[valid_idx] = probs[i][target_index]

    return batch_results

# 6. 遍历每一行，准备图片路径
print("Preparing image paths...")
image_paths = []
for _, row in df.iterrows():
    parent_asin = row["parent_asin"]
    image_path = os.path.join(image_folder, f"0_{parent_asin}_img1.jpg")
    image_paths.append(image_path)

# 分批处理
batch_size = 32
probs = []

for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images in batches"):
    batch_paths = image_paths[i:i + batch_size]
    batch_probs = infer_images_batch(batch_paths)
    probs.extend(batch_probs)

# 7. 保存结果到新列 (这里列名改成了 wooden_item_probability，如果你需要原来的列名可以自行修改)
df["ML3_proxy1_probability"] = probs
df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

print(f"推理完成，'wooden item' 的概率结果已保存到：{output_csv_path}")