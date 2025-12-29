from PIL import Image
import numpy as np

# 加载图片
image_path = '/home/wangshuo/home/wangshuo/ws/python_project/IOS_Data_Prepare_Exp/testData/1.png'
img = Image.open(image_path)

# 调整图片大小，例如调整为 640x640（YOLOv5 的常见输入大小）
img_resized = img.resize((640, 640))

# 转换为 numpy 数组
img_array = np.array(img_resized)

# 归一化到 [0, 1]
img_array = img_array.astype(np.float32) / 255.0

# 转换为 [batch_size, channels, height, width] 格式
img_array = np.transpose(img_array, (2, 0, 1))  # (C, H, W)
img_array = np.expand_dims(img_array, axis=0)    # (1, C, H, W)

# 保存为 .raw 格式
raw_path = '/home/wangshuo/home/wangshuo/ws/python_project/IOS_Data_Prepare_Exp/testData/1.raw'
img_array.tofile(raw_path)

print(f"Image saved as .raw at {raw_path}")
