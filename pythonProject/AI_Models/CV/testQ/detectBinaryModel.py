import torch
import os
import cv2  # 导入 OpenCV 库
from ultralytics import YOLO
from tqdm import tqdm  # 用于显示进度条

# 加载 YOLO 模型（这里没有加载权重）
quantized_model = YOLO("yolov5s.pt")  # 载入原始 YOLOv5s 模型

# 加载二值化后的模型权重
model_path = "/home/wangshuo/ws/AI_models/CV/quantizationModel/yolov5s_quantized.pth"
state_dict = torch.load(model_path)  # 加载权重

# 将加载的权重映射到 YOLO 模型上
quantized_model.model.load_state_dict(state_dict)

# 确保模型使用了适当的设备
device = "cuda" if torch.cuda.is_available() else "cpu"
quantized_model.model.to(device)  # 将模型移动到 GPU 或 CPU

# 遍历模型中的所有参数，检查权重是否被二值化为 1 或 -1
for name, param in quantized_model.model.named_parameters():
    if 'weight' in name:  # 仅查看卷积层的权重
        print(f"{name}:")

        # 打印权重的最小值和最大值，确认是否为 -1 和 1
        print(f"  Min value: {param.min().item()}, Max value: {param.max().item()}")

        # 打印权重的部分值（只显示前几个元素）
        print(f"  Weight values:\n {param.data[:5]}")  # 只打印前 5 个值，避免输出过多

        # 检查权重是否二值化为 -1 或 1
        binary_check = torch.all(torch.abs(param.data) == 1)
        print(f"  Are the weights binary (-1 or 1)? {binary_check.item()}")

        # 如果需要，您可以打印权重的绝对值是否全为 1
        # 这样可以进一步确认是否所有权重都为 1 或 -1
        abs_check = torch.all(torch.abs(param.data) == 1)
        print(f"  All weights are binary? {abs_check.item()}")
