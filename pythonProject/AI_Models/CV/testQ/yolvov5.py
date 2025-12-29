import torch
from ultralytics import YOLO

# 加载预训练的 YOLOv5s 模型
model = YOLO("yolov5s.pt")  # 选择合适的模型，yolov5s 是较小的模型

# 使用动态量化对模型进行量化
# 动态量化针对 Conv2d 和 Linear 层进行量化
quantized_model = torch.quantization.quantize_dynamic(
    model.model,  # 获取 YOLOv5 模型的核心部分
    {torch.nn.Conv2d, torch.nn.Linear},  # 对卷积层和全连接层进行量化
    dtype=torch.qint8  # 使用 8 位整数进行量化
)

# 打印量化后的模型架构
print(quantized_model)

# 保存量化后的模型到指定路径
save_path = "/home/wangshuo/ws/AI_models/CV/quantizationModel/yolov5s_quantized.pth"
torch.save(quantized_model.state_dict(), save_path)

print(f"量化后的模型已保存到 {save_path}")
