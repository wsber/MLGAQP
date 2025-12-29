import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

# 加载预训练的 YOLOv5s 模型
model = YOLO("yolov5s.pt")

# 自定义二值化函数：将权重值二值化为 0 或 1
def binarize_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            # 归一化每个权重张量，然后将其二值化
            # 这里使用 sign 函数来进行二值化 (+1/-1)
            binary_weight = torch.sign(param)
            # 将权重限制为 0 或 1
            binary_weight[binary_weight == -1] = 0
            param.data.copy_(binary_weight)
    return model

# 二值化模型的权重
binarized_model = binarize_weights(model.model)

# 打印二值化后的模型架构
print(binarized_model)

# 保存二值化后的模型
save_path = "/home/wangshuo/ws/AI_models/CV/quantizationModel/yolov5s_binarized.pth"
torch.save(binarized_model.state_dict(), save_path)

print(f"二值化后的模型已保存到 {save_path}")
