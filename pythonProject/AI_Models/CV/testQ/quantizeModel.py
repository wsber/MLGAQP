import torch
from ultralytics import YOLO
import time
import os
import cv2  # 导入 OpenCV 库

# 加载未量化的YOLOv5s模型
original_model = YOLO("yolov5s.pt")

# 加载量化后的YOLOv5s模型
quantized_model = YOLO("yolov5s.pt")  # 重新加载原始模型
quantized_model.model.load_state_dict(torch.load("/home/wangshuo/ws/AI_models/CV/quantizationModel/yolov5s_quantized.pth"))
# quantized_model.model.load_state_dict(torch.load("/home/wangshuo/ws/AI_models/CV/quantizationModel/yolov5s_binarized.pth"))

# 准备测试图片
def load_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} does not exist.")
        return None
    img = cv2.imread(image_path)  # 读取图片
    if img is None:
        print(f"Error: Failed to read image {image_path}.")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换颜色格式
    return img

# 测试函数：用指定的模型进行推理
def inference(model, img):
    results = model(img)
    return results

# 计算推理时间的函数
def calculate_inference_time(model, img, num_runs=10):
    start_time = time.time()
    for _ in range(num_runs):
        _ = inference(model, img)
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    return avg_time

# 获取一张测试图像
test_image_path = "/home/wangshuo/home/wangshuo/ws/python_project/IOS_Data_Prepare_Exp/testData/1.png"  # 替换为你的图片路径
img = load_image(test_image_path)

if img is None:
    print("Failed to load image. Exiting...")
    exit()

# 对比实验：未量化模型与量化模型的推理时间
print("Testing with unquantized model...")
unquantized_time = calculate_inference_time(original_model, img)
print(f"Average inference time (unquantized): {unquantized_time:.4f} seconds")

print("Testing with quantized model...")
quantized_time = calculate_inference_time(quantized_model, img)
print(f"Average inference time (quantized): {quantized_time:.4f} seconds")

# 获取未量化模型和量化模型的推理结果
unquantized_results = inference(original_model, img)
quantized_results = inference(quantized_model, img)

# 打印未量化模型的推理结果
print("Unquantized model predictions:")
for i, pred in enumerate(unquantized_results[0].boxes):  # 修改访问方式
    label = int(pred.cls.item())  # 获取类别索引
    confidence = pred.conf.item()  # 获取置信度
    bbox = pred.xywh.tolist()  # 检测框的 [x, y, w, h]
    print(f"Prediction {i+1}: Label: {label}, Confidence: {confidence:.4f}, Bounding box: {bbox}")

# 打印量化模型的推理结果
print("Quantized model predictions:")
for i, pred in enumerate(quantized_results[0].boxes):  # 修改访问方式
    label = int(pred.cls.item())  # 获取类别索引
    confidence = pred.conf.item()  # 获取置信度
    bbox = pred.xywh.tolist()  # 检测框的 [x, y, w, h]
    print(f"Prediction {i+1}: Label: {label}, Confidence: {confidence:.4f}, Bounding box: {bbox}")
