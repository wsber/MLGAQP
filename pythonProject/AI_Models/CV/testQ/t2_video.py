import torch
import time
import os
import cv2  # 导入 OpenCV 库
from ultralytics import YOLO

# 检查是否有可用的 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载量化后的YOLOv5s模型
quantized_model = YOLO("yolov5s.pt")
quantized_model.model.load_state_dict(
    # torch.load("/home/wangshuo/ws/AI_models/CV/quantizationModel/yolov5s_quantized.pth"))
    torch.load("/home/wangshuo/ws/AI_models/CV/quantizationModel/yolov5s_binarized.pth" ) )
quantized_model.model.to(device)  # 将模型移动到 GPU 或 CPU


# 视频解码函数
def decode_video(video_path):
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} does not exist.")
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}.")
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 转换为 RGB 格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames


# 测试函数：用指定的模型进行推理
def inference(model, img):
    results = model(img)
    return results


# 计算视频推理速度
def calculate_video_inference_time(model, frames):
    total_time = 0
    for frame in frames:
        start_time = time.time()
        _ = inference(model, frame)
        total_time += time.time() - start_time
    avg_time = total_time / len(frames)
    return avg_time


# 视频文件路径
video_path = "/home/wangshuo/home/wangshuo/ws/python_project/IOS_Data_Prepare_Exp/testData/video_1min.mp4"  # 替换为你的视频路径
frames = decode_video(video_path)

if frames is None or len(frames) == 0:
    print("Failed to decode video. Exiting...")
    exit()

# # 计算视频推理时间
# print(f"Testing quantized model on video {video_path}...")
# quantized_time = calculate_video_inference_time(quantized_model, frames)
# print(f"Average inference time per frame (quantized): {quantized_time:.4f} seconds")

# 对视频逐帧进行推理并打印结果
print("Processing video frames...")
for frame_idx, frame in enumerate(frames):
    results = inference(quantized_model, frame)
    print(f"Frame {frame_idx + 1} predictions:")
    for i, pred in enumerate(results[0].boxes):  # 修改访问方式
        label = int(pred.cls.item())  # 获取类别索引
        confidence = pred.conf.item()  # 获取置信度
        bbox = pred.xywh.tolist()  # 检测框的 [x, y, w, h]
        print(f"  Prediction {i + 1}: Label: {label}, Confidence: {confidence:.4f}, Bounding box: {bbox}")
