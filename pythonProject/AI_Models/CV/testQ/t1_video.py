import torch
import os
import cv2  # 导入 OpenCV 库
from ultralytics import YOLO
from tqdm import tqdm  # 用于显示进度条

# 检查是否有可用的 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载量化后的YOLOv5s模型
quantized_model = YOLO("yolov5s.pt")
quantized_model.model.load_state_dict(
    # torch.load("/home/wangshuo/ws/AI_models/CV/quantizationModel/yolov5s_quantized.pth"))
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

    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    print("Decoding video frames...")
    for _ in tqdm(range(total_frames), desc="Decoding Progress"):
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
    results = model(img, verbose=False)  # 禁用日志输出
    return results


# 视频文件路径
video_path = "/home/wangshuo/home/wangshuo/ws/python_project/IOS_Data_Prepare_Exp/testData/video_1min.mp4"  # 替换为你的视频路径
frames = decode_video(video_path)

if frames is None or len(frames) == 0:
    print("Failed to decode video. Exiting...")
    exit()

# 对视频逐帧进行推理并显示进度条
print("Processing video frames...")
for frame_idx, frame in tqdm(enumerate(frames), total=len(frames), desc="Inference Progress"):
    results = inference(quantized_model, frame)
