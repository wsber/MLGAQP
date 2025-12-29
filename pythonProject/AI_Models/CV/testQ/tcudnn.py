import onnxruntime as ort
import cv2
import os
import numpy as np
from tqdm import tqdm

# 检查是否有可用的 GPU
gpu_available = ort.get_device() == 'GPU'
print(f"Using device: {ort.get_device()}")
# os.environ["CUDA_LAZY_LOADING"] = "1"
# 加载量化后的ONNX模型
model_int8 = '/home/wangshuo/home/wangshuo/ws/python_project/IOS_Data_Prepare_Exp/AI_Models/CV/yolov5s_quantized.onnx'
# model_int8 = '/home/wangshuo/home/wangshuo/ws/python_project/IOS_Data_Prepare_Exp/AI_Models/CV/yolov5su.onnx'

session = ort.InferenceSession(model_int8,
                               providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])


# 视频解码函数
def decode_video(video_path):
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} does not exist.")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}.")
        return None

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
def inference(session, input_name, img):
    img_input = cv2.resize(img, (640, 640))  # 假设模型输入是640x640
    img_input = np.transpose(img_input, (2, 0, 1)).astype(np.float32)  # (H, W, C) -> (C, H, W)
    img_input = np.expand_dims(img_input, axis=0)  # 添加批量维度 (1, 3, 640, 640)

    outputs = session.run(None, {input_name: img_input})
    return outputs


# 视频文件路径
video_path = "/home/wangshuo/home/wangshuo/ws/python_project/IOS_Data_Prepare_Exp/testData/video_1min.mp4"
frames = decode_video(video_path)

if frames is None or len(frames) == 0:
    print("Failed to decode video. Exiting...")
    exit()

# 获取模型输入名称
input_name = session.get_inputs()[0].name

# 对视频逐帧进行推理并显示进度条
print("Processing video frames...")
for frame_idx, frame in tqdm(enumerate(frames), total=len(frames), desc="Inference Progress"):
    results = inference(session, input_name, frame)
    # 处理结果，例如打印检测框等
    # 这里需要根据你的模型输出进行适当的后处理