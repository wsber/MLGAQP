import os
from PIL import Image

image_folder = '/home/wangshuo/resource/AI-Models/CV/yolov5/new_model_qua/calibrationData/val2017_sub'
for img_name in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_name)
    try:
        img = Image.open(img_path)
        img.verify()  # 确保图像文件不是损坏的
    except (IOError, SyntaxError):
        print(f"Invalid image: {img_path}")
