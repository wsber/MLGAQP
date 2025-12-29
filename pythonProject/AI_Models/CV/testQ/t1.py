from ultralytics import YOLO

# 加载YOLOv5s模型
model = YOLO('yolov5s.pt')

# 导出为ONNX格式
model.export(format='onnx')
from onnxruntime.quantization import quantize_dynamic, QuantType

# 定义文件路径
model_fp32 = '/home/wangshuo/home/wangshuo/ws/python_project/IOS_Data_Prepare_Exp/AI_Models/CV/yolov5su.onnx'  # 导出的原始ONNX模型路径
model_int8 = '/home/wangshuo/home/wangshuo/ws/python_project/IOS_Data_Prepare_Exp/AI_Models/CV/yolov5s_quantized.onnx'  # 量化后的ONNX模型路径

# 进行动态量化（量化为INT8）
quantize_dynamic(model_fp32, model_int8, weight_type=QuantType.QUInt8)

print("量化完成，保存至：", model_int8)
