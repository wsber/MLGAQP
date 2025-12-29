import tensorrt as trt
import numpy as np
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import pycuda.driver as cuda
import pycuda.autoinit

# 加载TensorRT引擎
engine_path = "/home/wangshuo/home/wangshuo/ws/python_project/IOS_Data_Prepare_Exp/AI_Models/CV/Molde_instances/yolov5/yolov5s_fp16.engine"
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# 反序列化引擎
with open(engine_path, 'rb') as f:
    engine_data = f.read()
runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(engine_data)

# 创建执行上下文
context = engine.create_execution_context()

# 准备输入图片
image_path = "/home/wangshuo/home/wangshuo/ws/python_project/IOS_Data_Prepare_Exp/testData/1.png"
image = Image.open(image_path)

# 图片预处理
preprocess = transforms.Compose([
    transforms.Resize((640, 640)),  # YOLOv5的默认输入尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_image = preprocess(image).unsqueeze(0).numpy()

# 重新调整为TensorRT需要的格式
input_data = np.ascontiguousarray(input_image)

# 获取输入和输出绑定的信息
bindings = []
for binding in engine:
    binding_shape = engine.get_binding_shape(binding)
    dtype = trt.nptype(engine.get_binding_dtype(binding))
    size = np.prod(binding_shape) * np.dtype(dtype).itemsize
    bindings.append(np.empty(binding_shape, dtype=dtype))

# 获取GPU内存和数据类型
d_input = cuda.mem_alloc(input_data.nbytes)
d_output = cuda.mem_alloc(np.prod(engine.get_binding_shape(1)) * np.dtype(np.float32).itemsize)

# 将输入数据复制到GPU
cuda.memcpy_htod(d_input, input_data)

# 执行推理
context.execute_v2(bindings=[int(d_input), int(d_output)])

# 从GPU取回输出数据
cuda.memcpy_dtoh(bindings[1], d_output)

# 输出结果
output_data = bindings[1]
print("Inference results:", output_data)
