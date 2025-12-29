import numpy as np
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt


class YoLov5TRT(object):
    def __init__(self, engine_file_path):
        # 创建一个 Runtime 对象
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.runtime = trt.Runtime(self.trt_logger)

        # 反序列化 TensorRT 引擎 (加载 .engine 文件)
        with open(engine_file_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        # 创建执行上下文
        self.context = self.engine.create_execution_context()

        # 分配输入和输出内存
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            # 分配主机和设备内存
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # 记录绑定
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(host_mem)
            else:
                outputs.append(host_mem)

        return inputs, outputs, bindings, stream

    def infer(self, image):
        # 预处理图像
        input_image = self.preprocess_image(image)
        np.copyto(self.inputs[0], input_image.ravel())

        # 将输入数据从主机复制到设备
        cuda.memcpy_htod_async(self.inputs[0], self.bindings[0], self.stream)

        # 执行推理
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # 将输出数据从设备复制到主机
        cuda.memcpy_dtoh_async(self.outputs[0], self.bindings[1], self.stream)
        self.stream.synchronize()

        return self.postprocess_results(self.outputs[0])

    def preprocess_image(self, image):
        # Resize and normalize the image
        input_shape = (self.engine.max_batch_size, 3, 640, 640)
        image = cv2.resize(image, (640, 640))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return image

    def postprocess_results(self, output):
        # 后处理推理结果
        # 这里的实现取决于模型的输出格式
        return output


# 创建 YOLOv5 TensorRT 推理对象
yolov5_trt = YoLov5TRT("/home/wangshuo/home/wangshuo/ws/python_project/IOS_Data_Prepare_Exp/AI_Models/CV/Molde_instances/yolov5/yolov5s_fp16.engine")  # 这里使用 .engine 文件

# 读取输入图像
image = cv2.imread("/home/wangshuo/home/wangshuo/ws/python_project/IOS_Data_Prepare_Exp/testData/1.png")

# 执行推理
output = yolov5_trt.infer(image)

# 打印或处理输出结果
print(output)
