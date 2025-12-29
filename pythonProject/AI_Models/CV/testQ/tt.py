import torch
print(torch.__version__)  # 检查 PyTorch 版本
print(torch.version.cuda)  # 检查 PyTorch 编译时的 CUDA 版本
print(torch.cuda.is_available())  # 检查是否检测到 CUDA
