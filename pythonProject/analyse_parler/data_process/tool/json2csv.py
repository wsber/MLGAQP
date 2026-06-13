import json
import pandas as pd
from tqdm import tqdm
import os

# 定义 NDJSON 文件的路径和输出的 CSV 文件路径
# ndjson_dir = '/home/wangshuo/resource/DataSets/parler/parler_users/'  # NDJSON 文件所在目录
# csv_dir = '/home/wangshuo/resource/DataSets/parler/csv_data/user/'  # 输出 CSV 文件所在目录
ndjson_dir = '/home/wangshuo/resource/datasets/parler/parler_datas_json_90/'  # NDJSON 文件所在目录
csv_dir = '/home/wangshuo/resource/datasets/parler/csv_data/pc/'  # 输出 CSV 文件所在目录
# 获取 NDJSON 文件的文件名列表（假设文件名是类似 parler_user000000000002-8.ndjson）
ndjson_files = [f for f in os.listdir(ndjson_dir) if f.endswith('.ndjson')]

# 遍历每个 NDJSON 文件
for ndjson_file in ndjson_files:
    ndjson_path = os.path.join(ndjson_dir, ndjson_file)
    csv_file = os.path.join(csv_dir, f"{ndjson_file.replace('.ndjson', '.csv')}")

    # 初始化一个空列表，用来存储所有的 JSON 对象
    data = []

    # 获取 NDJSON 文件的总行数，用于设置 tqdm 的进度条总数
    with open(ndjson_path, 'r') as f:
        total_lines = sum(1 for line in f)

    # 打开 NDJSON 文件并读取，每次处理一行，加入进度条
    with open(ndjson_path, 'r') as f:
        for line in tqdm(f, total=total_lines, desc=f"Processing {ndjson_file}", unit="line"):
            # 清除异常行终止符
            line = line.replace('\u2028', ' ').replace('\u2029', ' ')
            data.append(json.loads(line.strip()))

    # 将 JSON 数据转换为 DataFrame
    df = pd.DataFrame(data)

    # 将 DataFrame 写入 CSV 文件
    df.to_csv(csv_file, index=False)

    print(f"Data from {ndjson_file} has been successfully converted to {csv_file}")
