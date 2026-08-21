import os
import pandas as pd
from tqdm import tqdm  # 导入 tqdm 库

# 定义要处理的目录路径
csv_dir = '/home/wangshuo/resource/datasets/parler/csv_data/pc/'
# 获取该目录下所有的 CSV 文件
csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
# 使用 tqdm 包裹文件列表，添加进度条
for csv_file in tqdm(csv_files, desc="Processing CSV files", unit="file"):
    csv_path = os.path.join(csv_dir, csv_file)
    # 读取 CSV 文件
    df = pd.read_csv(csv_path)
    # 删除 body 字段为空值的行
    df_cleaned = df[df['body'].notna() & (df['body'] != '')]

    # 将清理后的数据保存回 CSV 文件
    df_cleaned.to_csv(csv_path, index=False)

    # 打印进度信息
    print(f"Processed and cleaned {csv_file}")
