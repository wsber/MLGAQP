import os
import re
import sys
import math
import time
import tempfile
import traceback
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import json
project_root = "/home/wangshuo/projects/Neo4j_Exp"
if project_root not in sys.path:
    sys.path.append(project_root)
from pythonProject.src.Structure_first.fastest_pipeline import FastestGraphConverter, FastestEstimateMerger
from pythonProject.src.Structure_first.graph_sample import FastestRunner
from pythonProject.src.algorithms.precision_submatching import ExactSubgraphMatcher
from pythonProject.src.algorithms.proxy_sample import ProxyStratifiedSampler, compute_T_true

# 一级测试数据集
# datasets_name = "parler_data"
datasets_name = "amazon_data"
# 一级数据集下根据查询和图结构的差异划分的子测试数据集
# dataset_name = "dataset_test"
dataset_name = "amazon_extend"

# 转换后GraphLib数据存放路径
Graph_Lib_Dir = f"/home/wangshuo/resource/datasets/{datasets_name}/{dataset_name}/data_graph"

matcher = ExactSubgraphMatcher(
    exe_path="/home/wangshuo/projects/SubgraphMatching/build/matching/SubgraphMatching.out",
    default_args=["-filter", "GQL", "-order", "GQL", "-engine", "LFTJ", "-num", "MAX"],
    timeout=3000,
)
matcher.run_batch(
    data_graph=f"/home/wangshuo/resource/datasets/{datasets_name}/{dataset_name}/data_graph/parler.graph",
    query_graph_dir=f"/home/wangshuo/resource/datasets/{datasets_name}/{dataset_name}/query_graph",
    output_dir=f"/home/wangshuo/resource/datasets/{datasets_name}/{dataset_name}/ground_truth",
)