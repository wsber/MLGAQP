import os
import json
import math
import sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
project_root = "/home/wangshuo/projects/Neo4j_Exp"
if project_root not in sys.path:
    sys.path.append(project_root)
from pythonProject.src.Structure_first.proxy_sample import ProxyStratifiedSampler, run_budget_curve_multi_predicate_fast


run_budget_curve_multi_predicate_fast(
    dataset_name="dataset_test3",
    budget_fracs=[0.01, 0.05,0.075, 0.1,0.125, 0.15,0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95],
    run_times=5
)