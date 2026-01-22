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
from pythonProject.src.Structure_first.proxy_sample import run_adaptive_sampling_experiment


run_adaptive_sampling_experiment(dataset_name="dataset_test", run_times=5)