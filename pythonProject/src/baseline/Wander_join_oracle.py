import os
import random
import json
import numpy as np
import pandas as pd
import networkx as nx
import collections
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import List, Dict, Tuple

# ==========================================
# 1. 基础类定义 (DataGraph 优化版)
# ==========================================
class DataGraph:
    def __init__(self, filepath: str):
        self.adj = collections.defaultdict(list)
        self.labels = {}
        self.nodes_by_label = collections.defaultdict(list)
        # 优化索引: adj_label[u][target_label] -> [v1, v2...]
        self.adj_by_label = collections.defaultdict(lambda: collections.defaultdict(list))
        self.load_from_file(filepath)

    def load_from_file(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Graph file not found: {filepath}")
        
        print(f"[DataGraph] Reading raw file: {filepath}")
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                if parts[0] == 'v':
                    vid, lbl = int(parts[1]), int(parts[2])
                    self.labels[vid] = lbl
                    self.nodes_by_label[lbl].append(vid)
                elif parts[0] == 'e':
                    u, v = int(parts[1]), int(parts[2])
                    self.adj[u].append(v)
                    self.adj[v].append(u)
        
        print("[DataGraph] Building label indices (Progress bar)...")
        # 使用 tqdm 显示进度
        for u, neighbors in tqdm(self.adj.items(), desc="Indexing Edges"):
            for v in neighbors:
                if v in self.labels:
                    self.adj_by_label[u][self.labels[v]].append(v)
        print("[DataGraph] Ready.")

    def get_neighbors_by_label(self, u: int, label: int) -> List[int]:
        return self.adj_by_label[u].get(label, [])

class PatternGraph:
    def __init__(self, filepath: str):
        self.G = nx.Graph()
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                if parts[0] == 'v':
                    self.G.add_node(int(parts[1]), label=int(parts[2]))
                elif parts[0] == 'e':
                    self.G.add_edge(int(parts[1]), int(parts[2]))
    @property
    def nodes(self): return self.G.nodes
    def neighbors(self, u): return self.G.neighbors(u)
    def number_of_nodes(self): return self.G.number_of_nodes()

# ==========================================
# 2. OracleManager (优化版，向量化加载)
# ==========================================
class OracleManager:
    def __init__(self, dataset_path: str, 
                 oracle_col: str = "ML1_oracle1_probability", 
                 proxy_col: str = "ML1_proxy4b1_probability"):
        self.id_type_map = {}
        self.oracle_probs = {} 
        self.proxy_probs = {}  
        self.load_data(dataset_path, oracle_col, proxy_col)

    def load_data(self, dataset_path, oracle_col, proxy_col):
        idmap_path = os.path.join(dataset_path, "data_graph/id_mapping.csv")
        post_path = os.path.join(dataset_path, "csv_data/post.csv")
        
        print("[OracleManager] Loading ID Mapping...")
        # 1. 加载 ID 映射 (只保留 Post 类型)
        map_df = pd.read_csv(idmap_path, dtype={'internal_id': int, 'orig_id': str, 'type': str}, usecols=['internal_id', 'orig_id', 'type'])
        
        # 保存 type map
        self.id_type_map = dict(zip(map_df['internal_id'], map_df['type']))
        
        # 只保留 Post 类型的映射用于合并
        post_map_df = map_df[map_df['type'] == 'Post'][['internal_id', 'orig_id']]

        print("[OracleManager] Loading Post Data (Vectorized)...")
        try:
            # 2. 加载 Post 数据
            header = pd.read_csv(post_path, nrows=0)
            id_col = 'id:ID' if 'id:ID' in header.columns else header.columns[0]
            
            # 读取
            post_df = pd.read_csv(post_path, usecols=[id_col, oracle_col, proxy_col], dtype={id_col: str})
            
            # 3. 使用 Merge 代替循环
            print("[OracleManager] Merging Data...")
            merged = pd.merge(post_map_df, post_df, left_on='orig_id', right_on=id_col, how='left')
            
            # 4. 填充 NaN 并转换为字典
            merged[oracle_col] = pd.to_numeric(merged[oracle_col], errors='coerce').fillna(0.0)
            merged[proxy_col] = pd.to_numeric(merged[proxy_col], errors='coerce').fillna(0.0)
            
            self.oracle_probs = dict(zip(merged['internal_id'], merged[oracle_col]))
            self.proxy_probs = dict(zip(merged['internal_id'], merged[proxy_col]))
            
            print(f"[OracleManager] Loaded {len(self.oracle_probs)} probabilities.")
            
        except Exception as e:
            print(f"[Error] Failed to load post.csv: {e}")

    def check_oracle(self, internal_id: int, threshold: float = 0.5) -> bool:
        if internal_id not in self.oracle_probs:
            return self.id_type_map.get(internal_id) != 'Post'
        return self.oracle_probs[internal_id] > threshold

    def get_proxy_prob(self, internal_id: int) -> float:
        return self.proxy_probs.get(internal_id, 0.0)

# ==========================================
# 3. 修改后的 WanderJoinSampler (核心修改)
# ==========================================
class WanderJoinSampler:
    # 定义需要重要性采样的标签 (Post)
    POST_LABEL = 1 

    def __init__(self, data_graph, query_graph, oracle_manager):
        self.G = data_graph
        self.Q = query_graph
        self.oracle_manager = oracle_manager
        self.id_type_map = oracle_manager.id_type_map

    def get_matching_order(self) -> List[int]:
        start_node = 0
        order = []
        visited = set()
        queue = [start_node]
        visited.add(start_node)
        while queue:
            u = queue.pop(0)
            order.append(u)
            for v in sorted(self.Q.neighbors(u)):
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
        return order

    def get_node_weight(self, node_id: int) -> float:
        prob = self.oracle_manager.get_proxy_prob(node_id)
        return np.sqrt(prob) # sqrt 平滑

    def single_walk(self, order: List[int], mode: str = 'uniform') -> Tuple[float, bool, Dict[int, int]]:
        mapping = {}
        used_data_nodes = set()
        walk_weight = 1.0
        
        for i, q_u in enumerate(order):
            target_label = self.Q.nodes[q_u]['label']
            
            # --- 1. 获取候选集 ---
            if i == 0:
                candidates = self.G.nodes_by_label.get(target_label, [])
            else:
                matched_q_neighbors = [n for n in self.Q.neighbors(q_u) if n in mapping]
                possible_v = None
                for q_neighbor in matched_q_neighbors:
                    v_prev = mapping[q_neighbor]
                    neighbors_subset = set(self.G.get_neighbors_by_label(v_prev, target_label))
                    if possible_v is None: possible_v = neighbors_subset
                    else: possible_v &= neighbors_subset
                    if not possible_v: break
                
                if possible_v: possible_v -= used_data_nodes
                candidates = list(possible_v) if possible_v else []

            if not candidates:
                return 0.0, False, {}

            # --- 2. 采样逻辑 (优化：NumPy 加速) ---
            if mode == 'importance' and target_label == self.POST_LABEL:
                # 批量计算权重
                weights = np.array([self.get_node_weight(v) for v in candidates])
                total_weight = np.sum(weights)
                
                if total_weight == 0:
                    return 0.0, False, {}
                
                # 归一化概率
                probs = weights / total_weight
                
                # 使用 numpy.random.choice 采样索引
                idx = np.random.choice(len(candidates), p=probs)
                
                v_selected = candidates[idx]
                w_selected = weights[idx]
                
                # 更新估计值 (Horvitz-Thompson)
                walk_weight *= (total_weight / w_selected)
                
            else:
                # 均匀采样逻辑
                v_selected = random.choice(candidates)
                walk_weight *= len(candidates)

            mapping[q_u] = v_selected
            used_data_nodes.add(v_selected)
                
        return walk_weight, True, mapping

    def check_oracle(self, mapping):
        for d_node in mapping.values():
            if not self.oracle_manager.check_oracle(d_node, threshold=0.5):
                return False
        return True

    def estimate(self, num_walks: int = 1000, mode: str = 'uniform'):
        order = self.get_matching_order()
        total_est = 0.0
        success_cnt = 0
        
        # 统计唯一 Post 节点 (采样开销)
        sampled_post_nodes = set()
        
        for _ in range(num_walks): 
            weight, struct_success, mapping = self.single_walk(order, mode=mode)
            
            if struct_success:
                # 记录 Post 节点
                for node_id in mapping.values():
                    if self.id_type_map.get(node_id) == 'Post':
                        sampled_post_nodes.add(node_id)

                if self.check_oracle(mapping):
                    total_est += weight
                    success_cnt += 1
                    
        return {
            "T_hat": total_est / num_walks,
            "success_rate": success_cnt / num_walks,
            "unique_post_count": len(sampled_post_nodes) # 返回统计结果
        }

# ==========================================
# 5. 批处理函数 (Batch Processing)
# ==========================================
def batch_evaluate(dataset_name="dataset_one", num_walks=1000):
    # --- 1. 路径配置 ---
    base_path = f"/home/wangshuo/resource/datasets/parler_data/{dataset_name}"
    query_dir = os.path.join(base_path, "query_graph")
    data_graph_file = os.path.join(base_path, "data_graph/parler.graph")
    # 使用你指定的 T_true 文件名
    true_file = os.path.join(base_path, "ground_truth/T_true_ML1_oracle1_probability.txt") 
    
    output_csv = f"wanderjoin_comparison_{dataset_name}.csv"

    # --- 2. 加载静态资源 ---
    print(f"====== Batch Evaluation: {dataset_name} ======")
    print(">>> 1. Loading Data Graph...")
    if not os.path.exists(data_graph_file):
        print(f"[Error] Data graph not found: {data_graph_file}")
        return None
    DG = DataGraph(data_graph_file)
    
    print(">>> 2. Loading Oracle Manager...")
    oracle = OracleManager(base_path, oracle_col="ML1_oracle1_probability", proxy_col="ML1_proxy4b1_probability")
    
    print(f">>> 3. Loading Ground Truth from {true_file}...")
    if not os.path.exists(true_file):
        print("[Error] T_true file not found.")
        return None
    
    try:
        with open(true_file, 'r') as f:
            true_vals = json.load(f)
    except Exception as e:
        print(f"[Error] JSON load failed: {e}")
        return None

    query_files = sorted([f for f in os.listdir(query_dir) if f.endswith(".graph")])
    print(f">>> Found {len(query_files)} queries. Starting comparison (Walks={num_walks})...")
    
    records = []

    # --- 3. 循环评估 ---
    for q_file in tqdm(query_files, desc="Evaluating"):
        # 匹配 T_true 中的 Key
        if q_file in true_vals:
            T_true = float(true_vals[q_file])
        elif q_file.replace(".graph", "") in true_vals:
            T_true = float(true_vals[q_file.replace(".graph", "")])
        else:
            continue
            
        if T_true == 0: continue 

        q_path = os.path.join(query_dir, q_file)
        
        try:
            QG = PatternGraph(q_path)
            sampler = WanderJoinSampler(DG, QG, oracle)
            
            # --- 方法 A: 均匀采样 (Uniform) ---
            res_u = sampler.estimate(num_walks, mode='uniform')
            est_u = res_u['T_hat']
            err_u = (est_u - T_true) / T_true
            
            records.append({
                "query_basename": q_file,
                "method": "Uniform",
                "T_true": T_true,
                "T_hat": est_u,
                "Qerror": err_u,
                "AbsRelativeError": abs(err_u),
                "SuccessRate": res_u['success_rate'],
                "n_post": res_u['unique_post_count']
            })
            
            # --- 方法 B: 重要性采样 (Importance - Post Only) ---
            res_i = sampler.estimate(num_walks, mode='importance')
            est_i = res_i['T_hat']
            err_i = (est_i - T_true) / T_true
            
            records.append({
                "query_basename": q_file,
                "method": "Importance (Post)",
                "T_true": T_true,
                "T_hat": est_i,
                "Qerror": err_i,
                "AbsRelativeError": abs(err_i),
                "SuccessRate": res_i['success_rate'],
                "n_post": res_i['unique_post_count']
            })
            
        except Exception as e:
            print(f"[Warn] Failed on {q_file}: {e}")

    # --- 4. 保存结果 ---
    df = pd.DataFrame(records)
    if not df.empty:
        df.to_csv(output_csv, index=False)
        print(f"\n>>> Results saved to {output_csv}")
        return df
    else:
        print("No results generated.")
        return None

# ==========================================
# 6. 绘图函数
# ==========================================
def plot_results(df):
    if df is None or df.empty:
        print("No data to plot.")
        return

    # 设置绘图风格
    sns.set(style="whitegrid")
    
    # --- 图 1: 相对误差箱型图 ---
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="method", y="Qerror", data=df, showfliers=False, width=0.5, palette="Set2")
    plt.axhline(0, color='red', linestyle='--', linewidth=1, label="Perfect Estimate")
    plt.title("Relative Error Distribution (WanderJoin: Uniform vs Importance)")
    plt.ylabel("(Estimated - True) / True")
    plt.ylim(-1.0, 1.0) 
    plt.yticks(np.arange(-1.0, 1.25, 0.25))  # 从 -1 到 1，每隔 0.25 设置一个刻度
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # --- 图 2: 绝对误差条形图 (MAPE) ---
    mape = df.groupby("method")["AbsRelativeError"].mean().reset_index()
    
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x="method", y="AbsRelativeError", data=mape, palette="viridis")
    plt.title("Mean Absolute Percentage Error (MAPE)")
    plt.ylabel("Average |Relative Error|")
    
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 0.01,
                f'{height:.4f}', ha="center")
    
    plt.tight_layout()
    plt.show()

    # --- 图 3: 采样效率对比 (采样节点数) ---
    # 统计每个方法采样的唯一 Post 节点数
    avg_samples = df.groupby("method")["n_post"].mean().reset_index()

    plt.figure(figsize=(8, 5))
    ax2 = sns.barplot(x="method", y="n_post", data=avg_samples, palette="Blues_d")
    plt.title("Average Unique Post Nodes Sampled (Cost)")
    plt.ylabel("Avg Unique Count")
    

    for p in ax2.patches:
        height = p.get_height()
        ax2.text(p.get_x() + p.get_width()/2., height + 1,
                f'{height:.0f}', ha="center")
    
    plt.tight_layout()
    plt.show()

    # --- 打印统计摘要 ---
    print("\n====== Statistical Summary ======")
    stats = df.groupby("method")[["Qerror", "n_post"]].describe()
    print(stats)

# ==========================================
# 7. 主执行入口
# ==========================================
# 1. 运行批处理评估
df_results = batch_evaluate(dataset_name="dataset_two", num_walks=20000)
# 2. 绘制对比图
plot_results(df_results)