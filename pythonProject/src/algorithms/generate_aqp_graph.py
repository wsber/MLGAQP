import os
import csv
import random
import argparse

def load_id_mapping(mapping_csv):
    """读取 id_mapping.csv, 返回 orig_id -> internal_id"""
    orig_to_internal = {}
    with open(mapping_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)  # 尝试跳过表头
        for row in reader:
            if len(row) >= 2:
                try:
                    internal_id = int(row[0].strip())
                    orig_id = row[1].strip()
                    orig_to_internal[orig_id] = internal_id
                except ValueError:
                    continue
    return orig_to_internal

def load_scores(csv_path, orig_to_internal, proxy_col, oracle_col):
    """读取指定表的 Proxy 和 Oracle 分数"""
    proxy_scores = {}
    oracle_scores = {}
    if not os.path.exists(csv_path):
        return proxy_scores, oracle_scores

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            orig_id = row.get("id:ID", "").strip()
            if orig_id in orig_to_internal:
                internal_id = orig_to_internal[orig_id]
                try:
                    proxy_scores[internal_id] = float(row[proxy_col])
                    oracle_scores[internal_id] = float(row[oracle_col])
                except (KeyError, ValueError):
                    pass
    return proxy_scores, oracle_scores

def main():
    parser = argparse.ArgumentParser(description="Generate AQP Cascade Filtered Graph")
    parser.add_argument("--dataset_dir", required=True, help="Base path to dataset (e.g., .../amazon_extend)")
    parser.add_argument("--budget", type=int, required=True, help="Oracle Budget B")
    parser.add_argument("--table1", required=True, help="Table 1 name (e.g., product)")
    parser.add_argument("--table1_proxy", required=True)
    parser.add_argument("--table1_oracle", required=True)
    parser.add_argument("--table2", required=True, help="Table 2 name (e.g., review)")
    parser.add_argument("--table2_proxy", required=True)
    parser.add_argument("--table2_oracle", required=True)
    
    args = parser.parse_args()

    # 1. 路径设置
    graph_path = os.path.join(args.dataset_dir, "data_graph", "parler.graph") # 假设原图文件名
    out_graph_path = os.path.join(args.dataset_dir, "data_graph", f"parler_cascade_B{args.budget}.graph")
    mapping_path = os.path.join(args.dataset_dir, "data_graph", "id_mapping.csv")
    csv_dir = os.path.join(args.dataset_dir, "csv_data")

    # 2. 读取 ID 映射和概率分数
    print("[*] Loading mappings and scores...")
    orig_to_internal = load_id_mapping(mapping_path)
    
    p1_scores, o1_scores = load_scores(
        os.path.join(csv_dir, f"{args.table1}.csv"), orig_to_internal, args.table1_proxy, args.table1_oracle
    )
    p2_scores, o2_scores = load_scores(
        os.path.join(csv_dir, f"{args.table2}.csv"), orig_to_internal, args.table2_proxy, args.table2_oracle
    )

    # 合并两张表的概率
    proxy_scores = {**p1_scores, **p2_scores}
    oracle_scores = {**o1_scores, **o2_scores}

    # 3. 解析原始图结构
    print("[*] Parsing original graph...")
    header = ""
    vertices = {}  # vid -> (label, degree)
    edges = []
    
    with open(graph_path, 'r') as f:
        for line in f:
            if line.startswith('t'):
                header = line.strip()
            elif line.startswith('v'):
                parts = line.split()
                vertices[int(parts[1])] = (int(parts[2]), int(parts[3]))
            elif line.startswith('e'):
                edges.append(line.strip())

    # 4. 执行 PROXY-CASCADE-FILTER 逻辑
    print("[*] Applying Cascade Filter...")
    uncertain_nodes = []
    rejected_nodes = set()

    for vid, (label, deg) in vertices.items():
        if vid in proxy_scores:
            p_score = proxy_scores[vid]
            # Hard-pruning
            if p_score < 0.4:
                rejected_nodes.add(vid)
            elif p_score > 0.6:
                pass # 接受 (即使可能是 False Positive，这就是 AQP 偏差来源)
            else:
                uncertain_nodes.append(vid)

    # 打乱灰色地带节点，模拟随机验证
    random.shuffle(uncertain_nodes)
    
    budget_used = 0
    for vid in uncertain_nodes:
        if budget_used < args.budget:
            # 在预算内，使用 Oracle 验证真实性
            if oracle_scores.get(vid, 0.0) <= 0.5:
                rejected_nodes.add(vid)
            budget_used += 1
        else:
            # 预算耗尽，退化为硬代理判断 (阈值 0.5)
            if proxy_scores.get(vid, 0.0) <= 0.5:
                rejected_nodes.add(vid)

    print(f"[*] Filter Complete. Budget Used: {budget_used}. Rejected Nodes: {len(rejected_nodes)}")

    # 5. 输出新图 (拒绝的节点 Label 置为 -1)
    print(f"[*] Writing new graph to {out_graph_path}...")
    with open(out_graph_path, 'w') as f:
        f.write(header + "\n")
        for vid in range(len(vertices)): # 保证严格递增顺序
            label, deg = vertices[vid]
            if vid in rejected_nodes:
                label = -1
            f.write(f"v {vid} {label} {deg}\n")
        for e in edges:
            f.write(e + "\n")
            
    print("[+] Done!")

if __name__ == "__main__":
    main()