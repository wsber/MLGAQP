import os
import csv
import random
import subprocess
import re
import argparse
import json
import threading
import concurrent.futures

# 全局线程锁，用于保护 CSV 文件的并发写入和终端打印
csv_lock = threading.Lock()
print_lock = threading.Lock()

def safe_print(msg):
    """线程安全的打印函数，防止控制台输出错乱"""
    with print_lock:
        print(msg)

def load_query_budgets(csv_path, target_frac=0.1, target_method="8_POSSA"):
    query_budgets = {}
    if not os.path.exists(csv_path):
        return query_budgets
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frac = float(row.get('budget_frac', 0))
                if abs(frac - target_frac) > 1e-4: continue
                method = row.get('method', '').strip()
                if method not in ['POSS', '8_POSSA'] and target_method in ['POSS', '8_POSSA']: continue
                if method != target_method and target_method not in ['POSS', '8_POSSA']: continue
                q_name = row['query_basename'].strip()
                cost = int(float(row['oracle_cost']))
                if q_name not in query_budgets: query_budgets[q_name] = []
                query_budgets[q_name].append(cost)
            except Exception: continue
    return {q: int(round(sum(c)/len(c))) for q, c in query_budgets.items()}

def load_id_mapping(mapping_csv):
    orig_to_internal = {}
    if os.path.exists(mapping_csv):
        with open(mapping_csv, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    try: orig_to_internal[row[1].strip()] = int(row[0].strip())
                    except ValueError: pass
    return orig_to_internal

def load_scores(csv_path, orig_to_internal, proxy_col, oracle_col):
    proxy_scores, oracle_scores = {}, {}
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                orig_id = row.get("id:ID", "").strip()
                if orig_id in orig_to_internal:
                    internal_id = orig_to_internal[orig_id]
                    try:
                        proxy_scores[internal_id] = float(row[proxy_col])
                        oracle_scores[internal_id] = float(row[oracle_col])
                    except Exception: pass
    return proxy_scores, oracle_scores

def process_single_query(q_name, budget, args, header, vertices, edges, p1_scores, p2_scores, o1_scores, o2_scores, core_config):
    """供线程池调用的独立处理函数，所有操作都在线程专属的临时文件上进行"""
    try:
        # 1. 解析当前 Query 需要的标签
        has_table1, has_table2 = False, False
        target_labels = set()
        if q_name in core_config:
            raw_labels = {int(k) for k in core_config[q_name].keys()}
            for l in raw_labels:
                if l < 20 or l in [1, 2, 10, 11, 12, 13, 14]: has_table1 = True
                if l >= 20 or l in [3, 4, 5, 30, 31, 32, 33, 34]: has_table2 = True
        else:
            has_table1, has_table2 = True, True

        # 2. 对当前图进行专属代理过滤
        uncertain_nodes = []
        rejected_nodes = set()
        
        for vid in range(len(vertices)):
            p_score, o_score = None, None
            t_low, t_high = None, None
            
            if vid in p1_scores and has_table1:
                p_score, o_score = p1_scores[vid], o1_scores.get(vid, 0.0)
                t_low, t_high = args.t1_low, args.t1_high
            elif vid in p2_scores and has_table2:
                p_score, o_score = p2_scores[vid], o2_scores.get(vid, 0.0)
                t_low, t_high = args.t2_low, args.t2_high

            if p_score is not None:
                if p_score < t_low: rejected_nodes.add(vid)
                elif p_score > t_high: pass
                else: uncertain_nodes.append((vid, o_score, p_score, (t_low + t_high) / 2.0))

        random.shuffle(uncertain_nodes)
        
        budget_used = 0
        for vid, o_score, p_score, t_mid in uncertain_nodes:
            if budget_used < budget:
                if o_score <= 0.5: rejected_nodes.add(vid)
                budget_used += 1
            else:
                if p_score <= t_mid: rejected_nodes.add(vid)

        # 3. 构造线程专属安全文件名 (防止读写冲突)
        clean_q_name = q_name.replace(".graph", "")
        thread_graph_name = f"parler_tmp_{clean_q_name}_{threading.get_ident()}.graph"
        thread_ans_name = f"ans_tmp_{clean_q_name}_{threading.get_ident()}.txt"

        base_dir = f"/home/wangshuo/resource/datasets/{args.parent_dataset}/{args.dataset}"
        thread_graph_path = os.path.join(base_dir, "data_graph", thread_graph_name)
        thread_ans_path = os.path.join(base_dir, "ground_truth", thread_ans_name)

        # 4. 覆写线程专属的 Graph
        with open(thread_graph_path, 'w') as f:
            f.write(header + "\n")
            for vid in range(len(vertices)):
                label, deg = vertices[vid]
                if vid in rejected_nodes: label = -1
                f.write(f"v {vid} {label} {deg}\n")
            for e in edges: f.write(e + "\n")

        # 5. 覆写线程专属的 Ans
        with open(thread_ans_path, 'w') as f:
            f.write(f"{q_name} 1.0 1\n")

        # 6. 调用 C++ (传入专属图名)
        cmd = [
            args.fastest_bin, 
            "-d", args.dataset,
            "-q", thread_ans_path,
            "--PARENT_DATASET", args.parent_dataset,
            "--DATA_GRAPH", thread_graph_name,  # 重要：传入线程专属图名
            "--ROOT_LABEL", "1",
            "--SAMPLE_BUDGET", "60000",
            "--AGG_FUNC", "sum",
            "--SUM_TABLE", args.table1,
            "--SUM_COL", args.sum_col,
            "--SUM_LABEL", str(args.sum_label)
        ]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        # 7. 清理临时文件 (重要)
        if os.path.exists(thread_graph_path): os.remove(thread_graph_path)
        if os.path.exists(thread_ans_path): os.remove(thread_ans_path)

        # 8. 提取结果
        est_value = None
        match = re.search(r"Global Estimated Value:\s+([\d\.]+)", stdout)
        if match: est_value = float(match.group(1))
        else:
            match2 = re.search(r"\[Result\]\s+Est\s*:\s*([\d\.]+)", stdout)
            if match2: est_value = float(match2.group(1))

        rejected_t1 = sum(1 for vid in rejected_nodes if vid in p1_scores)
        rejected_t2 = sum(1 for vid in rejected_nodes if vid in p2_scores)

        if est_value is not None:
            safe_print(f"✅ [{q_name}] B={budget} | 剔除 T1:{rejected_t1} T2:{rejected_t2} | AQP= {est_value:.2f}")
            return {"query_basename": q_name, "T_hat_aqp": est_value}
        else:
            safe_print(f"❌ [{q_name}] 提取失败! 错误日志: {stderr[:100]}")
            return None

    except Exception as e:
        safe_print(f"❌ [{q_name}] 线程异常: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Batch Run Query-Specific AQP Pipeline")
    parser.add_argument("--parent_dataset", default="amazon")
    # parser.add_argument("--dataset", required=True)
    parser.add_argument("--fastest_bin", required=True)
    parser.add_argument("--ablation_csv", required=True)
    parser.add_argument("--table1", required=True)
    parser.add_argument("--table1_proxy", required=True)
    parser.add_argument("--table1_oracle", required=True)
    parser.add_argument("--table2", required=True)
    parser.add_argument("--table2_proxy", required=True)
    parser.add_argument("--table2_oracle", required=True)
    parser.add_argument("--t1_low", type=float, default=0.2)
    parser.add_argument("--t1_high", type=float, default=0.3)
    parser.add_argument("--t2_low", type=float, default=0.5)
    parser.add_argument("--t2_high", type=float, default=0.7)
    parser.add_argument("--sum_col", default="price")
    parser.add_argument("--sum_label", type=int, default=12)
    parser.add_argument("--workers", type=int, default=8, help="最大并发线程数")
    args = parser.parse_args()

    base_dir = f"/home/hp/PROXY/datasets/{args.parent_dataset}"
    graph_path = os.path.join(base_dir, "data_graph", "parler.graph")
    mapping_path = os.path.join(base_dir, "data_graph", "id_mapping.csv")
    config_path = os.path.join(base_dir, "data_graph", "core_nodes_config.json")
    out_csv = os.path.join(base_dir, "results", "efficiency", "AQP_Cascade_results.csv")

    safe_print("[*] 读取配置文件与图拓扑...")
    if not os.path.exists(config_path): return
    with open(config_path, 'r') as f: core_config = json.load(f)

    query_budgets = load_query_budgets(args.ablation_csv, target_frac=0.1)
    orig_to_internal = load_id_mapping(mapping_path)
    p1, o1 = load_scores(os.path.join(base_dir, "csv_data", f"{args.table1}.csv"), orig_to_internal, args.table1_proxy, args.table1_oracle)
    p2, o2 = load_scores(os.path.join(base_dir, "csv_data", f"{args.table2}.csv"), orig_to_internal, args.table2_proxy, args.table2_oracle)

    header = ""
    vertices = {}
    edges = []
    with open(graph_path, 'r') as f:
        for line in f:
            if line.startswith('t'): header = line.strip()
            elif line.startswith('v'): 
                p = line.split()
                vertices[int(p[1])] = (int(p[2]), int(p[3]))
            elif line.startswith('e'): edges.append(line.strip())

    safe_print(f"🚀 开始多线程并发执行 (Threads={args.workers})...")
    
    # 初始化清空 CSV 头部
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["query_basename", "T_hat_aqp"])

    # 启动线程池
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for q_name, budget in query_budgets.items():
            # 提交任务到线程池
            future = executor.submit(
                process_single_query, q_name, budget, args, header, vertices, edges, 
                p1, p2, o1, o2, core_config
            )
            futures.append(future)

        # 异步即时处理结果
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res is not None:
                # 线程安全地即时写入 CSV
                with csv_lock:
                    with open(out_csv, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=res.keys())
                        writer.writerow(res)

    safe_print(f"🎉 全部执行完毕！结果已实时安全保存至: {out_csv}")

if __name__ == "__main__":
    main()