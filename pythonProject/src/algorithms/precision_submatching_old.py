import os
import csv
import random
import subprocess
import re
import argparse
import json

def load_query_budgets(csv_path, target_frac=0.1, target_method="8_POSSA"):
    """从消融实验 CSV 中读取每个查询在 0.1 采样率下的专属 oracle_cost"""
    query_budgets = {}
    if not os.path.exists(csv_path):
        print(f"[Error] 找不到消融 CSV 文件: {csv_path}")
        return query_budgets

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frac = float(row.get('budget_frac', 0))
                if abs(frac - target_frac) > 1e-4: 
                    continue
                
                method = row.get('method', '').strip()
                if method not in ['POSS', '8_POSSA'] and target_method in ['POSS', '8_POSSA']: 
                    continue
                if method != target_method and target_method not in ['POSS', '8_POSSA']: 
                    continue
                
                q_name = row['query_basename'].strip()
                cost = int(float(row['oracle_cost']))
                
                if q_name not in query_budgets:
                    query_budgets[q_name] = []
                query_budgets[q_name].append(cost)
            except Exception:
                continue
                
    # 计算多轮 run_id 的平均预算并四舍五入
    final_budgets = {q: int(round(sum(c)/len(c))) for q, c in query_budgets.items()}
    return final_budgets

def load_id_mapping(mapping_csv):
    """读取 id_mapping.csv, 返回 orig_id -> internal_id"""
    orig_to_internal = {}
    if not os.path.exists(mapping_csv):
        print(f"[Error] 找不到 ID Mapping 文件: {mapping_csv}")
        return orig_to_internal

    with open(mapping_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None) # 跳过表头
        for row in reader:
            if len(row) >= 2:
                try: 
                    orig_to_internal[row[1].strip()] = int(row[0].strip())
                except ValueError: 
                    pass
    return orig_to_internal

def load_scores(csv_path, orig_to_internal, proxy_col, oracle_col):
    """读取指定表 (如 product.csv / review.csv) 的 Proxy 和 Oracle 概率"""
    proxy_scores, oracle_scores = {}, {}
    if not os.path.exists(csv_path): 
        return proxy_scores, oracle_scores

    with open(csv_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            orig_id = row.get("id:ID", "").strip()
            if orig_id in orig_to_internal:
                internal_id = orig_to_internal[orig_id]
                try:
                    proxy_scores[internal_id] = float(row[proxy_col])
                    oracle_scores[internal_id] = float(row[oracle_col])
                except Exception: 
                    pass
    return proxy_scores, oracle_scores

def main():
    parser = argparse.ArgumentParser(description="Batch Run Query-Specific AQP Pipeline")
    parser.add_argument("--parent_dataset", default="amazon_data", help="Parent dataset directory name")
    parser.add_argument("--dataset", required=True, help="Dataset folder name (e.g., amazon_extend)")
    parser.add_argument("--fastest_bin", required=True, help="Path to Fastest C++ binary")
    parser.add_argument("--ablation_csv", required=True, help="Path to allocation_strategy_comparison_ablation_sum.csv")
    parser.add_argument("--table1", required=True, help="Table 1 name (e.g., product)")
    parser.add_argument("--table1_proxy", required=True)
    parser.add_argument("--table1_oracle", required=True)
    parser.add_argument("--table2", required=True, help="Table 2 name (e.g., review)")
    parser.add_argument("--table2_proxy", required=True)
    parser.add_argument("--table2_oracle", required=True)
    parser.add_argument("--sum_col", default="price", help="Column name for SUM aggregation")
    parser.add_argument("--sum_label", type=int, default=12, help="Label index for SUM aggregation")
    args = parser.parse_args()

    # 1. 基础路径配置
    base_dir = f"/home/wangshuo/resource/datasets/{args.parent_dataset}/{args.dataset}"
    graph_path = os.path.join(base_dir, "data_graph", "parler.graph")
    graph_bak_path = os.path.join(base_dir, "data_graph", "parler.graph.bak")
    mapping_path = os.path.join(base_dir, "data_graph", "id_mapping.csv")
    config_path = os.path.join(base_dir, "data_graph", "core_nodes_config.json")
    temp_ans_path = os.path.join(base_dir, "ground_truth", "temp_ans.txt")
    out_csv = os.path.join(base_dir, "results", "efficiency", "AQP_Cascade_results.csv")

    # 2. 读取核心节点配置
    print("[*] 正在加载 core_nodes_config.json...")
    if not os.path.exists(config_path):
        print(f"[Error] 找不到配置文件: {config_path}")
        return
    with open(config_path, 'r') as f:
        core_config = json.load(f)

    # 3. 提取 Query-specific Budgets
    print("[*] 提取 Query-specific Budgets (frac=0.1)...")
    query_budgets = load_query_budgets(args.ablation_csv, target_frac=0.1)
    print(f"[+] 成功提取到 {len(query_budgets)} 个查询的专属预算")

    # 4. 加载原始数据图与 ML 概率分数
    print("[*] 加载物理图结构与概率分数...")
    orig_to_internal = load_id_mapping(mapping_path)
    p1_scores, o1_scores = load_scores(os.path.join(base_dir, "csv_data", f"{args.table1}.csv"), orig_to_internal, args.table1_proxy, args.table1_oracle)
    p2_scores, o2_scores = load_scores(os.path.join(base_dir, "csv_data", f"{args.table2}.csv"), orig_to_internal, args.table2_proxy, args.table2_oracle)

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

    # 安全备份原始图
    if not os.path.exists(graph_bak_path):
        os.rename(graph_path, graph_bak_path)
        print("[+] 原始图数据已安全备份为 parler.graph.bak")

    results = []

    try:
        # 5. 逐 Query 自动化运行
        for q_idx, (q_name, budget) in enumerate(query_budgets.items(), 1):
            print(f"\n>>> [{q_idx}/{len(query_budgets)}] 正在处理 {q_name} (Budget B = {budget})")
            
            # 5.1 判断当前 Query 包含哪些表 (根据原始 Label 范围判断)
            has_table1 = False
            has_table2 = False

            if q_name in core_config:
                raw_labels = {int(k) for k in core_config[q_name].keys()}
                # 假设 Amazon 表1(Product)原始Label包含 10..19 或 12；表2(Review)原始Label包含 30..39 或 30
                for l in raw_labels:
                    if l < 20 or l in [1, 2, 10, 11, 12, 13, 14]:
                        has_table1 = True
                    if l >= 20 or l in [3, 4, 5, 30, 31, 32, 33, 34]:
                        has_table2 = True
            else:
                has_table1, has_table2 = True, True

            print(f"    --> 作用的数据表: Table1({args.table1})={has_table1}, Table2({args.table2})={has_table2}")

            # 5.2 为当前 Query 专属裁剪图
            uncertain_nodes = []
            rejected_nodes = set()
            
            for vid in range(len(vertices)):
                # 判断当前节点所属的表，并获取分数
                p_score, o_score = None, None
                
                if vid in p1_scores and has_table1:
                    p_score = p1_scores[vid]
                    o_score = o1_scores.get(vid, 0.0)
                elif vid in p2_scores and has_table2:
                    p_score = p2_scores[vid]
                    o_score = o2_scores.get(vid, 0.0)

                # 如果节点属于当前 Query 激活的表，进行硬裁剪判断
                if p_score is not None:
                    if p_score < 0.4:
                        rejected_nodes.add(vid)  # 硬剔除 (<0.4)
                    elif p_score > 0.6:
                        pass  # 硬接受 (>0.6)
                    else:
                        uncertain_nodes.append((vid, o_score, p_score)) # 灰色地带

            # 打乱灰色地带节点
            random.shuffle(uncertain_nodes)
            
            budget_used = 0
            for vid, o_score, p_score in uncertain_nodes:
                if budget_used < budget:
                    # 在 Oracle 预算内验证真值
                    if o_score <= 0.5:
                        rejected_nodes.add(vid)
                    budget_used += 1
                else:
                    # 预算耗尽，退化为 Proxy 判断
                    if p_score <= 0.5:
                        rejected_nodes.add(vid)

            # print(f"    --> 裁剪完成! 消耗 Oracle 预算: {budget_used}/{budget}. 剔除节点数: {len(rejected_nodes)} (其中硬剔除数: {len(rejected_nodes) - (budget_used - sum(1 for v, o, p in uncertain_nodes[:budget_used] if o > 0.5))})")

            # 统计分别剔除了多少 Product 和 Review
            rejected_t1 = sum(1 for vid in rejected_nodes if vid in p1_scores)
            rejected_t2 = sum(1 for vid in rejected_nodes if vid in p2_scores)

            print(f"    --> 裁剪完成! 消耗 Oracle 预算: {budget_used}/{budget}. 总剔除数: {len(rejected_nodes)} (Product剔除: {rejected_t1}, Review剔除: {rejected_t2})")

            # 覆写当前用于运行的 parler.graph
            with open(graph_path, 'w') as f:
                f.write(header + "\n")
                for vid in range(len(vertices)):
                    label, deg = vertices[vid]
                    if vid in rejected_nodes:
                        label = -1
                    f.write(f"v {vid} {label} {deg}\n")
                for e in edges:
                    f.write(e + "\n")

            # 5.3 写入单 Query 的临时答案文件
            with open(temp_ans_path, 'w') as f:
                f.write(f"{q_name} 1.0 1\n")

            # 5.4 调用 C++ 进行纯结构估计
            cmd = [
                args.fastest_bin, 
                "-d", args.dataset,
                "-q", temp_ans_path,
                "--PARENT_DATASET", args.parent_dataset,
                "--ROOT_LABEL", "1",
                "--SAMPLE_BUDGET", "60000",
                "--AGG_FUNC", "sum",
                "--SUM_TABLE", args.table1,
                "--SUM_COL", args.sum_col,
                "--SUM_LABEL", str(args.sum_label)
            ]
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()

            # 5.5 从终端输出提取估算结果
            est_value = None
            match = re.search(r"Global Estimated Value:\s+([\d\.]+)", stdout)
            if match:
                est_value = float(match.group(1))
            else:
                match2 = re.search(r"\[Result\]\s+Est\s*:\s*([\d\.]+)", stdout)
                if match2: 
                    est_value = float(match2.group(1))

            if est_value is not None:
                print(f"    --> AQP 估计值: {est_value}")
                results.append({"query_basename": q_name, "T_hat_aqp": est_value})
            else:
                print(f"    --> [Error] 提取估算结果失败！错误输出: {stderr[:200]}")

    finally:
        # 6. 安全恢复原始数据图
        print("\n[*] 正在恢复原始 parler.graph 文件...")
        if os.path.exists(graph_path): 
            os.remove(graph_path)
        if os.path.exists(graph_bak_path):
            os.rename(graph_bak_path, graph_path)
            print("[+] 原图已成功还原！")
        
        # 7. 导出最终的 AQP 估计结果 CSV
        if results:
            keys = results[0].keys()
            os.makedirs(os.path.dirname(out_csv), exist_ok=True)
            with open(out_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results)
            print(f"\n✅ 批量 AQP 实验完成！结果已导出至: {out_csv}")

if __name__ == "__main__":
    main()