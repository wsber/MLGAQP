import os
import csv
import ast
import json
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures

def safe_extract_list(val):
    if pd.isna(val) or val == "":
        return []
    if isinstance(val, (list, tuple)):
        return list(val)
    if isinstance(val, (int, float)):
        return [val]
    if isinstance(val, str):
        val = val.strip()
        if val in ["", "nan", "[]"]:
            return []
        if val.startswith('[') and val.endswith(']'):
            try:
                res = json.loads(val.replace("'", '"'))
                return res if isinstance(res, list) else [res]
            except Exception:
                try:
                    res = ast.literal_eval(val)
                    return res if isinstance(res, list) else [res]
                except Exception:
                    return []
        else:
            return [val]
    return []

def parse_float_list(lst):
    out = []
    for x in lst:
        try: out.append(float(x))
        except (ValueError, TypeError): pass
    return out

def load_query_budgets(csv_path, target_frac=0.1, target_method="8_POSSA"):
    query_budgets = {}
    if not os.path.exists(csv_path):
        print(f"[Error] 找不到消融 CSV 文件: {csv_path}")
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
                query_budgets.setdefault(q_name, []).append(cost)
            except Exception:
                continue
                
    return {q: int(round(sum(c)/len(c))) for q, c in query_budgets.items()}

def process_single_file(fname, agg_dir, query_budgets, args_dict):
    try:
        q_basename = fname.replace("aggregated_list_", "").replace(".csv", "") + ".graph"
        budget_B = query_budgets.get(q_basename, 500)

        filepath = os.path.join(agg_dir, fname)
        df = pd.read_csv(filepath)
        if df.empty:
            return None

        weight_col = "estimateW" if "estimateW" in df.columns else "a"
        if weight_col not in df.columns:
            return None

        records = df.to_dict('records')

        # ----------------------------------------------------
        # 步骤 1：收集唯一物理节点 (增加列名自适应兼容)
        # ----------------------------------------------------
        unique_nodes = {}

        # 智能匹配 ID 列名，防止参数与 CSV 实际列名微小差异导致解析为空
        t1_id_col = args_dict['t1_ids']
        t2_id_col = args_dict['t2_ids']
        if records:
            sample_r = records[0]
            if t1_id_col not in sample_r:
                t1_id_col = "post_id_list" if "post_id_list" in sample_r else ("product_id_list" if "product_id_list" in sample_r else t1_id_col)
            if t2_id_col not in sample_r:
                t2_id_col = "comment_id_list" if "comment_id_list" in sample_r else ("review_id_list" if "review_id_list" in sample_r else t2_id_col)

        for row_dict in records:
            p1_list = parse_float_list(safe_extract_list(row_dict.get(args_dict['t1_proxy'])))
            o1_list = parse_float_list(safe_extract_list(row_dict.get(args_dict['t1_oracle'])))
            id1_list = safe_extract_list(row_dict.get(t1_id_col))
                
            for idx, (p, o) in enumerate(zip(p1_list, o1_list)):
                nid = id1_list[idx] if idx < len(id1_list) else f"t1_{idx}"
                key = ("T1", str(nid))
                if key not in unique_nodes:
                    unique_nodes[key] = {"p_val": p, "o_val": o, "low": args_dict['t1_low'], "high": args_dict['t1_high']}

            p2_list = parse_float_list(safe_extract_list(row_dict.get(args_dict['t2_proxy'])))
            o2_list = parse_float_list(safe_extract_list(row_dict.get(args_dict['t2_oracle'])))
            id2_list = safe_extract_list(row_dict.get(t2_id_col))
            
            for idx, (p, o) in enumerate(zip(p2_list, o2_list)):
                nid = id2_list[idx] if idx < len(id2_list) else f"t2_{idx}"
                key = ("T2", str(nid))
                if key not in unique_nodes:
                    unique_nodes[key] = {"p_val": p, "o_val": o, "low": args_dict['t2_low'], "high": args_dict['t2_high']}

        # ----------------------------------------------------
        # 步骤 2：构建验证队列
        # ----------------------------------------------------
        tier1_gray, tier2_outside = [], []
        for key, info in unique_nodes.items():
            p_val, low, high = info['p_val'], info['low'], info['high']
            if low <= p_val <= high:
                tier1_gray.append(key)
            else:
                dist_to_mid = abs(p_val - 0.5)
                tier2_outside.append((dist_to_mid, key))

        random.seed(42)
        random.shuffle(tier1_gray)
        tier2_outside.sort(key=lambda x: x[0])
        
        evaluation_queue = tier1_gray + [x[1] for x in tier2_outside]

        # ----------------------------------------------------
        # 步骤 3：消耗预算 B
        # ----------------------------------------------------
        oracle_cache = {}
        budget_used = 0
        for key in evaluation_queue:
            if budget_used >= budget_B: break
            oracle_cache[key] = unique_nodes[key]['o_val'] > 0.5
            budget_used += 1

        # ----------------------------------------------------
        # 步骤 4：评估全部实例
        # ----------------------------------------------------
        accepted_weight_sum = 0.0
        accepted_instances_count = 0

        for row_dict in records:
            weight = float(row_dict[weight_col])
            if weight <= 0: continue

            instance_passed = True

            p1_list = parse_float_list(safe_extract_list(row_dict.get(args_dict['t1_proxy'])))
            id1_list = safe_extract_list(row_dict.get(t1_id_col))
            for idx, p_val in enumerate(p1_list):
                nid = id1_list[idx] if idx < len(id1_list) else f"t1_{idx}"
                key = ("T1", str(nid))

                if key in oracle_cache: node_ok = oracle_cache[key]
                else:
                    low, high = args_dict['t1_low'], args_dict['t1_high']
                    if p_val < low: node_ok = False
                    elif p_val > high: node_ok = True
                    else: node_ok = p_val > ((low + high) / 2.0)

                if not node_ok:
                    instance_passed = False
                    break

            if not instance_passed: continue

            p2_list = parse_float_list(safe_extract_list(row_dict.get(args_dict['t2_proxy'])))
            id2_list = safe_extract_list(row_dict.get(t2_id_col))
            for idx, p_val in enumerate(p2_list):
                nid = id2_list[idx] if idx < len(id2_list) else f"t2_{idx}"
                key = ("T2", str(nid))

                if key in oracle_cache: node_ok = oracle_cache[key]
                else:
                    low, high = args_dict['t2_low'], args_dict['t2_high']
                    if p_val < low: node_ok = False
                    elif p_val > high: node_ok = True
                    else: node_ok = p_val > ((low + high) / 2.0)

                if not node_ok:
                    instance_passed = False
                    break

            if instance_passed:
                accepted_weight_sum += weight
                accepted_instances_count += 1

        return {
            "query_basename": q_basename,
            "T_hat_double_truncation": round(accepted_weight_sum, 4),
            "total_instances": len(df),
            "accepted_instances": accepted_instances_count,
            "budget_limit_B": budget_B,
            "budget_used": budget_used,
            "total_oracle_calls": budget_used
        }
    except Exception as e:
        print(f"\n[Error] 处理文件 {fname} 失败: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Double Truncation Baseline")
    # 【核心简化】：仅保留 parent_dataset 唯一定位数据集路径
    parser.add_argument("--parent_dataset", required=True, help="数据集目录名称 (如 amazon, parler, parler-E)")
    parser.add_argument("--agg-mode", dest="agg_mode", default="count", choices=["count", "sum"], help="聚合模式")
    parser.add_argument("--ablation_csv", required=True, help="消融实验 CSV 预算来源路径")
    
    parser.add_argument("--table1", default="product")
    parser.add_argument("--table1_proxy", dest="t1_proxy", default="ML3_proxy2_probability")
    parser.add_argument("--table1_oracle", dest="t1_oracle", default="ML3_oracle2_probability")
    parser.add_argument("--t1_ids", default="post_id_list")
    parser.add_argument("--t1_low", type=float, default=0.2)
    parser.add_argument("--t1_high", type=float, default=0.3)

    parser.add_argument("--table2", default="review")
    parser.add_argument("--table2_proxy", dest="t2_proxy", default="ML2_proxy2_probability")
    parser.add_argument("--table2_oracle", dest="t2_oracle", default="ML2_oracle1_probability")
    parser.add_argument("--t2_ids", default="comment_id_list")
    parser.add_argument("--t2_low", type=float, default=0.2)
    parser.add_argument("--t2_high", type=float, default=0.3)

    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--out_csv", default="Core_Double_Truncation_results.csv")
    
    args = parser.parse_args()

    print(f"[*] 正在加载专属 Oracle 预算 (budget_frac = 0.1)...")
    query_budgets = load_query_budgets(args.ablation_csv, target_frac=0.1)
    
    # 动态定位项目根目录 (无论在哪层启动都能精准推算)
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
    
    # 【核心简化路径】：直接精准指向 datasets/{parent_dataset}/
    base_dir = os.path.join(PROJECT_ROOT, "datasets", args.parent_dataset)
    
    agg_dir = os.path.join(base_dir, "results", f"aggregated_results_{args.agg_mode}")
    if not os.path.exists(agg_dir):
        agg_dir = os.path.join(base_dir, "results", "aggregated_results")
        if not os.path.exists(agg_dir):
            print(f"[Error] 找不到目录: {agg_dir}")
            return

    agg_files = sorted([f for f in os.listdir(agg_dir) if f.startswith("aggregated_list_") and f.endswith(".csv")])
    print(f"[*] 在 {agg_dir} 中找到 {len(agg_files)} 个核心实例文件。\n")

    out_path = os.path.join(base_dir, "results", "efficiency", args.out_csv)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fieldnames = [
        "query_basename", "T_hat_double_truncation", "total_instances", 
        "accepted_instances", "budget_limit_B", "budget_used", "total_oracle_calls"
    ]
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    args_dict = vars(args)
    completed_cnt = 0

    print(f"🚀 开始多进程评估 (Workers = {args.num_workers}, Dataset = {args.parent_dataset}, Mode = {args.agg_mode})...\n")

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(process_single_file, fname, agg_dir, query_budgets, args_dict): fname 
            for fname in agg_files
        }

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(agg_files), desc=f"Evaluating {args.parent_dataset}-{args.agg_mode}", ncols=100):
            res = future.result()
            if res is not None:
                with open(out_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(res)
                    f.flush() 
                completed_cnt += 1

    print(f"\n✅ [100% 预算对齐基线评估完成]！共处理并保存了 {completed_cnt} 个查询至:\n👉 {out_path}")

if __name__ == "__main__":
    main()