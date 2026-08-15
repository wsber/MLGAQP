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

# python PSF.py   --parent_dataset amazon_data   --dataset amazon_extend   --ablation_csv /home/wangshuo/resource/datasets/amazon_data/amazon_extend/results/efficiency/allocation_strategy_comparison_ablation_sum.csv   --table1 product   --table1_proxy ML3_proxy2_probability   --table1_oracle ML3_oracle2_probability   --t1_ids post_id_list   --t1_low 0.4   --t1_high 0.6   --table2 review   --table2_proxy ML2_proxy2_probability   --table2_oracle ML2_oracle1_probability   --t2_ids comment_id_list   --t2_low 0.4   --t2_high 0.6   --num_workers 16   --out_csv Core_Double_Truncation_amazon_sum.csv

def safe_extract_list(val):
    """【极速且稳健版】：支持标量与列表，不强制转 float 防止 ID 报错"""
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
    """安全地将列表元素转换为 float"""
    out = []
    for x in lst:
        try:
            out.append(float(x))
        except (ValueError, TypeError):
            pass
    return out

def load_query_budgets(csv_path, target_frac=0.1, target_method="8_POSSA"):
    """从消融实验 CSV 中读取每个查询在 frac=0.1 下的专属 oracle_cost 预算"""
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
                if q_name not in query_budgets: query_budgets[q_name] = []
                query_budgets[q_name].append(cost)
            except Exception:
                continue
                
    return {q: int(round(sum(c)/len(c))) for q, c in query_budgets.items()}

def process_single_file(fname, agg_dir, query_budgets, args_dict):
    """【子进程核心函数】100% 预算用满策略 + 纯内存计算"""
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
        # 步骤 1：收集当前查询中涉及的所有唯一物理节点
        # ----------------------------------------------------
        unique_nodes = {}

        for row_dict in records:
            # Table 1 节点
            p1_list = parse_float_list(safe_extract_list(row_dict.get(args_dict['t1_proxy'])))
            o1_list = parse_float_list(safe_extract_list(row_dict.get(args_dict['t1_oracle'])))
            id1_list = safe_extract_list(row_dict.get(args_dict['t1_ids']))
            for idx, (p, o) in enumerate(zip(p1_list, o1_list)):
                nid = id1_list[idx] if idx < len(id1_list) else f"t1_{idx}"
                key = ("T1", str(nid))
                if key not in unique_nodes:
                    unique_nodes[key] = {
                        "p_val": p, "o_val": o,
                        "low": args_dict['t1_low'], "high": args_dict['t1_high']
                    }

            # Table 2 节点
            p2_list = parse_float_list(safe_extract_list(row_dict.get(args_dict['t2_proxy'])))
            o2_list = parse_float_list(safe_extract_list(row_dict.get(args_dict['t2_oracle'])))
            id2_list = safe_extract_list(row_dict.get(args_dict['t2_ids']))
            for idx, (p, o) in enumerate(zip(p2_list, o2_list)):
                nid = id2_list[idx] if idx < len(id2_list) else f"t2_{idx}"
                key = ("T2", str(nid))
                if key not in unique_nodes:
                    unique_nodes[key] = {
                        "p_val": p, "o_val": o,
                        "low": args_dict['t2_low'], "high": args_dict['t2_high']
                    }

        # ----------------------------------------------------
        # 步骤 2：构建优先级验证队列 (保证 100% 用满预算 B)
        # ----------------------------------------------------
        tier1_gray = []      # 第一优先级：灰色地带节点 [low, high]
        tier2_outside = []   # 第二优先级：区间外节点 (< low 或 > high)

        for key, info in unique_nodes.items():
            p_val, low, high = info['p_val'], info['low'], info['high']
            if low <= p_val <= high:
                tier1_gray.append(key)
            else:
                # 离 0.5 越近说明代理模型越不确定，越优先调用 Oracle 纠错
                dist_to_mid = abs(p_val - 0.5)
                tier2_outside.append((dist_to_mid, key))

        # 灰色节点随机打乱
        random.seed(42)
        random.shuffle(tier1_gray)

        # 区间外节点按距离 0.5 从近到远排序
        tier2_outside.sort(key=lambda x: x[0])
        tier2_keys = [x[1] for x in tier2_outside]

        # 终极验证队列：灰色地带排最前，不够用时区间外节点补上！
        evaluation_queue = tier1_gray + tier2_keys

        # ----------------------------------------------------
        # 步骤 3：严格调用 Oracle 校验直到预算 B 耗尽
        # ----------------------------------------------------
        oracle_cache = {}
        budget_used = 0

        for key in evaluation_queue:
            if budget_used >= budget_B:
                break
            info = unique_nodes[key]
            # 调用 Oracle 验证 (> 0.5 为合格)
            oracle_cache[key] = info['o_val'] > 0.5
            budget_used += 1

        # ----------------------------------------------------
        # 步骤 4：评估所有核心实例
        # ----------------------------------------------------
        accepted_weight_sum = 0.0
        accepted_instances_count = 0

        for row_dict in records:
            weight = float(row_dict[weight_col])
            if weight <= 0:
                continue

            instance_passed = True

            # 校验 Table 1 节点
            p1_list = parse_float_list(safe_extract_list(row_dict.get(args_dict['t1_proxy'])))
            id1_list = safe_extract_list(row_dict.get(args_dict['t1_ids']))
            for idx, p_val in enumerate(p1_list):
                nid = id1_list[idx] if idx < len(id1_list) else f"t1_{idx}"
                key = ("T1", str(nid))

                if key in oracle_cache:
                    node_ok = oracle_cache[key] # 命中 Oracle 结果
                else:
                    # 没分到 Oracle 预算的区间外节点，使用标准硬门槛判定
                    low, high = args_dict['t1_low'], args_dict['t1_high']
                    if p_val < low:
                        node_ok = False
                    elif p_val > high:
                        node_ok = True
                    else:
                        node_ok = p_val > ((low + high) / 2.0)

                if not node_ok:
                    instance_passed = False
                    break

            if not instance_passed:
                continue

            # 校验 Table 2 节点
            p2_list = parse_float_list(safe_extract_list(row_dict.get(args_dict['t2_proxy'])))
            id2_list = safe_extract_list(row_dict.get(args_dict['t2_ids']))
            for idx, p_val in enumerate(p2_list):
                nid = id2_list[idx] if idx < len(id2_list) else f"t2_{idx}"
                key = ("T2", str(nid))

                if key in oracle_cache:
                    node_ok = oracle_cache[key] # 命中 Oracle 结果
                else:
                    low, high = args_dict['t2_low'], args_dict['t2_high']
                    if p_val < low:
                        node_ok = False
                    elif p_val > high:
                        node_ok = True
                    else:
                        node_ok = p_val > ((low + high) / 2.0)

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
    parser = argparse.ArgumentParser(description="Double Truncation Baseline on Core Instances")
    parser.add_argument("--parent_dataset", default="amazon_data")
    parser.add_argument("--dataset", required=True, help="数据集 (e.g., amazon_extend)")
    parser.add_argument("--fastest_bin", default="") 
    parser.add_argument("--ablation_csv", required=True, help="消融实验 CSV 文件路径")
    
    # 兼容长短参数
    parser.add_argument("--table1", default="product")
    parser.add_argument("--t1_proxy", "--table1_proxy", dest="t1_proxy", default="ML3_proxy2_probability")
    parser.add_argument("--t1_oracle", "--table1_oracle", dest="t1_oracle", default="ML3_oracle2_probability")
    parser.add_argument("--t1_ids", "--table1_ids", dest="t1_ids", default="post_id_list")
    parser.add_argument("--t1_low", type=float, default=0.2)
    parser.add_argument("--t1_high", type=float, default=0.3)

    parser.add_argument("--table2", default="review")
    parser.add_argument("--t2_proxy", "--table2_proxy", dest="t2_proxy", default="ML2_proxy2_probability")
    parser.add_argument("--t2_oracle", "--table2_oracle", dest="t2_oracle", default="ML2_oracle1_probability")
    parser.add_argument("--t2_ids", "--table2_ids", dest="t2_ids", default="comment_id_list")
    parser.add_argument("--t2_low", type=float, default=0.2)
    parser.add_argument("--t2_high", type=float, default=0.3)

    parser.add_argument("--sum_col", default="price")
    parser.add_argument("--sum_label", default="12")
    
    parser.add_argument("--num_workers", type=int, default=8, help="并行处理的核心数")
    parser.add_argument("--out_csv", default="Core_Double_Truncation_results.csv", help="最终结果输出 CSV")
    
    args = parser.parse_args()

    print(f"[*] 正在加载专属 Oracle 预算 (budget_frac = 0.1)...")
    query_budgets = load_query_budgets(args.ablation_csv, target_frac=0.1)
    print(f"[+] 成功匹配 {len(query_budgets)} 个查询预算。")

    base_dir = f"/home/wangshuo/resource/datasets/{args.parent_dataset}/{args.dataset}"
    agg_dir = os.path.join(base_dir, "results", "aggregated_results")
    if not os.path.exists(agg_dir):
        print(f"[Error] 找不到目录: {agg_dir}")
        return

    agg_files = sorted([f for f in os.listdir(agg_dir) if f.startswith("aggregated_list_") and f.endswith(".csv")])
    print(f"[*] 找到 {len(agg_files)} 个核心实例文件。\n")

    if args.out_csv.startswith("/"):
        out_path = args.out_csv
    else:
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

    print(f"🚀 开始多进程极速评估 (Workers = {args.num_workers})...\n")

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(process_single_file, fname, agg_dir, query_budgets, args_dict): fname 
            for fname in agg_files
        }

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(agg_files), desc="Progress", ncols=100):
            res = future.result()
            if res is not None:
                with open(out_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(res)
                    f.flush() 
                completed_cnt += 1

    print(f"\n✅ [100% 预算对齐基线评估完成]！共处理并即时保存了 {completed_cnt} 个查询。")
    print(f"📁 最终结果路径: {out_path}")

if __name__ == "__main__":
    main()