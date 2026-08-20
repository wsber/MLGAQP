import os
import ast
import json
import math
import csv
import sys
import random
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from tqdm import tqdm
import concurrent.futures

# ==============================================================================
# 1. 动态自动推导项目根目录 (自动定位至 .../PROXY 根目录)
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 向上追溯找到包含 datasets 目录的真实项目根目录
DEFAULT_PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))

if DEFAULT_PROJECT_ROOT not in sys.path:
    sys.path.append(DEFAULT_PROJECT_ROOT)

class ProjectionABaeSampler:
    def __init__(self, df: pd.DataFrame, T_true: float = None, K: int = 5,
                 post_proxy: str = "ML3_proxy2_probability",
                 comment_proxy: str = "ML2_proxy2_probability",
                 post_oracle: str = "ML3_oracle2_probability",
                 comment_oracle: str = "ML2_oracle1_probability"):
        self.T_true = T_true
        self.K = K
        self.instances = self._prepare_instances(df, post_proxy, comment_proxy, post_oracle, comment_oracle)

    def _prepare_instances(self, df: pd.DataFrame, p_proxy_col: str, c_proxy_col: str,
                           p_oracle_col: str, c_oracle_col: str) -> pd.DataFrame:
        if df.empty: return pd.DataFrame()
        df = df.copy()
        weight_col = "estimateW" if "estimateW" in df.columns else "a"
        df.rename(columns={weight_col: "a"}, inplace=True)
        df["a"] = pd.to_numeric(df["a"], errors="coerce").fillna(0.0)

        def safe_extract_list(val):
            if pd.isna(val) or val == "": return []
            if isinstance(val, (list, tuple)): return list(val)
            if isinstance(val, (int, float)): return [val]
            if isinstance(val, str):
                val = val.strip()
                if val in ["", "nan", "[]"]: return []
                if val.startswith('[') and val.endswith(']'):
                    try:
                        res = json.loads(val.replace("'", '"'))
                        return res if isinstance(res, list) else [res]
                    except Exception:
                        try:
                            res = ast.literal_eval(val)
                            return res if isinstance(res, list) else [res]
                        except Exception: return []
                else:
                    return [val]
            return []

        def to_num_list(lst):
            out = []
            for x in lst:
                try: out.append(float(x))
                except (ValueError, TypeError): pass
            return out

        id1_col = "post_id_list" if "post_id_list" in df.columns else "product_id_list"
        id2_col = "comment_id_list" if "comment_id_list" in df.columns else "review_id_list"
        df["t1_ids"] = df[id1_col].apply(safe_extract_list) if id1_col in df.columns else [[] for _ in range(len(df))]
        df["t2_ids"] = df[id2_col].apply(safe_extract_list) if id2_col in df.columns else [[] for _ in range(len(df))]

        p_proxy_list = df[p_proxy_col].apply(safe_extract_list).apply(to_num_list) if p_proxy_col in df.columns else df["a"].apply(lambda x: [])
        c_proxy_list = df[c_proxy_col].apply(safe_extract_list).apply(to_num_list) if c_proxy_col in df.columns else df["a"].apply(lambda x: [])

        df["proxy"] = p_proxy_list.apply(lambda l: float(np.prod(l)) if len(l)>0 else 1.0) * \
                      c_proxy_list.apply(lambda l: float(np.prod(l)) if len(l)>0 else 1.0)

        df["t1_oracle_probs"] = df[p_oracle_col].apply(safe_extract_list).apply(to_num_list) if p_oracle_col in df.columns else df["a"].apply(lambda x: [])
        df["t2_oracle_probs"] = df[c_oracle_col].apply(safe_extract_list).apply(to_num_list) if c_oracle_col in df.columns else df["a"].apply(lambda x: [])

        instances = df[df["a"] > 0].reset_index(drop=True)
        return instances

    def _eval_oracle_strict(self, row: pd.Series, oracle_cache: Dict, budget_used: int, budget_limit: int) -> Tuple[int, int, bool]:
        t1_ids, t2_ids = row.get("t1_ids", []), row.get("t2_ids", [])
        t1_probs, t2_probs = row.get("t1_oracle_probs", []), row.get("t2_oracle_probs", [])

        nodes_to_check = []
        for nid, prob in zip(t1_ids, t1_probs): nodes_to_check.append(("T1", str(nid), prob))
        for nid, prob in zip(t2_ids, t2_probs): nodes_to_check.append(("T2", str(nid), prob))

        calls = 0
        for t_name, nid, o_prob in nodes_to_check:
            key = (t_name, nid)
            if key in oracle_cache:
                ok = oracle_cache[key]
            else:
                if budget_used + calls < budget_limit:
                    ok = float(o_prob) > 0.5
                    oracle_cache[key] = ok
                    calls += 1
                else:
                    return 0, calls, True # 预算耗尽熔断

            if not ok:
                return 0, calls, False 
                
        return 1, calls, False

    def stratify_by_proxy_quantile(self, df: pd.DataFrame, K: int) -> pd.DataFrame:
        df = df.copy()
        try:
            df["stratum"] = pd.qcut(df["proxy"], K, labels=False, duplicates="drop")
        except Exception:
            df["stratum"] = pd.cut(df["proxy"].rank(method="first"), bins=K, labels=False)
        df["stratum"] = df["stratum"].fillna(0).astype(int)
        return df

    def run_abae_sum(self, target_budget_frac: float = 0.1, budget_B: int = 500, random_seed: int = 42) -> Dict:
        if self.instances.empty:
            return {"T_hat_sum": 0.0, "ARE": 0.0, "Signed_RE": 0.0, "oracle_cost": 0}

        rng = np.random.default_rng(random_seed)
        df = self.stratify_by_proxy_quantile(self.instances, self.K)
        N_total_pop = len(df)
        
        oracle_cache = {}
        total_budget_used = 0
        budget_exhausted = False

        strata_groups = dict(list(df.groupby("stratum")))
        shuffled_grps = {k: grp.sample(frac=1.0, random_state=random_seed+k) for k, grp in strata_groups.items()}
        grp_iters = {k: shuffled_grps[k].iterrows() for k in strata_groups}
        
        samples = {k: [] for k in strata_groups}
        active_strata = list(strata_groups.keys())

        # 全局轮询无偏抽样
        while active_strata and not budget_exhausted:
            rng.shuffle(active_strata)
            
            for k in list(active_strata):
                if budget_exhausted: break
                
                try:
                    idx, row = next(grp_iters[k])
                    is_valid, calls, is_out = self._eval_oracle_strict(row, oracle_cache, total_budget_used, budget_B)
                    
                    if is_out:
                        budget_exhausted = True
                        break 
                        
                    total_budget_used += calls
                    samples[k].append(row["a"] * is_valid)
                    
                except StopIteration:
                    active_strata.remove(k)

        # 分层汇总估计
        total_sum_hat = 0.0
        sampled_N = 0  

        for k, grp in strata_groups.items():
            Nk = len(grp)
            y_list = samples[k]
            nk_total = len(y_list)

            if nk_total > 0:
                mean_y_k = np.mean(y_list)
                total_sum_hat += mean_y_k * Nk
                sampled_N += Nk
                
        # 补偿低预算下未采样层的人口权重
        if 0 < sampled_N < N_total_pop:
            total_sum_hat = total_sum_hat * (N_total_pop / sampled_N)

        t_true_safe = self.T_true if self.T_true and self.T_true > 0 else 1e-9
        signed_re = (total_sum_hat - t_true_safe) / t_true_safe
        are = abs(total_sum_hat - t_true_safe) / t_true_safe

        return {
            "T_hat_sum": float(total_sum_hat),
            "ARE": float(are),
            "Signed_RE": float(signed_re),
            "oracle_cost": int(total_budget_used)
        }

def process_single_task(fname, run_id, agg_dir, gt_map, query_budgets, args_dict):
    """单个查询任务执行单元"""
    try:
        q_clean = fname.replace("aggregated_list_", "").replace(".csv", "")
        q_basename = q_clean + ".graph"
        T_true = gt_map.get(q_clean)

        if T_true is None: return None

        budget_B = query_budgets.get(q_basename, 500)
        filepath = os.path.join(agg_dir, fname)
        df = pd.read_csv(filepath)

        sampler = ProjectionABaeSampler(
            df=df, T_true=T_true, K=args_dict["K"],
            post_proxy=args_dict["t1_proxy"], comment_proxy=args_dict["t2_proxy"],
            post_oracle=args_dict["t1_oracle"], comment_oracle=args_dict["t2_oracle"]
        )

        # 确保随机种子为非负整数
        seed = (abs(hash(q_basename)) % 99999999) + run_id

        res = sampler.run_abae_sum(
            target_budget_frac=args_dict["budget_frac"],
            budget_B=budget_B,
            random_seed=seed
        )

        return {
            "query_basename": q_basename, 
            "run_id": run_id,
            "T_hat_abae": res["T_hat_sum"], 
            "T_true": T_true,
            "Signed_RE": res["Signed_RE"], 
            "ARE": res["ARE"],
            "oracle_cost": res["oracle_cost"], 
            "budget_limit_B": budget_B
        }
    except Exception as e:
        print(f"\n[Error] 处理文件 {fname} (Run {run_id}) 失败: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Strict Projection-ABae Single-Run Evaluator (Amazon SUM)")
    parser.add_argument("--base_dir", type=str, default=DEFAULT_PROJECT_ROOT, help="项目根目录")
    parser.add_argument("--dataset_name", type=str, default="amazon", help="数据集文件夹名 (如 amazon, parler)")
    parser.add_argument("--ablation_csv", type=str, default=None, help="预算对齐消融 CSV 路径 (可选)")
    
    # 针对 Amazon 默认配置的列名
    parser.add_argument("--t1_proxy", default="ML3_proxy2_probability")
    parser.add_argument("--t1_oracle", default="ML3_oracle2_probability")
    parser.add_argument("--t2_proxy", default="ML2_proxy2_probability")
    parser.add_argument("--t2_oracle", default="ML2_oracle1_probability")
    
    parser.add_argument("--budget_frac", type=float, default=0.1, help="采样率 (默认 0.1)")
    parser.add_argument("--K", type=int, default=5, help="分层数")
    parser.add_argument("--runs", type=int, default=1, help="重复运行次数 (单次运行设为 1)")
    parser.add_argument("--workers", type=int, default=16, help="并行进程数")
    parser.add_argument("--out_csv", default="Projection_ABae_amazon_sum.csv", help="输出 CSV 名称")
    args = parser.parse_args()

    # 动态路径解析
    dataset_path = os.path.join(args.base_dir, "datasets", args.dataset_name)
    results_dir = os.path.join(dataset_path, "results")
    
    # 自动定位核心实例目录 (优先带 _sum 后缀)
    agg_dir = os.path.join(results_dir, "aggregated_results_sum")
    if not os.path.exists(agg_dir):
        agg_dir = os.path.join(results_dir, "aggregated_results")

    # 自动定位消融预算 CSV
    if args.ablation_csv is None:
        c1 = os.path.join(results_dir, "efficiency", "allocation_strategy_comparison_ablation_sum.csv")
        c2 = os.path.join(results_dir, "efficiency", "allocation_strategy_comparison_sum.csv")
        ablation_csv_path = c1 if os.path.exists(c1) else c2
    else:
        ablation_csv_path = args.ablation_csv

    # 自动定位 Ground Truth JSON
    gt_candidates = [
        os.path.join(results_dir, "T_true_ML3_oracle2_probability_ML2_oracle1_probability_sum.json"),
        os.path.join(results_dir, "T_true_ML1_oracle2_probability_ML2_oracle2_probability_sum.json"),
        os.path.join(results_dir, "T_true_ML3_oracle2_probability_ML2_oracle1_probability.json")
    ]
    gt_sum_path = next((p for p in gt_candidates if os.path.exists(p)), None)

    if not gt_sum_path or not os.path.exists(gt_sum_path):
        print(f"[Error] 找不到真值 JSON 文件于: {results_dir}")
        return

    out_csv_path = os.path.join(results_dir, "efficiency", args.out_csv)

    print("=" * 65)
    print(f"🚀 开始执行 Projection-ABae 评估 (Amazon SUM, Runs={args.runs})")
    print(f"   • 项目根目录 : {args.base_dir}")
    print(f"   • 数据集路径 : {dataset_path}")
    print(f"   • 核心实例库 : {agg_dir}")
    print(f"   • 预算源 CSV : {ablation_csv_path}")
    print(f"   • 结果输出至 : {out_csv_path}")
    print("=" * 65)

    with open(gt_sum_path, 'r') as f: gt_dict = json.load(f)
    gt_map = {str(k).replace(".graph", ""): float(v) for k, v in gt_dict.items() if v is not None and v > 0}

    query_budgets = {}
    if os.path.exists(ablation_csv_path):
        with open(ablation_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frac = float(row.get('budget_frac', 0))
                if abs(frac - args.budget_frac) < 1e-4 and row.get('method') in ['POSS', '8_POSSA']:
                    q_name = row['query_basename'].strip()
                    cost = int(float(row['oracle_cost']))
                    query_budgets.setdefault(q_name, []).append(cost)
        query_budgets = {q: int(round(sum(c)/len(c))) for q, c in query_budgets.items()}

    agg_files = sorted([f for f in os.listdir(agg_dir) if f.startswith("aggregated_list_") and f.endswith(".csv")])

    fieldnames = ["query_basename", "run_id", "T_hat_abae", "T_true", "Signed_RE", "ARE", "oracle_cost", "budget_limit_B"]
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    with open(out_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    args_dict = vars(args)
    completed_cnt = 0
    total_tasks = len(agg_files) * args.runs

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for fname in agg_files:
            for run_id in range(1, args.runs + 1):
                future = executor.submit(process_single_task, fname, run_id, agg_dir, gt_map, query_budgets, args_dict)
                futures[future] = (fname, run_id)

        for future in tqdm(concurrent.futures.as_completed(futures), total=total_tasks, desc="Evaluating ABae", ncols=100):
            res = future.result()
            if res is not None:
                with open(out_csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(res)
                    f.flush() 
                completed_cnt += 1
                
    res_df = pd.read_csv(out_csv_path)
    if not res_df.empty:
        mean_sre = res_df['Signed_RE'].mean()
        mean_are = res_df['ARE'].mean()
        print("\n" + "=" * 65)
        print(f"📊 Projection-ABae (Amazon SUM) 最终评估结果")
        print("=" * 65)
        print(f"有效评估查询数 : {len(res_df)}")
        print(f"1. 带符号误差均值 (Mean Signed RE): {mean_sre:.4f} ({mean_sre*100:.2f}%)")
        print(f"2. 绝对误差均值   (Mean ARE)      : {mean_are:.4f} ({mean_are*100:.2f}%)")
        print(f"3. 绝对误差中位数 (P50 ARE)       : {res_df['ARE'].median():.4f} ({res_df['ARE'].median()*100:.2f}%)")
        print(f"4. 绝对误差 P90   (P90 ARE)       : {res_df['ARE'].quantile(0.90):.4f} ({res_df['ARE'].quantile(0.90)*100:.2f}%)")
        print("=" * 65)

if __name__ == "__main__":
    main()