import os
# ==============================================================================
# 必须置于最顶端：防止 ProcessPoolExecutor 与 NumPy/BLAS 发生多线程死锁
# ==============================================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import ast
import json
import math
import csv
import random
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from tqdm import tqdm
import concurrent.futures

class ProjectionABaeSampler:
    def __init__(self, df: pd.DataFrame, config: dict, T_true: float = None, K: int = 5):
        self.T_true = T_true
        self.K = K
        self.config = config
        # 极低内存占用预处理
        self.instances = self._prepare_instances(df)

    def _prepare_instances(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: 
            return pd.DataFrame()
        df = df.copy()
        
        weight_col = "estimateW" if "estimateW" in df.columns else "a"
        df.rename(columns={weight_col: "a"}, inplace=True)
        df["a"] = pd.to_numeric(df["a"], errors="coerce").fillna(0.0)

        def fast_parse_num_list(val):
            if pd.isna(val): return []
            if isinstance(val, (list, tuple)): return [float(x) for x in val if pd.notna(x)]
            if isinstance(val, (int, float)): return [float(val)]
            s = str(val).strip()
            if not s or s in ("[]", "nan", "None"): return []
            if s.startswith('[') and s.endswith(']'):
                s = s[1:-1].strip()
                if not s: return []
                res = []
                for x in s.split(','):
                    x = x.strip().strip("'\"")
                    if x:
                        try: res.append(float(x))
                        except ValueError: pass
                return res
            try: return [float(s)]
            except ValueError: return []

        def compute_proxy_prod(val_str):
            lst = fast_parse_num_list(val_str)
            if not lst: return 1.0
            return float(np.prod(lst))

        p_proxy_col = self.config.get("t1_proxy")
        c_proxy_col = self.config.get("t2_proxy")

        p_proxies = df[p_proxy_col].apply(compute_proxy_prod) if (p_proxy_col and p_proxy_col in df.columns) else 1.0
        c_proxies = df[c_proxy_col].apply(compute_proxy_prod) if (c_proxy_col and c_proxy_col in df.columns) else 1.0

        df["proxy"] = p_proxies * c_proxies

        # 仅保留核心轻量列，释放显存与内存
        cols_to_keep = ["a", "proxy"]
        optional_cols = [
            self.config.get("t1_oracle"), self.config.get("t2_oracle"), 
            "post_id_list", "product_id_list", "comment_id_list", "review_id_list"
        ]
        for c in optional_cols:
            if c and c in df.columns:
                cols_to_keep.append(c)

        return df[df["a"] > 0][cols_to_keep].reset_index(drop=True)

    def _eval_oracle_strict(self, row: pd.Series, oracle_cache: Dict, budget_used: int, budget_limit: int) -> Tuple[int, int, bool]:
        """【即时解析机制】只在调用 Oracle 时才解析当前行"""
        def fast_parse_id_list(val):
            if pd.isna(val): return []
            if isinstance(val, (list, tuple)): return [str(x) for x in val]
            if isinstance(val, (int, float)): return [str(int(val))] if not np.isnan(val) else []
            s = str(val).strip()
            if not s or s in ("[]", "nan", "None"): return []
            if s.startswith('[') and s.endswith(']'):
                s = s[1:-1].strip()
                if not s: return []
                return [x.strip().strip("'\"") for x in s.split(',') if x.strip()]
            return [s]

        def fast_parse_num_list(val):
            if pd.isna(val): return []
            if isinstance(val, (list, tuple)): return [float(x) for x in val if pd.notna(x)]
            if isinstance(val, (int, float)): return [float(val)]
            s = str(val).strip()
            if not s or s in ("[]", "nan", "None"): return []
            if s.startswith('[') and s.endswith(']'):
                s = s[1:-1].strip()
                if not s: return []
                res = []
                for x in s.split(','):
                    x = x.strip().strip("'\"")
                    if x:
                        try: res.append(float(x))
                        except ValueError: pass
                return res
            try: return [float(s)]
            except ValueError: return []

        id1_col = "post_id_list" if "post_id_list" in row else "product_id_list"
        id2_col = "comment_id_list" if "comment_id_list" in row else "review_id_list"

        t1_ids = fast_parse_id_list(row.get(id1_col))
        t2_ids = fast_parse_id_list(row.get(id2_col))
        
        t1_probs = fast_parse_num_list(row.get(self.config.get("t1_oracle")))
        t2_probs = fast_parse_num_list(row.get(self.config.get("t2_oracle")))

        nodes_to_check = []
        for idx, prob in enumerate(t1_probs): 
            nid = t1_ids[idx] if idx < len(t1_ids) else f"t1_{idx}"
            nodes_to_check.append(("T1", str(nid), prob))
            
        for idx, prob in enumerate(t2_probs): 
            nid = t2_ids[idx] if idx < len(t2_ids) else f"t2_{idx}"
            nodes_to_check.append(("T2", str(nid), prob))

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
                    return 0, calls, True 

            if not ok:
                return 0, calls, False 
                
        return 1, calls, False

    def stratify_by_proxy_quantile(self, df: pd.DataFrame, K: int) -> pd.DataFrame:
        df = df.copy()
        # 按期望贡献 proxy * a 分层，保证层内方差极小
        exp_contrib = df["proxy"] * df["a"]
        try:
            df["stratum"] = pd.qcut(exp_contrib, K, labels=False, duplicates="drop")
        except Exception:
            df["stratum"] = pd.cut(exp_contrib.rank(method="first"), bins=K, labels=False)
        df["stratum"] = df["stratum"].fillna(0).astype(int)
        return df

    def run_abae_sum(self, target_budget_frac: float = 0.1, pilot_ratio: float = 0.1, budget_B: int = 500, random_seed: int = 42) -> Dict:
        """
        【严格无偏的两阶段 Pilot + Cost-Aware Neyman 分配采样】
        1. Stage 1: Uniform Pilot 获取经验方差与平均成本。
        2. Stage 2: Cost-Aware Neyman 动态配额，层内 Uniform 抽取剩余样本。
        3. 估计量: 严格两阶段 Horvitz-Thompson 展开: T_hat = sum(Pilot) + (N_rem / n2) * sum(Stage2)
        """
        if self.instances.empty:
            return {"T_hat_sum": 0.0, "ARE": 0.0, "Signed_RE": 0.0, "oracle_cost": 0}

        rng = np.random.default_rng(random_seed)
        df = self.stratify_by_proxy_quantile(self.instances, self.K)
        N_total_pop = len(df)
        
        # 1. 预算划分
        N_rows_budget = max(1, int(math.floor(target_budget_frac * N_total_pop)))
        # 当 pilot_ratio > 0 时启用 Pilot；若为 0 则默认每层至少抽 2 个做方差估计
        N1_pilot = max(self.K * 2, int(math.floor(N_rows_budget * (pilot_ratio if pilot_ratio > 0 else 0.1))))
        N1_pilot = min(N1_pilot, N_rows_budget)
        N2_stage2 = max(0, N_rows_budget - N1_pilot)

        oracle_cache = {}
        total_budget_used = 0
        budget_exhausted = False

        strata_groups = dict(list(df.groupby("stratum")))
        actual_K = len(strata_groups)
        n1_per_stratum = max(1, N1_pilot // actual_K)

        pilot_stats = {}
        pilot_y_vals = {k: [] for k in strata_groups}
        pilot_sampled_indices = {k: [] for k in strata_groups}

        # =================================================================
        # Stage 1: Pilot 采样 (Uniform)
        # =================================================================
        for k, grp in strata_groups.items():
            Nk = len(grp)
            n1_k = min(n1_per_stratum, Nk)
            
            sampled_grp = grp.sample(n=n1_k, random_state=rng.integers(0, 1 << 30))
            pilot_sampled_indices[k] = sampled_grp.index.tolist()
            
            cost_vals = []
            for _, row in sampled_grp.iterrows():
                if budget_exhausted: break
                is_valid, calls, is_out = self._eval_oracle_strict(row, oracle_cache, total_budget_used, budget_B)
                if is_out:
                    budget_exhausted = True
                    break
                
                total_budget_used += calls
                pilot_y_vals[k].append(row["a"] * is_valid)
                cost_vals.append(calls)
            
            y_vals = pilot_y_vals[k]
            sigma_hat_k = np.std(y_vals, ddof=1) if len(y_vals) > 1 else 0.0
            mean_cost_k = np.mean(cost_vals) if len(cost_vals) > 0 else 1.0
            
            # 方差平滑保底：若 pilot 未命中正样本，用 proxy*a 的先验方差保底
            if sigma_hat_k <= 1e-6:
                prior_sigma = float(np.std(grp["a"] * grp["proxy"]))
                if prior_sigma <= 1e-6:
                    prior_sigma = float(np.mean(grp["a"] * grp["proxy"])) * 0.5 + 1e-4
                smoothed_sigma = max(prior_sigma, 1e-4)
            else:
                smoothed_sigma = sigma_hat_k
            
            pilot_stats[k] = {
                "Nk": Nk, 
                "n1": len(y_vals),
                "smoothed_sigma": smoothed_sigma, 
                "mean_cost": max(mean_cost_k, 0.1)
            }

        # =================================================================
        # Stage 2: Cost-Aware Neyman 分配
        # =================================================================
        alloc_stage2 = {k: 0 for k in strata_groups}
        if not budget_exhausted and N2_stage2 > 0:
            alloc_weights = {
                k: ((st["Nk"] - st["n1"]) * st["smoothed_sigma"]) / math.sqrt(st["mean_cost"]) 
                for k, st in pilot_stats.items()
            }
            sum_w = sum(alloc_weights.values())
            if sum_w > 0:
                for k in pilot_stats:
                    ratio = alloc_weights[k] / sum_w
                    alloc_stage2[k] = int(math.floor(N2_stage2 * ratio))
            else:
                for k in pilot_stats:
                    alloc_stage2[k] = N2_stage2 // actual_K

            # 余数分配
            rem = N2_stage2 - sum(alloc_stage2.values())
            if rem > 0 and sum_w > 0:
                remainders = {k: (N2_stage2 * (alloc_weights[k] / sum_w) - alloc_stage2[k]) for k in pilot_stats}
                for k in sorted(remainders, key=remainders.get, reverse=True):
                    if rem <= 0: break
                    rem_len = pilot_stats[k]["Nk"] - pilot_stats[k]["n1"]
                    if alloc_stage2[k] < rem_len:
                        alloc_stage2[k] += 1
                        rem -= 1

        stage2_pool = []
        if not budget_exhausted:
            for k, grp in strata_groups.items():
                remaining_grp = grp.drop(index=pilot_sampled_indices[k], errors="ignore")
                n2_k = min(alloc_stage2.get(k, 0), len(remaining_grp))
                if n2_k > 0:
                    sampled_grp = remaining_grp.sample(n=n2_k, random_state=rng.integers(0, 1 << 30))
                    for _, row in sampled_grp.iterrows():
                        stage2_pool.append((k, row))
            
            # 全局打乱以防止按层串行截断引起的偏差
            random.seed(random_seed)
            random.shuffle(stage2_pool)

        stage2_y_vals = {k: [] for k in strata_groups}
        for k, row in stage2_pool:
            if budget_exhausted: break
            is_valid, calls, is_out = self._eval_oracle_strict(row, oracle_cache, total_budget_used, budget_B)
            if is_out:
                budget_exhausted = True
                break 
            total_budget_used += calls
            stage2_y_vals[k].append(row["a"] * is_valid)

        # =================================================================
        # 严格无偏两阶段 Horvitz-Thompson 展开估计
        # =================================================================
        total_sum_hat = 0.0

        for k, grp in strata_groups.items():
            Nk = len(grp)
            p_vals = pilot_y_vals[k]
            s2_vals = stage2_y_vals[k]
            
            n1_k = len(p_vals)
            n2_k = len(s2_vals)
            
            if n1_k == 0:
                continue
                
            y1_sum = sum(p_vals)
            if n2_k > 0:
                # 严格无偏：已抽 Pilot 确定值 + 剩余部分乘以 Stage2 样本均值
                y2_mean = sum(s2_vals) / n2_k
                t_hat_k = y1_sum + (Nk - n1_k) * y2_mean
            else:
                # 若第二阶段未抽到/预算耗尽，退化为第一阶段的扩张估计
                t_hat_k = (Nk / n1_k) * y1_sum
                
            total_sum_hat += t_hat_k

        t_true_safe = self.T_true if self.T_true and self.T_true > 0 else 1e-9
        signed_re = (total_sum_hat - t_true_safe) / t_true_safe
        are = abs(total_sum_hat - t_true_safe) / t_true_safe

        return {
            "T_hat_sum": float(total_sum_hat),
            "ARE": float(are),
            "Signed_RE": float(signed_re),
            "oracle_cost": int(total_budget_used)
        }

def process_single_task(fname, run_id, agg_dir, gt_map, query_budgets, config, pilot_ratio):
    try:
        q_clean = fname.replace("aggregated_list_", "").replace("aggregated_wide_", "").replace(".csv", "")
        q_basename = q_clean + ".graph"
        T_true = gt_map.get(q_clean)

        if T_true is None: return None

        budget_B = query_budgets.get(q_basename, 500)
        filepath = os.path.join(agg_dir, fname)
        df = pd.read_csv(filepath)

        sampler = ProjectionABaeSampler(df=df, config=config, T_true=T_true, K=config["K"])
        seed = (abs(hash(q_basename)) % 99999999) + run_id * 10007

        res = sampler.run_abae_sum(
            target_budget_frac=config["budget_frac"],
            pilot_ratio=pilot_ratio,  
            budget_B=budget_B,
            random_seed=seed
        )

        return {
            "query_basename": q_basename, "run_id": run_id,
            "T_hat_abae": res["T_hat_sum"], "T_true": T_true,
            "Signed_RE": res["Signed_RE"], "ARE": res["ARE"],
            "oracle_cost": res["oracle_cost"], "budget_limit_B": budget_B
        }
    except Exception as e:
        print(f"\n[Error] 处理文件 {fname} (Run {run_id}) 失败: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="/home/wangshuo/projects/PROXY")
    parser.add_argument("--dataset", default="parler")
    parser.add_argument("--agg_mode", choices=["count", "sum"], default="count")
    parser.add_argument("--budget_frac", type=float, default=0.1)
    parser.add_argument("--pilot_ratio", type=float, default=0.1)  # 默认使用 10% 做 Pilot
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out_csv", default="Projection_ABae_results_count.csv")
    args = parser.parse_args()

    config = {
        "K": args.K,
        "budget_frac": args.budget_frac,
        "t1_proxy": "ML1_proxy4b_probability",
        "t2_proxy": "ML2_proxy1_probability",
        "t1_oracle": "ML1_oracle2_probability",
        "t2_oracle": "ML2_oracle2_probability"
    }

    if "amazon" in args.dataset:
        config.update({
            "t1_proxy": "ML3_proxy2_probability", "t2_proxy": "ML2_proxy2_probability",
            "t1_oracle": "ML3_oracle2_probability", "t2_oracle": "ML2_oracle1_probability"
        })

    dataset_path = os.path.join(args.base_dir, "datasets", args.dataset)
    results_dir = os.path.join(dataset_path, "results")
    
    agg_dir = os.path.join(results_dir, f"aggregated_results_{args.agg_mode}")
    if not os.path.exists(agg_dir): 
        agg_dir = os.path.join(results_dir, "aggregated_results")
        
    ablation_csv_path = os.path.join(results_dir, "efficiency", f"allocation_strategy_comparison_ablation_{args.agg_mode}.csv")
    if not os.path.exists(ablation_csv_path): 
        ablation_csv_path = os.path.join(results_dir, "efficiency", f"allocation_strategy_comparison_{args.agg_mode}.csv")
        
    out_csv_path = os.path.join(results_dir, "efficiency", args.out_csv)

    gt_cands = [
        os.path.join(results_dir, f"T_true_ML3_oracle2_probability_ML2_oracle1_probability_{args.agg_mode}.json"),
        os.path.join(results_dir, f"T_true_ML1_oracle2_probability_ML2_oracle2_probability_{args.agg_mode}.json")
    ]
    gt_path = next((p for p in gt_cands if os.path.exists(p)), None)

    if not gt_path: 
        print(f"[Error] 未找到对应的 Ground Truth 文件: {results_dir}")
        return

    print("=" * 65)
    print(f"🚀 启动无偏 Projection-ABae 基线 (Pilot={args.pilot_ratio*100}% | Cost-Aware Neyman 分配)")
    print(f"   数据集: {args.dataset} | 模式: {args.agg_mode.upper()} | Runs: {args.runs}")
    print("=" * 65)

    with open(gt_path, 'r') as f: 
        gt_dict = json.load(f)
    gt_map = {str(k).replace(".graph", ""): float(v) for k, v in gt_dict.items() if v is not None and v > 0}

    query_budgets = {}
    if os.path.exists(ablation_csv_path):
        with open(ablation_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frac = float(row.get('budget_frac', 0))
                if abs(frac - args.budget_frac) < 1e-4 and row.get('method') in ['POSS', '8_POSSA']:
                    query_budgets.setdefault(row['query_basename'].strip(), []).append(int(float(row['oracle_cost'])))
        query_budgets = {q: int(round(sum(c)/len(c))) for q, c in query_budgets.items()}

    agg_files = sorted([f for f in os.listdir(agg_dir) if f.startswith("aggregated_list_") and f.endswith(".csv")])

    fieldnames = ["query_basename", "run_id", "T_hat_abae", "T_true", "Signed_RE", "ARE", "oracle_cost", "budget_limit_B"]
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    with open(out_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    completed_cnt = 0
    total_tasks = len(agg_files) * args.runs

    print(f"🚀 开始多进程独立评估 (Workers = {args.workers}, Tasks = {total_tasks})...\n")

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_task, fname, r, agg_dir, gt_map, query_budgets, config, args.pilot_ratio): (fname, r) 
                   for fname in agg_files for r in range(1, args.runs + 1)}

        for future in tqdm(concurrent.futures.as_completed(futures), total=total_tasks, desc="Evaluating", ncols=100):
            res = future.result()
            if res:
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
        print(f"📊 Projection-ABae (Pilot + Neyman) 无偏最终评估")
        print("=" * 65)
        print(f"1. 带符号误差均值 (Mean Signed RE): {mean_sre:.4f} ({mean_sre*100:.2f}%)")
        print(f"2. 绝对误差均值   (Mean ARE)      : {mean_are:.4f} ({mean_are*100:.2f}%)")
        print("=" * 65)

if __name__ == "__main__":
    main()