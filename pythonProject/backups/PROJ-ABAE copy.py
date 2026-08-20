import os
import ast
import json
import math
import csv
import random  
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from tqdm import tqdm
import concurrent.futures

# python PROJ-ABAE.py \
#   --base_dir /home/wangshuo/projects/PROXY \
#   --dataset amazon \
#   --agg_mode sum \
#   --budget_frac 0.1 \
#   --pilot_ratio 0.2 \
#   --K 5 \
#   --runs 5 \
#   --workers 16

# ==============================================================================
# 全局数据集模型配置
# ==============================================================================
WORKLOADS_CONFIG = {
    "parler": {
        "t1_proxy": "ML1_proxy4b_probability", "t1_oracle": "ML1_oracle2_probability",
        "t2_proxy": "ML2_proxy1_probability",  "t2_oracle": "ML2_oracle2_probability"
    },
    "parler-E": {
        "t1_proxy": "ML1_proxy4b_probability", "t1_oracle": "ML1_oracle2_probability",
        "t2_proxy": "ML2_proxy1_probability",  "t2_oracle": "ML2_oracle2_probability"
    },
    "amazon": {
        "t1_proxy": "ML3_proxy2_probability",  "t1_oracle": "ML3_oracle2_probability",
        "t2_proxy": "ML2_proxy2_probability",  "t2_oracle": "ML2_oracle1_probability"
    }
}

class ProjectionABaeSampler:
    def __init__(self, df: pd.DataFrame, T_true: float = None, K: int = 5, config: dict = None):
        self.T_true = T_true
        self.K = K
        self.instances = self._prepare_instances(
            df, config["t1_proxy"], config["t2_proxy"], config["t1_oracle"], config["t2_oracle"]
        )

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
                else: return [val]
            return []

        def to_num_list(lst): return [float(x) for x in lst if pd.notna(x)]

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

        return df[df["a"] > 0].reset_index(drop=True)

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
                    return 0, calls, True 
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

    def run_abae_split_sample(self, target_budget_frac: float = 0.1, pilot_ratio: float = 0.2, budget_B: int = 500, random_seed: int = 42) -> Dict:
        """
        【严格无偏的 ABae 估计】
        1. 严格分离：Pilot 仅用于估计方差和成本，最终均值仅由 Stage 2 独立样本计算，消除自适应采样偏差 (Adaptive Bias)。
        2. 方差平滑：防止零方差死结，保证极其稀疏的正样本层在 Stage 2 中也有被探索的机会。
        3. 全局打乱：消除因成本差异导致的固定层截断偏差。
        """
        if self.instances.empty:
            return {"T_hat": 0.0, "ARE": 0.0, "Signed_RE": 0.0, "oracle_cost": 0}

        rng = np.random.default_rng(random_seed)
        df = self.stratify_by_proxy_quantile(self.instances, self.K)
        N_total_pop = len(df)

        N_rows_budget = max(1, int(math.floor(target_budget_frac * N_total_pop)))
        N1_pilot = int(math.floor(N_rows_budget * pilot_ratio)) 
        N2_stage2 = N_rows_budget - N1_pilot                   

        oracle_cache = {}
        total_budget_used = 0
        budget_exhausted = False

        strata_groups = dict(list(df.groupby("stratum")))
        actual_K = len(strata_groups)
        n1_per_stratum = max(1, N1_pilot // actual_K)

        pilot_stats = {}
        # 【无偏核心 1】：只准备 Stage 2 结果的容器，严格隔离 Pilot 样本
        stage2_results = {k: [] for k in strata_groups}

        # =================================================================
        # Stage 1: Pilot Sampling (仅为获取先验，不计入最终求和)
        # =================================================================
        pilot_sampled_indices = {k: [] for k in strata_groups}
        for k, grp in strata_groups.items():
            Nk = len(grp)
            n1_k = min(n1_per_stratum, Nk)
            
            sampled_grp = grp.sample(n=n1_k, random_state=rng.integers(0, 1<<30))
            pilot_sampled_indices[k] = sampled_grp.index.tolist()
            
            y_vals = []
            cost_vals = []
            for _, row in sampled_grp.iterrows():
                if budget_exhausted: break
                is_valid, calls, is_out = self._eval_oracle_strict(row, oracle_cache, total_budget_used, budget_B)
                if is_out:
                    budget_exhausted = True
                    break
                
                total_budget_used += calls
                val = row["a"] * is_valid
                y_vals.append(val)
                cost_vals.append(calls)
            
            sigma_hat_k = np.std(y_vals, ddof=1) if len(y_vals) > 1 else 0.0
            mean_cost_k = np.mean(cost_vals) if len(cost_vals) > 0 else 1.0
            mean_pilot_k = np.mean(y_vals) if len(y_vals) > 0 else 0.0
            
            # 【无偏核心 2】：Laplace平滑。防止样本方差为0导致的分配坍塌，彻底消除系统性低估。
            smoothed_sigma = max(sigma_hat_k, 1e-5) 
            
            pilot_stats[k] = {
                "Nk": Nk, 
                "smoothed_sigma": smoothed_sigma, 
                "mean_cost": max(mean_cost_k, 0.1),
                "mean_pilot": mean_pilot_k  # 仅作为预算枯竭时的极限备用
            }

        # =================================================================
        # Stage 2: Cost-Aware Neyman 分配 & 全局打乱验证
        # =================================================================
        alloc_stage2 = {}
        if not budget_exhausted:
            alloc_weights = {k: (st["Nk"] * st["smoothed_sigma"]) / math.sqrt(st["mean_cost"]) 
                             for k, st in pilot_stats.items()}
            sum_w = sum(alloc_weights.values())
            for k in pilot_stats:
                ratio = (alloc_weights[k] / sum_w) if sum_w > 0 else (1.0 / actual_K)
                alloc_stage2[k] = int(math.floor(N2_stage2 * ratio))

        stage2_pool = []
        if not budget_exhausted:
            for k, grp in strata_groups.items():
                # 从剩余样本中抽取，数学上能证明 E[mean(Stage2)] 严格等于当前层的总体均值
                remaining_grp = grp.drop(index=pilot_sampled_indices[k], errors="ignore")
                n2_k = min(alloc_stage2.get(k, 0), len(remaining_grp))
                if n2_k > 0:
                    sampled_grp = remaining_grp.sample(n=n2_k, random_state=rng.integers(0, 1<<30))
                    for _, row in sampled_grp.iterrows():
                        stage2_pool.append((k, row))
            
            # 全局打乱：确保预算截断是一个与单一样本值独立的过程 (Ignorable Stopping Time)
            random.seed(random_seed)
            random.shuffle(stage2_pool)

        # 收集 Stage 2 产生的结果
        for k, row in stage2_pool:
            if budget_exhausted: break
            is_valid, calls, is_out = self._eval_oracle_strict(row, oracle_cache, total_budget_used, budget_B)
            if is_out:
                budget_exhausted = True
                break 
            total_budget_used += calls
            # 【无偏核心 1】：仅仅记录这部分严格独立的样本
            stage2_results[k].append(row["a"] * is_valid)

        # =================================================================
        # 最终估计：【纯粹依赖独立无偏的 Stage 2 样本】
        # =================================================================
        total_hat = 0.0

        for k, grp in strata_groups.items():
            Nk = len(grp)
            y2_list = stage2_results[k]
            
            if len(y2_list) > 0:
                # 理论上最完美的无偏均值
                mean_k = np.mean(y2_list)
            else:
                # Fallback: 仅当该层被分配了样本，但还没来得及评估完预算就耗尽了，才会被迫使用 Pilot
                mean_k = pilot_stats[k]["mean_pilot"]
                
            total_hat += mean_k * Nk

        t_true_safe = self.T_true if self.T_true and self.T_true > 0 else 1e-9
        signed_re = (total_hat - t_true_safe) / t_true_safe
        are = abs(total_hat - t_true_safe) / t_true_safe

        return {
            "T_hat": float(total_hat),
            "ARE": float(are),
            "Signed_RE": float(signed_re),
            "oracle_cost": int(total_budget_used)
        }
    
def process_single_task(fname, run_id, agg_dir, gt_map, query_budgets, config, pilot_ratio):
    try:
        q_clean = fname.replace("aggregated_list_", "").replace(".csv", "")
        q_basename = q_clean + ".graph"
        T_true = gt_map.get(q_clean)

        if T_true is None: return None

        budget_B = query_budgets.get(q_basename, 500)
        filepath = os.path.join(agg_dir, fname)
        df = pd.read_csv(filepath)

        sampler = ProjectionABaeSampler(df=df, T_true=T_true, K=config["K"], config=config)
        seed = (abs(hash(q_basename)) % 99999999) + run_id

        # 传入动态接收到的 pilot_ratio
        res = sampler.run_abae_split_sample(
            target_budget_frac=config["budget_frac"],
            pilot_ratio=pilot_ratio,  
            budget_B=budget_B,
            random_seed=seed
        )

        return {
            "query_basename": q_basename, "run_id": run_id,
            "T_hat_abae": res["T_hat"], "T_true": T_true,
            "Signed_RE": res["Signed_RE"], "ARE": res["ARE"],
            "oracle_cost": res["oracle_cost"], "budget_limit_B": budget_B
        }
    except Exception as e:
        print(f"\n[Error] {fname} (Run {run_id}): {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="/home/wangshuo/projects/PROXY")
    parser.add_argument("--dataset", default="parler")
    parser.add_argument("--agg_mode", choices=["count", "sum"], default="count")
    parser.add_argument("--budget_frac", type=float, default=0.1)
    parser.add_argument("--pilot_ratio", type=float, default=0.1)  # 接收来自 Shell 的 pilot_ratio 参数
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    dataset_path = os.path.join(args.base_dir, "datasets", args.dataset)
    results_dir = os.path.join(dataset_path, "results")
    
    agg_dir = os.path.join(results_dir, f"aggregated_results_{args.agg_mode}")
    if not os.path.exists(agg_dir): 
        agg_dir = os.path.join(results_dir, "aggregated_results")
        
    ablation_csv_path = os.path.join(results_dir, "efficiency", f"allocation_strategy_comparison_ablation_{args.agg_mode}.csv")
    if not os.path.exists(ablation_csv_path): 
        ablation_csv_path = os.path.join(results_dir, "efficiency", f"allocation_strategy_comparison_{args.agg_mode}.csv")
        
    out_csv_path = os.path.join(results_dir, "efficiency", f"Projection_ABae_{args.dataset}_{args.agg_mode}.csv")

    gt_cands = [
        os.path.join(results_dir, f"T_true_ML3_oracle2_probability_ML2_oracle1_probability_{args.agg_mode}.json"),
        os.path.join(results_dir, f"T_true_ML1_oracle2_probability_ML2_oracle2_probability_{args.agg_mode}.json")
    ]
    gt_path = next((p for p in gt_cands if os.path.exists(p)), None)

    if not gt_path: 
        print(f"[Error] 未找到对应的 Ground Truth 文件，请检查路径: {results_dir}")
        return

    print("=" * 65)
    print(f"🚀 启动无偏 Projection-ABae 基线 (Pilot={args.pilot_ratio*100}% | Cost-Aware 分配 | 全局打乱防截断)")
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

    cfg = WORKLOADS_CONFIG.get(args.dataset, WORKLOADS_CONFIG["parler"])
    cfg.update({"K": args.K, "budget_frac": args.budget_frac})

    completed_cnt = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        # 传递 args.pilot_ratio 到任务中
        futures = {executor.submit(process_single_task, fname, r, agg_dir, gt_map, query_budgets, cfg, args.pilot_ratio): (fname, r) 
                   for fname in agg_files for r in range(1, args.runs + 1)}

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Evaluating"):
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
        print(f"📊 Projection-ABae 无偏最终评估")
        print("=" * 65)
        print(f"1. 带符号误差均值 (Mean Signed RE): {mean_sre:.4f} ({mean_sre*100:.2f}%)")
        print(f"2. 绝对误差均值   (Mean ARE)      : {mean_are:.4f} ({mean_are*100:.2f}%)")
        print("=" * 65)

if __name__ == "__main__":
    main()