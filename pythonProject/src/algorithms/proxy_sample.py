import ast 
import math
import numpy as np
import pandas as pd
from typing import Tuple, Dict
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import polars as pl
from scipy.stats.mstats import winsorize
import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List
from pythonProject.src.algorithms.compute_truth import GroundTruthManager
from scipy.stats import norm
from tqdm import tqdm


class ProxyStratifiedSampler:

    def __init__(self, csv_path: str,
                 c_stage: float = 0.05,
                 K: int = 5,
                 total_budget_frac: float = 0.1,
                 T_true: float = 12561, 
                 is_multi_predicate: bool = False,  
                 post_proxy: str = "ML1_proxy4b_probability",  
                 comment_proxy: str = "ML2_proxy4d2_probability",  
                 post_oracle: str = "ML1_oracle2_probability" ,
                 comment_oracle: str = "ML2_oracle2_probability"
                 ):

        self.csv_path = csv_path
        self.c_stage = c_stage
        self.K = K
        self.total_budget_frac = total_budget_frac
        self.T_true = T_true
        
        df = pd.read_csv(csv_path)

        if is_multi_predicate:
            self.posts = self.prepare_instances_from_aggregated(
                df,
                post_proxy_col=post_proxy,
                comment_proxy_col=comment_proxy,
                post_oracle_col=post_oracle,
                comment_oracle_col=comment_oracle,
            )
        else:
            self.posts = self.prepare_posts(df, proxy_model=post_proxy,oracle_model=post_oracle)
        self.stratification_cache = {}

    @staticmethod
    def prepare_instances_from_aggregated(
            df: pd.DataFrame,
            post_proxy_col: str = "ML1_proxy4b1_probability",
            comment_proxy_col: str = "ML2_proxy1_probability",
            post_oracle_col: str = "ML1_oracle1_probability",
            comment_oracle_col: str = "ML2_oracle2_probability"
    ) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        df = df.copy()
        df.rename(columns={"estimateW": "a"}, inplace=True)
        df["a"] = pd.to_numeric(df["a"], errors="coerce").fillna(0.0)

        def safe_literal_eval(val):
            if pd.isna(val) or not isinstance(val, str) or val == "nan":
                return []
            try:
                result = ast.literal_eval(val)
                return result if isinstance(result, list) else []
            except (ValueError, SyntaxError):
                return []

        def to_num_list(lst):
            out = []
            for x in lst:
                v = pd.to_numeric(x, errors="coerce")
                out.append(float(v) if pd.notna(v) else np.nan)
            return out

        if "post_id_list" in df.columns:
            df["post_ids"] = df["post_id_list"].apply(safe_literal_eval)
        else:
            df["post_ids"] = [[] for _ in range(len(df))]

        if "comment_id_list" in df.columns:
            df["comment_ids"] = df["comment_id_list"].apply(safe_literal_eval)
        else:
            df["comment_ids"] = [[] for _ in range(len(df))]

        if post_proxy_col not in df.columns:
            print(f"[警告] 代理列 '{post_proxy_col}' 不存在，将使用空列表。")
            df[post_proxy_col] = "[]"
        if comment_proxy_col not in df.columns:
            print(f"[警告] 代理列 '{comment_proxy_col}' 不存在，将使用空列表。")
            df[comment_proxy_col] = "[]"

        post_proxy_list = df[post_proxy_col].apply(safe_literal_eval).apply(to_num_list)
        comment_proxy_list = df[comment_proxy_col].apply(safe_literal_eval).apply(to_num_list)

        def nan_prod(lst):
            return float(np.nanprod(lst)) if len(lst) > 0 else 1.0

        df["proxy"] = post_proxy_list.apply(nan_prod) * comment_proxy_list.apply(nan_prod)

        if post_oracle_col not in df.columns:
            print(f"[警告] oracle 列 '{post_oracle_col}' 不存在，将按空列表处理。")
            df[post_oracle_col] = "[]"
        if comment_oracle_col not in df.columns:
            print(f"[警告] oracle 列 '{comment_oracle_col}' 不存在，将按空列表处理。")
            df[comment_oracle_col] = "[]"

        df["post_oracle_probs"] = df[post_oracle_col].apply(safe_literal_eval).apply(to_num_list)
        df["comment_oracle_probs"] = df[comment_oracle_col].apply(safe_literal_eval).apply(to_num_list)

        if "expected_post_slots" not in df.columns:
            df["expected_post_slots"] = df["post_oracle_probs"].apply(len)
        else:
            df["expected_post_slots"] = pd.to_numeric(df["expected_post_slots"], errors="coerce").fillna(0).astype(int)

        if "expected_comment_slots" not in df.columns:
            df["expected_comment_slots"] = df["comment_oracle_probs"].apply(len)
        else:
            df["expected_comment_slots"] = pd.to_numeric(df["expected_comment_slots"], errors="coerce").fillna(0).astype(int)

        # 保留 oracle 列用于兼容，运行时会被动态覆盖
        df["oracle"] = 0

        df.rename(columns={"instance_id": "id:ID"}, inplace=True)
        instances = df[df["a"] > 0].reset_index(drop=True)

        final_cols = [
            "id:ID", "a", "proxy", "oracle",
            "post_ids", "comment_ids",
            "post_oracle_probs", "comment_oracle_probs",
            "expected_post_slots", "expected_comment_slots"
        ]
        for col in final_cols:
            if col not in instances.columns:
                instances[col] = 0

        return instances[final_cols]

    def _count_unique_nodes(self, sampled_df: pd.DataFrame) -> Tuple[int, int]:
      
        if sampled_df.empty: return 0, 0

        
        if 'post_ids' not in sampled_df.columns:
            return len(sampled_df), 0

        unique_posts = set()
        unique_comments = set()

        
        for ids in sampled_df['post_ids']:
            if isinstance(ids, list): unique_posts.update(ids)

        for ids in sampled_df['comment_ids']:
            if isinstance(ids, list): unique_comments.update(ids)

        return len(unique_posts), len(unique_comments)

    def _eval_instance_oracle_short_circuit(self, row: pd.Series, oracle_cache: Dict) -> Tuple[int, int, int]:
        post_ids = row.get("post_ids", [])
        comment_ids = row.get("comment_ids", [])
        post_probs = row.get("post_oracle_probs", [])
        comment_probs = row.get("comment_oracle_probs", [])

        if not isinstance(post_ids, list):
            post_ids = []
        if not isinstance(comment_ids, list):
            comment_ids = []
        if not isinstance(post_probs, list):
            post_probs = []
        if not isinstance(comment_probs, list):
            comment_probs = []

        exp_post = pd.to_numeric(row.get("expected_post_slots", len(post_probs)), errors="coerce")
        exp_comment = pd.to_numeric(row.get("expected_comment_slots", len(comment_probs)), errors="coerce")
        exp_post = int(exp_post) if pd.notna(exp_post) else len(post_probs)
        exp_comment = int(exp_comment) if pd.notna(exp_comment) else len(comment_probs)

        calls_post, calls_comment = 0, 0

        has_post = len(post_ids) > 0 and len(post_probs) > 0
        has_comment = len(comment_ids) > 0 and len(comment_probs) > 0

        if not has_post and not has_comment:
            return 0, 0, 0

        if has_post:
            if len(post_ids) != len(post_probs):
                return 0, 0, 0
            if len(post_probs) != exp_post:
                return 0, 0, 0

            for nid, prob in zip(post_ids, post_probs):
                key = ("post", str(nid))
                if key in oracle_cache:
                    ok = oracle_cache[key]
                else:
                    pv = pd.to_numeric(prob, errors="coerce")
                    ok = bool(pd.notna(pv) and float(pv) > 0.5)
                    oracle_cache[key] = ok
                    calls_post += 1

                if not ok:
                    return 0, calls_post, calls_comment

        if has_comment:
            if len(comment_ids) != len(comment_probs):
                return 0, calls_post, calls_comment
            if len(comment_probs) != exp_comment:
                return 0, calls_post, calls_comment

            for nid, prob in zip(comment_ids, comment_probs):
                key = ("comment", str(nid))
                if key in oracle_cache:
                    ok = oracle_cache[key]
                else:
                    pv = pd.to_numeric(prob, errors="coerce")
                    ok = bool(pd.notna(pv) and float(pv) > 0.5)
                    oracle_cache[key] = ok
                    calls_comment += 1

                if not ok:
                    return 0, calls_post, calls_comment

        return 1, calls_post, calls_comment

    def _materialize_oracle_for_sample(self, sample_df: pd.DataFrame, oracle_cache: Dict) -> Tuple[pd.DataFrame, int, int]:
        if sample_df is None or sample_df.empty:
            return sample_df, 0, 0

        out = sample_df.copy()
        oracle_vals = []
        post_calls, comment_calls = 0, 0

        for _, row in out.iterrows():
            o, cp, cc = self._eval_instance_oracle_short_circuit(row, oracle_cache)
            oracle_vals.append(o)
            post_calls += cp
            comment_calls += cc

        out["oracle"] = np.array(oracle_vals, dtype=int)
        return out, post_calls, comment_calls


    @staticmethod
    def prepare_posts(df: pd.DataFrame, proxy_model: str,oracle_model: str) -> pd.DataFrame:
        df = df.copy()
        df["estimate"] = pd.to_numeric(df["estimate"], errors="coerce").fillna(0).astype(float)
        g = df.groupby("id:ID", sort=False)
        # g = df.groupby("postID", sort=False)
        posts = pd.DataFrame({
            "id:ID": list(g.groups.keys()),
            "w": g.size().values.astype(float),
            "a": g["estimate"].sum().values.astype(float),
            # "proxy": g["post_proxy4b1"].first().values.astype(float),
            "proxy": g[proxy_model].first().values.astype(float),
            # "oracle_val": g["post_oracle1"].first().values.astype(float)
            "oracle_val": g[oracle_model].first().values.astype(float)
        })
        posts["oracle"] = (posts["oracle_val"] > 0.5).astype(int)
        posts = posts[posts["a"] > 0].reset_index(drop=True)
        return posts


    @staticmethod
    def stratify_by_proxy(posts: pd.DataFrame, K: int) -> pd.DataFrame:
        try:
            posts["stratum"] = pd.qcut(posts["proxy"], K, labels=False, duplicates="drop")
        except Exception:
            posts["stratum"] = pd.cut(posts["proxy"].rank(method="first"), bins=K, labels=False)
        posts["stratum"] = posts["stratum"].fillna(0).astype(int)
        return posts

    @staticmethod
    def stratify_by_expected_contrib(posts: pd.DataFrame, K: int) -> pd.DataFrame:
        posts["exp_contrib"] = posts["proxy"] * posts["a"]
        try:
            posts["stratum"] = pd.qcut(posts["exp_contrib"], K, labels=False, duplicates="drop")
        except Exception:
            posts["stratum"] = pd.cut(posts["exp_contrib"].rank(method="first"), bins=K, labels=False)
        posts["stratum"] = posts["stratum"].fillna(0).astype(int)
        return posts

    @staticmethod
    def stratify_by_clustering_1d(posts: pd.DataFrame, K: int) -> pd.DataFrame:

        feature = np.sqrt(posts["proxy"] * posts["a"]).values.reshape(-1, 1)
        
        
        N = len(posts)
        if K <= 1 or N < K:
            posts["stratum"] = 0
            return posts

        
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
        labels = kmeans.fit_predict(feature)
        
        
        centers = kmeans.cluster_centers_.flatten()
        
        sorted_indices = np.argsort(centers)
        
       
        label_map = {old_lbl: new_lbl for new_lbl, old_lbl in enumerate(sorted_indices)}
        
    
        lookup = np.zeros(K, dtype=int)
        for old, new in label_map.items():
            lookup[old] = new
            
        posts["stratum"] = lookup[labels]
        
        return posts
    

    @staticmethod
    def allocate_pilot_budget(stats: Dict[int, dict], N1_total: int, min_per_stratum: int = 1) -> Dict[int, int]:
        if N1_total <= 0:
            return {k: 0 for k in stats}
        Nks = {k: st["N_k"] for k, st in stats.items()}
        total_N = sum(Nks.values())
        cont = {k: N1_total * Nks[k] / total_N for k in Nks}
        floored = {k: int(math.floor(v)) for k, v in cont.items()}
        assigned = sum(floored.values())
        rem = N1_total - assigned
        fracs = sorted(((k, cont[k] - floored[k]) for k in cont), key=lambda x: x[1], reverse=True)
        alloc = floored.copy()
        idx = 0
        while rem > 0 and idx < len(fracs):
            alloc[fracs[idx][0]] += 1
            rem -= 1
            idx += 1
        for k in alloc:
            alloc[k] = max(min_per_stratum, alloc[k])
        return alloc


    def pilot_stats(
            self,
            posts: pd.DataFrame,
            pilot_alloc: Dict[int, int],
            pilot_sampling_method: str = "uniform",  # "uniform" 或 "importance",
            oracle_cache: Dict = None,
            oracle_counter: Dict[str, int] = None
    ):

        stats, pilots = {}, {}
        if oracle_cache is None:
            oracle_cache = {}
        if oracle_counter is None:
            oracle_counter = {"post": 0, "comment": 0}
        for k, grp in posts.groupby("stratum"):
            Nk = len(grp)
            n1 = int(min(pilot_alloc.get(k, 0), Nk))

            if n1 <= 0:
                stats[k] = {"W_k": float(grp["a"].sum()), "p_hat": 0.0,
                            "sigma_hat": 0.0, "N_k": Nk, "n1": 0}
                pilots[k] = pd.DataFrame(columns=posts.columns)
                continue

       
            if pilot_sampling_method == "uniform":
       
                sample = grp.sample(
                    n1,
                    replace=False,
                    random_state=np.random.randint(1 << 30)
                ).copy()

            elif pilot_sampling_method == "importance":
                
                print('[check importance sampling in pilot stage]')
                eps = 1e-8
                imp_weights = np.sqrt((grp["proxy"] * grp["a"]).clip(lower=0)) + eps
                prob = imp_weights / imp_weights.sum()

               
                sample_indices = np.random.choice(
                    grp.index,
                    size=n1,
                    replace=False,
                    p=prob.values
                )
                sample = grp.loc[sample_indices].copy()

            else:
                raise ValueError(f"Unknown pilot_sampling_method: {pilot_sampling_method}")

            sample, cp, cc = self._materialize_oracle_for_sample(sample, oracle_cache)
            oracle_counter["post"] += cp
            oracle_counter["comment"] += cc

            
            sample["Y"] = sample["a"] * sample["oracle"]

            W_k_sample = sample["a"].sum()
            W_pos = sample.loc[sample["oracle"] == 1, "a"].sum()

            p_hat = (W_pos / W_k_sample) if W_k_sample > 0 else 0.0
            sigma_hat = sample["Y"].std(ddof=1) if len(sample) > 1 else 0.0

            W_k = grp["a"].sum()

            stats[k] = {
                "W_k": W_k,
                "p_hat": p_hat,
                "sigma_hat": sigma_hat,
                "N_k": Nk,
                "n1": n1
            }
            pilots[k] = sample

        return stats, pilots

    @staticmethod
    def allocate_second_stage_heuristic(posts: pd.DataFrame, N2: int, strategy: str = "root_wp") -> Dict[int, int]:
        
        alloc_weights = {}
        stratum_sizes = {}
        
        
        for k, grp in posts.groupby("stratum"):
            stratum_sizes[k] = len(grp)
            if grp.empty:
                alloc_weights[k] = 0.0
                continue
                
            a_vals = grp["a"].values
            p_vals = grp["proxy"].values
            
            if strategy == "root_wp":
                w_h = np.sum(a_vals * np.sqrt(p_vals + 1e-12))
            elif strategy == "sqrt_wp": 
                w_h = np.sum(np.sqrt(a_vals * p_vals + 1e-12))
            elif strategy == "neyman_bernoulli":
                sigma = np.sqrt(p_vals * (1 - p_vals) + 1e-6)
                w_h = np.sum(a_vals * sigma)
            elif strategy == "prop_value":
                w_h = np.sum(a_vals * p_vals)
            elif strategy == "w_root_mean_p":
                sum_w = np.sum(a_vals)
                mean_p = np.mean(p_vals)
                w_h = sum_w * np.sqrt(mean_p + 1e-12)
            else:
                w_h = 0.0
            
            alloc_weights[k] = w_h

        
        total_population = sum(stratum_sizes.values())
        if N2 >= total_population:
            return stratum_sizes.copy()

        
        final_alloc = {k: 0 for k in alloc_weights}
        remaining_budget = N2
        
        
        active_strata = set(k for k in alloc_weights if stratum_sizes[k] > 0)

        
        while remaining_budget > 0 and active_strata:
            current_total_w = sum(alloc_weights[k] for k in active_strata)
            
            if current_total_w <= 0:
               
                chunk = remaining_budget // len(active_strata)
                chunk = max(1, chunk)
                for k in list(active_strata):
                    add = min(chunk, stratum_sizes[k] - final_alloc[k])
                    final_alloc[k] += add
                    remaining_budget -= add
                    if final_alloc[k] == stratum_sizes[k]:
                        active_strata.remove(k)
                    if remaining_budget <= 0:
                        break
                continue

            
            tentative = {}
            for k in active_strata:
                tentative[k] = remaining_budget * (alloc_weights[k] / current_total_w)
                
            
            assigned_this_round = 0
            for k in list(active_strata):
                add = int(math.floor(tentative[k]))
                if add > 0:
                    actual_add = min(add, stratum_sizes[k] - final_alloc[k])
                    final_alloc[k] += actual_add
                    remaining_budget -= actual_add
                    assigned_this_round += actual_add
                    
                    if final_alloc[k] == stratum_sizes[k]:
                        active_strata.remove(k)
            
            
            if assigned_this_round == 0 and remaining_budget > 0 and active_strata:
                
                remainders = {k: tentative[k] - math.floor(tentative[k]) for k in active_strata}
                sorted_k = sorted(remainders.keys(), key=lambda x: remainders[x], reverse=True)
                
                for k in sorted_k:
                    if remaining_budget <= 0: break
                    if final_alloc[k] < stratum_sizes[k]:
                        final_alloc[k] += 1
                        remaining_budget -= 1
                        if final_alloc[k] == stratum_sizes[k]:
                            active_strata.remove(k)
                            
        return final_alloc



    @staticmethod
    def allocate_second_stage(stats: Dict[int, dict], N2: int) -> Dict[int, int]:
        weights = {k: math.sqrt(max(1e-12, st["p_hat"]) * max(1e-12, st["sigma_hat"])) for k, st in stats.items()}
        total_w = sum(weights.values()) or 1e-12
        alloc = {k: max(1, int(N2 * weights[k] / total_w)) for k in stats}
        return alloc


    def second_stage_and_estimate(
            self,
            posts: pd.DataFrame,
            pilots: Dict[int, pd.DataFrame],
            alloc: Dict[int, int],
            sampling: str = "uniform",
            oracle_cache: Dict = None,
            oracle_counter: Dict[str, int] = None
    ) -> Dict:
       
        combined = {}
        summaries = {}
        all_sampled_frames = []
        
        if oracle_cache is None:
            oracle_cache = {}
        if oracle_counter is None:
            oracle_counter = {"post": 0, "comment": 0}

        for k, grp in posts.groupby("stratum"):
           
            pilot = pilots.get(k, pd.DataFrame(columns=posts.columns))
            if not pilot.empty:
                pilot = pilot.copy() 
                pilot, cp, cc = self._materialize_oracle_for_sample(pilot, oracle_cache)
                oracle_counter["post"] += cp
                oracle_counter["comment"] += cc
                N_h = len(grp)
                n_pilot = len(pilot)
               
                pilot_pi = n_pilot / N_h if N_h > 0 else 0
                pilot["pi"] = pilot_pi
            pilot_ids = set(pilot["id:ID"].tolist()) if not pilot.empty else set()
            
            
            remaining = grp[~grp["id:ID"].isin(pilot_ids)]
            n2_budget = alloc.get(k, 0) 

            add_sample = pd.DataFrame()
            T_hat_stage2 = 0.0
            
           
            if n2_budget > 0 and not remaining.empty:
                if sampling == "uniform":
                   
                    actual_n2 = min(n2_budget, len(remaining))
                    add_sample = remaining.sample(actual_n2, replace=False)
                   
                    weight = len(remaining) / actual_n2
                    
                    
                    add_sample["pi"] = 1.0 / weight
                    if not add_sample.empty:
                        add_sample, cp, cc = self._materialize_oracle_for_sample(add_sample, oracle_cache)
                        oracle_counter["post"] += cp
                        oracle_counter["comment"] += cc
                   
                    add_sample["Y"] = add_sample["a"] * add_sample["oracle"]
                    T_hat_stage2 = add_sample["Y"].sum() * weight
                elif sampling == "importance_nrs":
                    
                    eps = 1e-10
                    w = np.sqrt(remaining["proxy"].values * remaining["a"].values + eps)
                    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    
                    n_target = min(n2_budget, len(remaining))
                    
                    if n_target > 0:
                       
                        pi_vals = np.zeros(len(remaining))
                        idx_pool = np.arange(len(remaining))
                        w_pool = w.copy()
                        rem_n = float(n_target)

                        while True:
                            sum_w = w_pool.sum()
                            if sum_w <= 0:
                                
                                if len(idx_pool) > 0:
                                    pi_vals[idx_pool] = rem_n / len(idx_pool)
                                break
                            
                           
                            temp_pi = rem_n * (w_pool / sum_w)
                            
                            certain_mask = temp_pi >= (1.0 - 1e-9)
                            
                            if not np.any(certain_mask):
                                
                                pi_vals[idx_pool] = temp_pi
                                break
                                
                            
                            certain_idx = idx_pool[certain_mask]
                            pi_vals[certain_idx] = 1.0
                            
                           
                            rem_n -= np.sum(certain_mask)
                            idx_pool = idx_pool[~certain_mask]
                            w_pool = w_pool[~certain_mask]
                            
                            if rem_n <= 1e-9 or len(idx_pool) == 0:
                                break

                        
                        rng = np.random.default_rng()
                        perm_indices = rng.permutation(len(remaining))
                        
                        
                        perm_pi = pi_vals[perm_indices]
                        
                        
                        cumsum = np.cumsum(perm_pi)
                        total_length = cumsum[-1] 
                        
                        
                        u = rng.uniform(0, 1)
                        sample_points = np.arange(u, total_length, 1.0)
                        
                        
                        selected_positions = np.searchsorted(cumsum, sample_points)
                        selected_positions = np.clip(selected_positions, 0, len(remaining) - 1)
                        
                        
                        sampled_perm_indices = np.unique(selected_positions)
                        final_sample_idx = perm_indices[sampled_perm_indices]
                        
                        
                        add_sample = remaining.iloc[final_sample_idx].copy()
                        if not add_sample.empty:
                            add_sample, cp, cc = self._materialize_oracle_for_sample(add_sample, oracle_cache)
                            oracle_counter["post"] += cp
                            oracle_counter["comment"] += cc
                            
                        
                        pi_used = pi_vals[final_sample_idx]
                        
                        
                        y_vals = add_sample["a"].values * add_sample["oracle"].values
                        estimate_terms = y_vals / (pi_used + 1e-12)
                        T_hat_stage2 = np.sum(estimate_terms)
                        
                        add_sample["pi"] = pi_used
                    
                    else:
                        T_hat_stage2 = 0.0
                
                else: 
                    w = np.sqrt(remaining["proxy"].values * remaining["a"].values + 1e-10)
                    sum_w = w.sum()
                    if sum_w == 0:
                        probs = np.ones(len(remaining)) / len(remaining)
                    else:
                        probs = w / sum_w

                    
                    rng = np.random.default_rng()
                    unique_indices = set()
                    sampled_indices = [] # 记录所有 trial
                    
                   
                    batch_size = max(50, int(n2_budget * 1.5))
                    
                    while len(unique_indices) < n2_budget:
                       
                        needed = n2_budget - len(unique_indices)
                        curr_batch = max(needed, 50)
                        
                        
                        raw_idx = rng.choice(len(remaining), size=curr_batch, replace=True, p=probs)
                        
                        for idx in raw_idx:
                           
                            if idx not in unique_indices:
                                if len(unique_indices) >= n2_budget:
                                    break # 预算满，该样本不能算入
                                unique_indices.add(idx)
                            
                           
                            sampled_indices.append(idx)
                        
                        
                        if len(sampled_indices) > n2_budget * 200 and len(unique_indices) < n2_budget:
                            break

                   
                    add_sample = remaining.iloc[sampled_indices].copy()
                    if not add_sample.empty:
                        add_sample, cp, cc = self._materialize_oracle_for_sample(add_sample, oracle_cache)
                        oracle_counter["post"] += cp
                        oracle_counter["comment"] += cc
                    sample_probs = probs[sampled_indices]
                    
                  
                    Y_vals = add_sample["a"].values * add_sample["oracle"].values
                    n_trials = len(sampled_indices)
                    
                    if n_trials > 0:
                       
                        T_hat_stage2 = np.sum(Y_vals / sample_probs) / n_trials
                    else:
                        T_hat_stage2 = 0.0
                    
                    
                    add_sample["pi"] = 1.0 

           
            Y_pilot_sum = 0.0
            if not pilot.empty:
                
                pilot["Y"] = pilot["a"] * pilot["oracle"]
                Y_pilot_sum = pilot["Y"].sum()
                
            
            T_hat_k = Y_pilot_sum + T_hat_stage2
            
            summaries[k] = {"T_hat": T_hat_k}
            
            
            final = pd.concat([pilot, add_sample], ignore_index=True, sort=False)
            if "pi" in final.columns:
                final["pi"] = final["pi"].fillna(0.0)
            
            combined[k] = final
            all_sampled_frames.append(final)

       
        T_hat = sum(v["T_hat"] for v in summaries.values())
        full_sample = pd.concat(all_sampled_frames) if all_sampled_frames else pd.DataFrame()
        
       
        return {
            "T_hat": T_hat,
            "full_sample": full_sample,
            "oracle_calls_post": int(oracle_counter["post"]),
            "oracle_calls_comment": int(oracle_counter["comment"])
        }

    def _run_generic_unstratified_nrs(self, weights: np.ndarray, budget_frac: float) -> Dict:
        """
        通用的全局无放回系统采样引擎。
        保证所有的 Ablation Baselines (UN, PO, WO) 都使用与 POSSA 同样公平的
        Tillé 截断机制和 Oracle 缓存短路机制。
        """
        if self.posts.empty:
            return {"T_hat": 0.0, "T_true": self.T_true, "Qerror": 0.0, "n_post": 0, "n_comment": 0, "oracle_cost": 0}

        N = len(self.posts)
        n_target = int(math.floor(budget_frac * N))
        n_target = min(n_target, N)

        if n_target <= 0:
            return {"T_hat": 0.0, "T_true": self.T_true, "Qerror": 1.0, "n_post": 0, "n_comment": 0, "oracle_cost": 0}

        # 1. Tillé 截断算法，确保所有样本 pi <= 1.0
        pi_vals = np.zeros(N)
        idx_pool = np.arange(N)
        w_pool = weights.copy()
        rem_n = float(n_target)

        while True:
            sum_w = w_pool.sum()
            if sum_w <= 0:
                if len(idx_pool) > 0:
                    pi_vals[idx_pool] = rem_n / len(idx_pool)
                break
            
            temp_pi = rem_n * (w_pool / sum_w)
            certain_mask = temp_pi >= (1.0 - 1e-9)
            
            if not np.any(certain_mask):
                pi_vals[idx_pool] = temp_pi
                break
                
            certain_idx = idx_pool[certain_mask]
            pi_vals[certain_idx] = 1.0
            
            rem_n -= np.sum(certain_mask)
            idx_pool = idx_pool[~certain_mask]
            w_pool = w_pool[~certain_mask]
            
            if rem_n <= 1e-9 or len(idx_pool) == 0:
                break

        # 2. 系统采样 (Systematic Sampling)
        rng = np.random.default_rng()
        perm_indices = rng.permutation(N)
        perm_pi = pi_vals[perm_indices]
        
        cumsum = np.cumsum(perm_pi)
        total_length = cumsum[-1]
        
        u = rng.uniform(0, 1)
        sample_points = np.arange(u, total_length, 1.0)
        
        selected_positions = np.searchsorted(cumsum, sample_points)
        selected_positions = np.clip(selected_positions, 0, N - 1)
        
        sampled_perm_indices = np.unique(selected_positions)
        final_sample_idx = perm_indices[sampled_perm_indices]

        # 3. 执行 Oracle 测试与 HT 估计
        oracle_cache = {}
        post_calls, comment_calls = 0, 0
        T_hat = 0.0

        for idx in final_sample_idx:
            row = self.posts.iloc[idx]
            
            # 使用统一的短路连接验证 (公平计费)
            ok, cp, cc = self._eval_instance_oracle_short_circuit(row, oracle_cache)
            post_calls += cp
            comment_calls += cc

            # Horvitz-Thompson 估计 (y_i / pi_i)
            y_val = row["a"] * ok
            T_hat += y_val / max(pi_vals[idx], 1e-12)

        Qerror = abs(T_hat - self.T_true) / (self.T_true if self.T_true != 0 else 1.0)

        return {
            "T_hat": float(T_hat),
            "T_true": float(self.T_true),
            "Qerror": float(Qerror),
            "n_post": post_calls,
            "n_comment": comment_calls,
            "oracle_cost": post_calls + comment_calls
        }


    def run(self, stratify_mode: str = "proxy", sampling: str = "uniform", alloc_strategy: str = "neyman_pilot",
            force_oracle: bool = False) -> Dict:
        """
        alloc_strategy: 
          - "neyman_pilot": 使用 Pilot 样本的方差估计 (原方法)
          - "root_wp": 使用 sum(w * sqrt(p))
          - "w_root_mean_p": 使用 sum(w) * sqrt(mean(p))
        """
        oracle_cache = {}
        oracle_counter = {"post": 0, "comment": 0}
        if self.posts.empty:
            return {"T_hat": 0.0, "T_true": self.T_true, "Qerror": 0.0, "n_post": 0, "n_comment": 0} # +++ 返回 0 计数
        posts = self.posts.copy()

        # ==========================================
        # === [逻辑分支 1] 强制全量 Oracle (针对小数据集) ===
        # ==========================================
        if force_oracle:
             # 不做分层，不做采样，直接计算总和
             # 相当于对所有行都进行了 Oracle 检查
            #  T_hat = (self.posts["a"] * self.posts["oracle"]).sum()
             oracle_cache = {}
             full_eval, calls_post, calls_comment = self._materialize_oracle_for_sample(self.posts, oracle_cache)
             T_hat = (full_eval["a"] * full_eval["oracle"]).sum()
             # 计算 Qerror (如果 T_true 已知)
             if self.T_true is not None and self.T_true != 0:
                 Qerror = abs(T_hat - self.T_true) / self.T_true
             else:
                 Qerror = 0.0
             
             # 统计所有唯一节点 (全量开销)
             n_post, n_comment = self._count_unique_nodes(self.posts)
             
             # 返回结果 (pi 设为 1.0 表示全采)
             pi_stats = {"pi_min": 1.0, "pi_max": 1.0, "pi_mean": 1.0}
            #  return {"T_hat": T_hat, "T_true": self.T_true, "Qerror": Qerror, 
                    #  "n_post": n_post, "n_comment": n_comment, **pi_stats}
             return {
                    "T_hat": T_hat,
                    "T_true": self.T_true,
                    "Qerror": Qerror,
                    "n_post": calls_post,
                    "n_comment": calls_comment,
                    "oracle_cost": calls_post + calls_comment,
                    **pi_stats
                }

        # ==========================================
        # === [逻辑分支 2] 正常分层采样 ===
        # ==========================================

        # 1. 预算划分
        N_total = int(math.floor(self.total_budget_frac * len(posts)))
        N1_total = int(math.floor(self.c_stage * N_total))
        N2 = N_total - N1_total

        # ==========================================
        # === [新增] 保底策略：动态调整 K ===
        # ==========================================
        
        # 设定每层最少需要的样本数 (建议 3 到 5)
        # 如果是无放回系统采样，2-3 也可以；如果是为了算方差，至少 5
        MIN_SAMPLES_PER_STRATUM = 10 
        
        # 计算理论上允许的最大层数
        max_allowed_k = int(N_total / MIN_SAMPLES_PER_STRATUM)
        
        # 1. 动态 K 调整
        # 如果预算很少(比如20个)，max_allowed_k=4。即使你设定K=20，这里也会强制降为4
        # 如果 max_allowed_k < 1 (比如预算只有3个)，强制 K=1
        actual_K = max(1, min(self.K, max_allowed_k))
        
        # 2. 极低预算回退 (可选)
        # 如果预算比例极低 (例如 < 0.5%)，或者总数太少，直接强制 K=1 (即退化为 FOIS)
        # 这种情况下全局采样的抗风险能力最强
        if self.total_budget_frac < 0.005 or N_total < 10:
            actual_K = 1
            
        # 打印调试信息 (可选)
        if actual_K != self.K:
            print(f"[Auto-Tune] Budget={N_total}, Reduced K from {self.K} to {actual_K}")
        # actual_K = self.K  # 先注释掉自动调 K 的逻辑，保持行为一致性
        # 2. 分层
        # ==========================================
        # === B. 分层 (带缓存加速) ===
        # ==========================================
        cache_key = (stratify_mode, actual_K)
        
        if cache_key in self.stratification_cache:
            # >>> 命中缓存：直接使用已分好层的数据 <<<
            # print(f"Cache Hit: {cache_key}") # 调试用
            posts = self.stratification_cache[cache_key].copy()
        else:
            # >>> 未命中：执行分层计算 <<<
            posts = self.posts.copy() # 从原始数据拷贝
            
            if stratify_mode == "proxy":
                posts = self.stratify_by_proxy(posts, actual_K)
            elif stratify_mode == "proxyE":
                posts = self.stratify_by_expected_contrib(posts, actual_K)
            elif stratify_mode == "cluster":
                posts = self.stratify_by_clustering_1d(posts, actual_K)
            else:
                raise ValueError(f"Unsupported stratify_mode: {stratify_mode}")
            
            # 存入缓存 (保存一份带有 stratum 列的副本)
            self.stratification_cache[cache_key] = posts.copy()
        
        # 3. Pilot 采样 (第一阶段)
        stats_init = {k: {"N_k": len(g), "W_k": g["a"].sum()} for k, g in posts.groupby("stratum")}
        pilot_alloc = self.allocate_pilot_budget(stats_init, N1_total)
        stats, pilots = self.pilot_stats(posts, pilot_alloc,oracle_cache=oracle_cache,oracle_counter=oracle_counter)

        # 4. 第二阶段分配 (核心修改点)
        if alloc_strategy == "neyman_pilot":
            # 原方法：基于 Pilot 的 stats 计算
            alloc2 = self.allocate_second_stage(stats, N2)
        else:
            # 新方法：基于全局 Proxy 和 Weight 计算
            alloc2 = self.allocate_second_stage_heuristic(posts, N2, strategy=alloc_strategy)

        oracle_cache = {}
        oracle_counter = {"post": 0, "comment": 0}

        res = self.second_stage_and_estimate(
            posts, pilots, alloc2,
            sampling=sampling,
            oracle_cache=oracle_cache,
            oracle_counter=oracle_counter
        )

        full_sample = res.get('full_sample', pd.DataFrame())
        n_post, n_comment = self._count_unique_nodes(full_sample)

        # === 计算 PI 统计信息 ===
        # 注意：如果没有样本，pi_stats 会全是 0
        if not full_sample.empty and "pi" in full_sample.columns:
            pi_stats = self._calc_pi_stats(full_sample["pi"].values)
        else:
            pi_stats = self._calc_pi_stats([])

        T_hat = res["T_hat"]
        Qerror = abs(T_hat - self.T_true) / (self.T_true if self.T_true != 0 else 1.0)
        n_post = int(res.get("oracle_calls_post", 0))
        n_comment = int(res.get("oracle_calls_comment", 0))
        oracle_cost = n_post + n_comment
        # return {"T_hat": T_hat, "T_true": self.T_true, "Qerror": Qerror, "n_post": n_post, "n_comment": n_comment,"full_sample": full_sample, **pi_stats}
        return {
            "T_hat": T_hat,
            "T_true": self.T_true,
            "Qerror": Qerror,
            "n_post": n_post,
            "n_comment": n_comment,
            "oracle_cost": n_post + n_comment,
            "full_sample": full_sample,
            **pi_stats
        }
            # ----------------------------
    # x种实验接口
    # ----------------------------
    def run_proxy_importance(self):
        return self.run("proxy", "importance")

    def run_proxy_uniform(self):
        return self.run("proxy", "uniform")

    def run_proxyE_importance(self):
        return self.run("proxyE", "importance")

    def run_proxyE_uniform(self):
        return self.run("proxyE", "uniform")
    
    # 方法 5: Alloc-Root-WP
    def run_proxyE_alloc_root_wp(self):
        # 分层: ProxyE (p*a), 层内: Importance, 分配: root_wp
        return self.run(stratify_mode="proxyE", sampling="importance", alloc_strategy="root_wp")

    # 方法 6: Alloc-W-Root-MeanP
    def run_proxyE_alloc_w_root_pbar(self):
        # 分层: ProxyE (p*a), 层内: Importance, 分配: w_root_mean_p
        return self.run(stratify_mode="proxyE", sampling="importance", alloc_strategy="w_root_mean_p")
    # 方法 7: Alloc-Sqrt-WP
    def run_proxyE_alloc_sqrt_wp(self):
        """策略1: Matched IS (推荐)"""
        return self.run(stratify_mode="proxyE", sampling="importance", alloc_strategy="sqrt_wp")
    # 方法 8: Alloc-Neyman-Bernoulli
    def run_proxyE_alloc_neyman(self):
        """策略2: Bernoulli Variance"""
        return self.run(stratify_mode="proxyE", sampling="importance", alloc_strategy="neyman_bernoulli")
    # 方法 9: Alloc-Prop-Value
    def run_proxyE_alloc_sqrt_wp_nrs(self):
        """
        新方法: 分层无放回重要性采样
        分层: ProxyE (p*a)
        分配: Sqrt_WP (Matched-IS)
        层内: Importance Without Replacement
        """
        # 注意：这里 sampling 参数传 "importance_nrs"
        return self.run(stratify_mode="proxyE", sampling="importance_nrs", alloc_strategy="sqrt_wp")
    
    # 方法 10: Cluster-Sqrt-WP-NRS
    def run_proxyE_cluster_sqrt_wp_nrs(self):
        """
        [新方法] 聚类分层 + 无放回
        分层: Cluster (K-Means on sqrt(p*a))
        分配: Sqrt_WP
        采样: 无放回重要性
        """
        return self.run(stratify_mode="cluster", sampling="importance_nrs", alloc_strategy="sqrt_wp")

    # 方法 11: POSSA 综合方法
    def run_possa(self, D_cnt: int = 100):
        """
        [综合方法] POSSA (Proxy Optimized Stratified Sampling Adaptive)
        策略切换逻辑：
        - 当 total_budget_frac < 0.15 时：使用无放回采样 (NRS)。
          原因：低预算下，无放回采样能保证更“硬”的覆盖率，避免有放回采样在小样本下因重复抽样导致的有效样本量不足。
        - 当 total_budget_frac >= 0.15 时：使用有放回采样 (WR)。
          原因：预算充足时，Hansen-Hurwitz 估计器(WR) 通常具有更好的方差收敛特性，且数学性质更简单。
        
        底层逻辑：
        - 分层: ProxyE (p*a)
        - 分配: Sqrt_WP (Matched-IS)
        """
        # 这里的 self.total_budget_frac 是在外部循环中动态赋值的
        current_N = len(self.posts)
        if current_N < D_cnt:
            # print(f"[Auto] Core size {current_N} < {D_cnt}, switching to Full Oracle.")
            # 强制开启 force_oracle
            return self.run(force_oracle=True)
        self.c_stage = 0.0
        if self.total_budget_frac < 0.15:
            # print(f"[POSSA] Budget={self.total_budget_frac:.2f} -> Mode: NRS (Without Replacement)")
            self.c_stage = 0
            # print(f'self.c_stage = {self.c_stage}')
            return self.run_proxyE_alloc_sqrt_wp_nrs()
        elif self.total_budget_frac >= 0.15 and self.total_budget_frac < 0.25:
            # print(f"[POSSA] Budget={self.total_budget_frac:.2f} -> Mode: WR (With Replacement)")
            # return self.run_proxyE_alloc_sqrt_wp()
            return self.run_proxyE_alloc_sqrt_wp_nrs()
        else:
            return self.run_proxyE_alloc_sqrt_wp_nrs()
    
    
    def run_mab_sampling(self, K: int = 5, batch_size: int = 50, ucb_scale: float = 1.0):
        """MAB: 多臂赌博机自适应分层采样"""
        if self.posts.empty:
            return {"T_hat": 0.0, "T_true": self.T_true, "Qerror": 0.0, "n_post": 0, "n_comment": 0, "oracle_cost": 0}
        
        posts = self.posts.copy()
        # MAB 依赖分层，按期望贡献分层
        posts = self.stratify_by_expected_contrib(posts, K)
        N = len(posts)
        
        # 这里的预算定义为目标行数
        budget_n = int(math.floor(self.total_budget_frac * N))
        budget_n = max(1, min(budget_n, N))
        
        # 共享的全局 Oracle 计费缓存
        oracle_cache = {}
        oracle_counter = {"post": 0, "comment": 0}
        
        arm_state = {}
        for k, grp in posts.groupby("stratum"):
            weights = np.sqrt(grp["proxy"].values * grp["a"].values + 1e-10)
            weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            total_w = weights.sum()
            probs = weights / total_w if total_w > 0 else np.ones(len(grp)) / len(grp)
            
            arm_state[k] = {
                "N_k": len(grp),
                "orig_idx": grp.index.values, # 原始在 self.posts 中的行索引
                "prob": probs,
                "n_k": 0,
                "sum_z": 0.0,
                "sum_sq_z": 0.0,
                "mean": 0.0,
                "std": 0.0
            }

        global_unique_indices = set()
        total_trials = 0
        
        # 预热：每个臂采一点
        init_samples = min(5, max(1, budget_n // (K * 2)))
        for k in arm_state:
            if arm_state[k]["N_k"] > 0:
                self._mab_sample_batch(arm_state[k], init_samples, global_unique_indices, oracle_cache, oracle_counter, budget_n)
                total_trials += init_samples

        # MAB 循环直到抽满唯一行预算
        max_trials = N * 50
        while len(global_unique_indices) < budget_n and total_trials < max_trials:
            best_arm = -1
            max_score = -1.0
            
            for k, state in arm_state.items():
                if state["N_k"] == 0: continue
                if state["n_k"] == 0:
                    score = float('inf')
                else:
                    exploration = ucb_scale * np.sqrt(2 * np.log(total_trials) / state["n_k"])
                    score = state["N_k"] * (state["std"] + exploration)
                
                if score > max_score:
                    max_score = score
                    best_arm = k
            
            if best_arm == -1: break
            
            self._mab_sample_batch(arm_state[best_arm], batch_size, global_unique_indices, oracle_cache, oracle_counter, budget_n)
            total_trials += batch_size

        # 最终估计 (Hansen-Hurwitz)
        T_hat = sum(state["mean"] for state in arm_state.values() if state["n_k"] > 0)
        Qerror = abs(T_hat - self.T_true) / (self.T_true if self.T_true != 0 else 1.0)
        
        return {
            "T_hat": float(T_hat), 
            "T_true": float(self.T_true), 
            "Qerror": float(Qerror), 
            "n_post": oracle_counter["post"], 
            "n_comment": oracle_counter["comment"],
            "oracle_cost": oracle_counter["post"] + oracle_counter["comment"]
        }

    def _mab_sample_batch(self, state, n_batch, global_unique_indices, oracle_cache, oracle_counter, max_budget):
        """MAB 的内部抽样、计费与估计函数"""
        indices_in_stratum = np.random.choice(state["N_k"], size=n_batch, replace=True, p=state["prob"])
        
        for idx_in_stratum in indices_in_stratum:
            orig_idx = state["orig_idx"][idx_in_stratum]
            
            # 控制唯一预算
            if orig_idx not in global_unique_indices:
                if len(global_unique_indices) >= max_budget:
                    break
                global_unique_indices.add(orig_idx)
                
            # 执行动态 Oracle 验证
            row = self.posts.iloc[orig_idx]
            ok, cp, cc = self._eval_instance_oracle_short_circuit(row, oracle_cache)
            oracle_counter["post"] += cp
            oracle_counter["comment"] += cc
            
            # 更新 HH 统计量
            p_i = state["prob"][idx_in_stratum]
            y_val = row["a"] * ok
            z_val = y_val / max(p_i, 1e-12)
            
            state["n_k"] += 1
            state["sum_z"] += z_val
            state["sum_sq_z"] += z_val ** 2

        # 更新均值和方差
        if state["n_k"] > 0:
            z_bar = state["sum_z"] / state["n_k"]
            state["mean"] = z_bar 
            if state["n_k"] > 1:
                var_z = (state["sum_sq_z"] - state["n_k"] * (z_bar ** 2)) / (state["n_k"] - 1)
                state["std"] = np.sqrt(max(0, var_z))
            else:
                state["std"] = state["mean"]
                

    def run_baseline_uniform(self):
        """UN: 均匀采样 (等概率无放回)"""
        budget_frac = self.total_budget_frac
        weights = np.ones(len(self.posts))
        return self._run_generic_unstratified_nrs(weights, budget_frac)

    def run_baseline_proxy(self):
        """PO: 仅 Proxy 采样 (权重正比于 proxy)"""
        budget_frac = self.total_budget_frac
        # 为了防0，加一个极小的 eps
        weights = self.posts["proxy"].values + 1e-10
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        return self._run_generic_unstratified_nrs(weights, budget_frac)

    def run_baseline_weight_only(self):
        """WO: 仅结构权重采样 (权重正比于 a)"""
        budget_frac = self.total_budget_frac
        weights = np.sqrt(self.posts["a"].values) + 1e-10
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        return self._run_generic_unstratified_nrs(weights, budget_frac)

    def run_baseline_proxy_a(self, budget_frac: float = None, eps: float = 1e-10):
        """proxy×a 采样 (p ∝ sqrt(proxy * a))"""
        # print('[Check_running_baseline_proxy_a_ws]')
        posts = self.posts.copy()
        N = len(posts)
        budget = int(budget_frac * N) if budget_frac else int(self.total_budget_frac * N)
        n = min(budget, N)

        weights = np.sqrt(posts["proxy"].values * posts["a"].values + eps)
        # weights = posts["proxy"].values * posts["a"].values + eps
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        probs = weights / (weights.sum() or 1e-12)

        rng = np.random.default_rng(np.random.randint(1 << 30))
        sample_idx = rng.choice(N, size=n, replace=False, p=probs)
        sample = posts.iloc[sample_idx]
        pi = np.minimum(1.0, n * probs[sample_idx])
        T_hat = np.sum((sample["a"].values * sample["oracle"].values) / pi)
        Qerror = abs(T_hat - self.T_true) / (self.T_true if self.T_true != 0 else 1.0)
        # 统计节点
        n_post, n_comment = self._count_unique_nodes(sample)
        pi_stats = self._calc_pi_stats(pi)
        return {"T_hat": T_hat, "T_true": self.T_true, "Qerror": Qerror, "n_post": n_post, "n_comment": n_comment,**pi_stats}


    # --- 理论最优基线 (Optimal for Sum Estimation with Proxy) ---
    def run_pa_optimal(self, budget_frac: float = None, eps: float = 1e-10):
        """
        基线方法：Optimal 采样。
        权重正比于 a * sqrt(proxy)。
        这是针对 Sum(a * O) 估算问题，在 O ~ Bernoulli(P) 假设下的方差最小化权重。
        """
        # 1. 健壮性检查
        if self.posts.empty:
            return {"T_hat": 0.0, "T_true": self.T_true, "Qerror": 0.0, "n_post": 0, "n_comment": 0, "pi_min": 0, "pi_max": 0, "pi_mean": 0}
            
        posts = self.posts.copy()
        N = len(posts)
        
        # 2. 确定样本量
        budget = int(budget_frac * N) if budget_frac else int(self.total_budget_frac * N)
        n = min(budget, N)

        # 3. 计算权重 (权重 = a * sqrt(proxy))
        # 注意：这里 a 不开根号，proxy 开根号
        weights = posts["a"].values * np.sqrt(posts["proxy"].values + eps)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 4. 计算选择概率
        total_weight = weights.sum()
        probs = weights / (total_weight if total_weight > 0 else 1e-12)

        # 5. 执行采样
        rng = np.random.default_rng(np.random.randint(1 << 30))
        sample_idx = rng.choice(N, size=n, replace=False, p=probs)
        sample = posts.iloc[sample_idx]
        
        # 6. 计算包含概率 pi
        pi = np.minimum(1.0, n * probs[sample_idx])
        
        # 7. 统计
        n_post, n_comment = self._count_unique_nodes(sample)
        pi_stats = self._calc_pi_stats(pi)

        # 8. 估计
        T_hat = np.sum((sample["a"].values * sample["oracle"].values) / pi)
        Qerror = abs(T_hat - self.T_true) / (self.T_true if self.T_true != 0 else 1.0)
        
        return {
            "T_hat": T_hat, "T_true": self.T_true, "Qerror": Qerror, 
            "n_post": n_post, "n_comment": n_comment,
            **pi_stats
        }
    # ---  仅基于 estimateW (a) 进行加权采样 ---
    def run_baseline_a(self, budget_frac: float = None, eps: float = 1e-10):
        print('[Check_running_baseline_a2]')
        """
        基线方法：a-weighted 采样。
        采样概率正比于 a (estimateW)。
        适用场景：假设图结构估计 (estimateW) 非常准确，与真实值高度线性相关。
        """
        # 1. 健壮性检查
        if self.posts.empty:
            return {"T_hat": 0.0, "T_true": self.T_true, "Qerror": 0.0, "n_post": 0, "n_comment": 0}

        posts = self.posts.copy()
        N = len(posts)

        # 2. 确定样本量
        budget = int(budget_frac * N) if budget_frac else int(self.total_budget_frac * N)
        n = min(budget, N)

        # 3. 计算权重 (权重 = a)
        # 添加 eps 防止全 0 导致除以零错误
        weights = np.log2(posts["a"].values + eps)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

        # 4. 计算选择概率
        total_weight = weights.sum()
        probs = weights / (total_weight if total_weight > 0 else 1e-12)

        # 5. 执行采样
        rng = np.random.default_rng(np.random.randint(1 << 30))
        # 注意：这里是无放回采样
        sample_idx = rng.choice(N, size=n, replace=False, p=probs)
        sample = posts.iloc[sample_idx]

        # 6. 计算包含概率 pi
        # 对于无放回加权采样，pi ≈ n * p_i (当 n << N 时近似准确，也是工业界常用做法)
        pi = np.minimum(1.0, n * probs[sample_idx])

        # 7. 统计节点开销
        n_post, n_comment = self._count_unique_nodes(sample)

        # 8. Horvitz-Thompson 无偏估计
        # formula: sum( (a * oracle) / pi )
        T_hat = np.sum((sample["a"].values * sample["oracle"].values) / pi)

        Qerror = abs(T_hat - self.T_true) / (self.T_true if self.T_true != 0 else 1.0)
        return {"T_hat": T_hat, "T_true": self.T_true, "Qerror": Qerror, "n_post": n_post, "n_comment": n_comment}



    def run_baseline_graph_only(self):
        """
        基线方法：仅图采样 (Graph Only)。
        不进行第二阶段采样，直接计算所有 estimateW > 0 的实例的 sum(estimateW * oracle)。
        """
        if self.posts.empty:
            return {
                "T_hat": 0.0,
                "T_true": self.T_true,
                "Qerror": 1.0 if self.T_true != 0 else 0.0,
            }

        oracle_cache = {}
        full_eval, n_post, n_comment = self._materialize_oracle_for_sample(self.posts, oracle_cache)

        T_hat = (full_eval["a"] * full_eval["oracle"]).sum()
        Qerror = abs(T_hat - self.T_true) / (self.T_true if self.T_true != 0 else 1.0)

        return {
            "T_hat": T_hat,
            "T_true": self.T_true,
            "Qerror": Qerror,
            "n_post": int(n_post),
            "n_comment": int(n_comment),
            "oracle_cost": int(n_post + n_comment),
            "pi_min": 1.0,
            "pi_max": 1.0,
            "pi_mean": 1.0,
        }

    

    def _calc_pi_stats(self, pi_values) -> Dict[str, float]:
        """辅助函数：计算采样概率 pi 的统计信息"""
        if len(pi_values) == 0:
            return {"pi_min": 0.0, "pi_max": 0.0, "pi_mean": 0.0}

        # 确保是 numpy array 以便计算
        pis = np.array(pi_values)
        return {
            "pi_min": float(np.min(pis)),
            "pi_max": float(np.max(pis)),
            "pi_mean": float(np.mean(pis))
        }
    
   
    def calculate_confidence_interval(self, result_dict: Dict, method: str = "eb", alpha: float = 0.05) -> Dict:
        """
        计算置信区间。
        
        :param result_dict: run() 返回的结果字典
        :param method: "eb" (Empirical Bernstein) 或 "normal" (Normal Approximation)
        :param alpha: 显著性水平 (默认 0.05 对应 95% 置信度)
        :return: 包含 lower_bound, upper_bound, epsilon, confidence 的字典
        """
        full_sample = result_dict.get('full_sample')
        T_hat = result_dict.get('T_hat', 0.0)
        print(f"[CI Calculation] Method: {method}, T_hat: {T_hat:.4f}, Sample Size: {len(full_sample) if full_sample is not None else 0}")
        # 基础检查
        if full_sample is None or full_sample.empty:
            return {"lower_bound": T_hat, "upper_bound": T_hat, "epsilon": 0.0, "confidence": 1-alpha}

        total_variance_est = 0.0
        max_range_R = 0.0
        total_n = 0
        
        # --- 1. 计算总方差 (共通步骤) ---
        for k, grp in full_sample.groupby("stratum"):
            n_h = len(grp)
            if n_h <= 1: continue
            
            # 确保 pi 存在 (如果没有则补全，假设 Pilot 是层内均匀)
            if 'pi' not in grp.columns or grp['pi'].isnull().any():
                # 简单补救：如果缺失，尝试用 n/N 补全 (如果有N信息)，或者忽略
                # 这里假设 Stage2 都有 pi，Pilot 需要在上游修复。
                # 如果真没有，只能跳过该层方差
                continue

            y_vals = grp["a"] * grp["oracle"]
            pi_vals = grp["pi"]
            
            # 构造 Z 变量 (Hansen-Hurwitz estimator for single draw: z = y/p * n)
            # 注意: 这里的 n_h 是该层实际样本数
            safe_pi = pi_vals.replace(0, 1e-12).fillna(1e-12)
            z_vals = n_h * (y_vals / safe_pi)
            
            # --- 核心：方差估计 ---
            # 样本方差 S^2_z
            var_z = np.var(z_vals, ddof=1)
            
            # 估计量 T_hat_h 的方差 = S^2_z / n_h
            # (这是保守估计，视为有放回)
            if pd.isna(var_z): var_z = 0.0
            var_estimator_h = var_z / n_h
            
            total_variance_est += var_estimator_h
            
            # 仅 EB 需要: 更新 Range
            if method == "eb":
                current_range = z_vals.max() - z_vals.min()
                if current_range > max_range_R:
                    max_range_R = current_range
            
            total_n += n_h

        if total_n <= 1:
             return {"lower_bound": T_hat, "upper_bound": T_hat, "epsilon": 0.0, "confidence": 1-alpha}

        # --- 2. 根据方法计算 Epsilon (半宽) ---
        epsilon = 0.0
        
        if method == "normal":
            # 正态分布区间: z * std_err
            # z_score for 2-sided interval
            z_score = norm.ppf(1 - alpha / 2)
            std_err = np.sqrt(total_variance_est)
            epsilon = z_score * std_err
            
        elif method == "eb":
            # 经验伯恩斯坦
            delta = alpha
            log_term = np.log(2.0 / delta)
            
            term1 = np.sqrt(2 * total_variance_est * log_term)
            term2 = (7 * max_range_R * log_term) / (3 * (total_n - 1))
            epsilon = term1 + term2
            
        else:
            raise ValueError(f"Unknown CI method: {method}")

        return {
            "lower_bound": T_hat - epsilon,
            "upper_bound": T_hat + epsilon,
            "epsilon": epsilon,
            "confidence": 1.0 - alpha,
            "method": method
        }
    
    def calculate_confidence_interval_t(self, result_dict: Dict, method: str = "normal", alpha: float = 0.05) -> Dict:
        full_sample = result_dict.get('full_sample')
        T_hat = result_dict.get('T_hat', 0.0)
        print(f"[ws CI Calculation] Method: {method}, T_hat: {T_hat:.4f}, Sample Size: {len(full_sample) if full_sample is not None else 0}")
        # 基础检查
        if full_sample is None or full_sample.empty:
            return {"lower_bound": T_hat, "upper_bound": T_hat, "epsilon": 0.0, "confidence": 1-alpha}
            
       
        if 'pi' not in full_sample.columns:
            
            return {"lower_bound": T_hat, "upper_bound": T_hat, "epsilon": 0.0, "confidence": 1-alpha}
        
        
        df = full_sample.copy()
        df['pi'] = pd.to_numeric(df['pi'], errors='coerce').fillna(1e-12)
        df.loc[df['pi'] <= 0, 'pi'] = 1e-12
        
        
        
        total_variance_est = 0.0
        max_range_R = 0.0
        total_n = 0
        
        for k, grp in df.groupby("stratum"):
            n_h = len(grp)
            if n_h <= 1: continue
            
            y_vals = grp["a"] * grp["oracle"]
            pi_vals = grp["pi"]
            
            z_vals = y_vals / pi_vals
            
          
            if n_h > 20:
                z_vals = winsorize(z_vals, limits=[0.0, 0.000001])
            
            # 计算样本方差 S^2
            var_z = np.var(z_vals, ddof=1)
            
           
            
            var_estimator_h = var_z / n_h
            total_variance_est += var_estimator_h
            
            # Range for EB
            if method == "eb":
                current_range = np.max(z_vals) - np.min(z_vals)
                max_range_R = max(max_range_R, current_range)
                
            total_n += n_h

        if total_n <= 1:
            return {"lower_bound": T_hat, "upper_bound": T_hat, "epsilon": 0.0, "confidence": 1-alpha}

       
        if method == "normal":
            
            z_score = norm.ppf(1 - alpha / 2)
            std_err = np.sqrt(total_variance_est)
            epsilon = z_score * std_err
            
        elif method == "eb":
            
            delta = alpha
            log_term = np.log(2.0 / delta)
          
            term1 = np.sqrt(2 * total_variance_est * log_term)
            
            term2 = (7 * max_range_R * log_term) / (3 * (total_n - 1))
            epsilon = term1 + term2
            
        else:
            return {"lower_bound": T_hat, "upper_bound": T_hat, "epsilon": 0.0, "confidence": 1-alpha}

        
        lower = max(0.0, T_hat - epsilon)
        upper = T_hat + epsilon
        
        return {
            "lower_bound": lower,
            "upper_bound": upper,
            "epsilon": epsilon,
            "confidence": 1 - alpha
        }

    def run_baseline_proxy_a_checkpoints(self, budget_fracs, eps: float = 1e-10, seed: int = None):
        if self.posts.empty:
            return []

        posts = self.posts.copy()
        N = len(posts)

        # 1. 预算序列处理
        budget_fracs = sorted(list(set(budget_fracs)))
        
        # 2. 计算基础权重 (p ∝ sqrt(proxy * a))
        weights = np.sqrt(posts["proxy"].values * posts["a"].values + eps)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        sum_weights = weights.sum()
        
        # 归一化概率 p_i
        if sum_weights > 0:
            probs = weights / sum_weights
        else:
            probs = np.ones(N) / N

        # 初始化随机数生成器
        rng = np.random.default_rng(seed if seed is not None else np.random.randint(1 << 30))

        results = []
        
        # 3. 针对每个采样率独立进行系统采样
        for frac in budget_fracs:
            # 目标样本量 n
            n = max(1, int(frac * N))
            n = min(n, N)

            # === 系统采样核心逻辑 ===
            
            # A. 计算名义包含概率 (Nominal Pi)
            # 这是每个样本在采样数轴上占据的长度
            nominal_pi = n * probs
            
            # B. 确定用于估计的实际概率 (Actual Pi)
            # 无放回采样中概率上限为 1.0。
            # 如果 nominal_pi > 1，说明该样本占据长度 > 1，必然被击中。
            pi_used = np.minimum(1.0, nominal_pi)
            
            # C. 随机打乱 (消除原始顺序偏差)
            perm_indices = rng.permutation(N)
            perm_nominal_pi = nominal_pi[perm_indices]
            
            # D. 构建累积概率数轴
            cumsum = np.cumsum(perm_nominal_pi)
            total_length = cumsum[-1] # 理论上接近 n
            
            # E. 生成等距采样点
            # 随机起点 u ~ [0, 1)
            u = rng.uniform(0, 1)
            # 采样点: u, u+1, u+2 ...
            sample_points = np.arange(u, total_length, 1.0)
            
            # F. 确定被击中的索引
            # searchsorted 找出采样点落在哪一段
            selected_positions = np.searchsorted(cumsum, sample_points)
            
            # 边界保护
            selected_positions = np.clip(selected_positions, 0, N - 1)
            
            # G. 映射回原始索引并去重
            # unique 去重实现了“无放回”逻辑 (处理 nominal_pi > 1 的情况)
            # 同时也处理了系统采样天然的去重
            sampled_perm_indices = np.unique(selected_positions)
            sample_idx = perm_indices[sampled_perm_indices]
            
            # === 估计量计算 (Horvitz-Thompson) ===
            
            # 提取被选中样本的数据
            sample_df = posts.iloc[sample_idx]
            
            # 获取对应的 pi (分母)
            current_pi = pi_used[sample_idx]
            
            # HT Estimator: sum( y_i / pi_i )
            y_vals = sample_df["a"].values * sample_df["oracle"].values
            
            # 避免除以极小值 (虽然理论上被选中的 pi 肯定 > 0)
            estimate_terms = y_vals / (current_pi + 1e-12)
            T_hat = np.sum(estimate_terms)

            Qerror = abs(T_hat - self.T_true) / (self.T_true if self.T_true != 0 else 1.0)

            # === Oracle Cost 统计 ===
            n_post, n_comment = self._count_unique_nodes(sample_df)
            oracle_cost = n_post + n_comment

            results.append({
                "budget_frac": frac,
                "budget_n": n,
                "T_hat": float(T_hat),
                "Qerror": float(Qerror),
                "n_post": int(n_post),
                "n_comment": int(n_comment),
                "oracle_cost": int(oracle_cost),
            })
            
        return results

    def run_baseline_proxy_a_unbiased_checkpoints(self, budget_fracs, eps: float = 1e-10, seed: int = None):
        """
        FOIS_rs (有放回) 的检查点版本：
        - 采样对象：核心集 (Rows)
        - 预算控制：budget_frac 对应唯一核心集数量 (Unique Core Sets)。
        - 逻辑：一直有放回采样，直到凑齐指定数量的唯一核心集。
        - 估计：使用总采样次数 (Trials) 进行 Hansen-Hurwitz 估计。
        """
        # print('[Check_running_baseline_proxy_a_unbiased_checkpoints]')
        if self.posts.empty:
            return []

        posts = self.posts.copy()
        N = len(posts)

        # 1. 确定最大目标 (Unique Core Sets)
        budget_fracs = sorted(list(set(budget_fracs)))
        max_frac = max(budget_fracs)
        target_max_unique = min(int(max_frac * N), N)
        
        if target_max_unique <= 0:
            return []

        # 2. 计算权重
        weights = np.sqrt(posts["proxy"].values * posts["a"].values + eps)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        probs = weights / (weights.sum() or 1e-12)

        rng = np.random.default_rng(seed if seed is not None else np.random.randint(1 << 30))

        # 3. 动态采样循环
        seen_indices = set()
        budget_cutoffs = {} # 记录达成每个 frac 时所需的 trials
        target_fracs_queue = sorted(budget_fracs)
        
        all_sampled_indices = [] # 记录每一次 Trial 的索引 (含重复)
        
        batch_size = 50000 
        total_trials = 0
        max_trials_limit = N * 500 # 防止死循环的熔断机制

        while target_fracs_queue and total_trials < max_trials_limit:
            # 批量生成
            batch_indices = rng.choice(N, size=batch_size, replace=True, p=probs)
            
            for idx in batch_indices:
                all_sampled_indices.append(idx)
                total_trials += 1
                
                # 检查是否是新的核心集
                if idx not in seen_indices:
                    seen_indices.add(idx)
                    
                    # 检查是否满足目标
                    while target_fracs_queue:
                        target_f = target_fracs_queue[0]
                        target_count = int(target_f * N)
                        
                        if len(seen_indices) >= target_count:
                            # 记录截止点：为了凑齐 target_count 个唯一核心集，我们一共抽了 total_trials 次
                            budget_cutoffs[target_f] = total_trials
                            target_fracs_queue.pop(0)
                        else:
                            break
                
                if not target_fracs_queue:
                    break
            
            # 动态调整步长
            if len(target_fracs_queue) > 0 and batch_size < 1000000:
                batch_size = min(batch_size * 2, 1000000)

        # 4. 准备计算数据
        final_indices = np.array(all_sampled_indices[:total_trials])
        a_vals_all = posts["a"].values[final_indices]
        oracle_vals_all = posts["oracle"].values[final_indices]
        p_vals_all = probs[final_indices]

        results = []
        for frac in budget_fracs:
            if frac not in budget_cutoffs:
                continue

            # n_trials: 为了达到 frac 比例的唯一核心集，实际进行的采样总次数
            n_trials = budget_cutoffs[frac]
            
            # --- 估计量计算 (Hansen-Hurwitz) ---
            # 公式: (1/n) * sum( y_i / p_i )
            # 必须使用前 n_trials 次所有的采样结果 (包含重复)
            y_subset = a_vals_all[:n_trials] * oracle_vals_all[:n_trials]
            p_subset = p_vals_all[:n_trials]
            
            T_hat = np.mean(y_subset / (p_subset + 1e-12))
            Qerror = abs(T_hat - self.T_true) / (self.T_true if self.T_true != 0 else 1.0)

            # --- Oracle Cost 统计 ---
            # 统计前 n_trials 次采样中，涉及到的唯一核心集的节点消耗
            # 先找到这 n_trials 次中包含的唯一行索引
            unique_rows_in_prefix = np.unique(final_indices[:n_trials])
            
            # 取出这些唯一行进行节点统计
            subset_df = posts.iloc[unique_rows_in_prefix]
            n_post, n_comment = self._count_unique_nodes(subset_df)
            oracle_cost = n_post + n_comment

            results.append({
                "budget_frac": frac,
                "budget_n": int(frac * N), # 这是目标的唯一核心集数
                "trials": n_trials,        # 实际采样次数
                "T_hat": float(T_hat),
                "Qerror": float(Qerror),
                "n_post": int(n_post),
                "n_comment": int(n_comment),
                "oracle_cost": int(oracle_cost),
            })
            
        return results




def evaluate_graph_only_baseline(dataset_name: str):
    """
    单独运行 'baseline_graph_only' 方法，并将结果追加到 results_summary.csv 中。
    (带进度条和实时结果输出)
    """
    print(f"\n{'=' * 10} 开始对数据集 '{dataset_name}' 运行 [Graph Only] 基线评估 {'=' * 10}")


    # --- 路径配置 ---
    base_path = f"/home/wangshuo/resource/datasets/parler_data/{dataset_name}"
    aggregated_dir = os.path.join(base_path, "results", "aggregated_results")
    core_config_path = os.path.join(base_path, "data_graph", "core_nodes_config.json")
    final_report_path = os.path.join(base_path, "results", "results_summary2.csv")

    # --- 代理配置 ---
    POST_PROXY_COL = "ML1_proxy4b1_probability"
    COMMENT_PROXY_COL = "ML2_proxy1_probability"

    # --- 1. 获取 T_true ---
    print(">>> 步骤 1: 获取 T_true...")
    gt_manager = GroundTruthManager(dataset_name=dataset_name)
    all_T_true_results = gt_manager.get_all()
    if not all_T_true_results:
        print("[错误] T_true 获取失败。")
        return

    # --- 2. 读取现有的 Summary CSV ---
    if os.path.exists(final_report_path):
        print(f">>> 步骤 2: 读取现有报告 {final_report_path} ...")
        existing_df = pd.read_csv(final_report_path)
        existing_df = existing_df[existing_df['method'] != 'baseline_graph_only']
    else:
        print(f"[警告] 现有报告 {final_report_path} 不存在，将创建新文件。")
        existing_df = pd.DataFrame()

    # --- 3. 加载核心节点配置 ---
    try:
        with open(core_config_path, 'r') as f:
            core_nodes_config = json.load(f)
    except FileNotFoundError:
        print("[错误] 核心节点配置未找到。")
        return

    # --- 4. 遍历并计算 (添加进度条) ---
    print(">>> 步骤 3: 计算 Graph Only 基线...")
    if not os.path.exists(aggregated_dir):
        print("[错误] 聚合结果目录不存在。")
        return

    agg_files = [f for f in os.listdir(aggregated_dir) if f.endswith('.csv')]
    new_records = []

    sorted_query_basenames = sorted(list(all_T_true_results.keys()))
    sorted_files = sorted(agg_files)

    # 使用 tqdm 包装循环，显示进度条
    # ncols=100 设置进度条宽度，unit="query" 设置单位
    progress_bar = tqdm(sorted_files, desc="Evaluating", unit="query", ncols=120)
    node_stats_records = []
    for agg_file in progress_bar:
        # 解析 query_basename
        base = agg_file.replace(".csv", "")
        if base.startswith("aggregated_list_"):
            base = base.replace("aggregated_list_", "")
        elif base.startswith("aggregated_wide_"):
            base = base.replace("aggregated_wide_", "")
        query_basename = base + ".graph"

        # 获取 T_true
        T_true = all_T_true_results.get(query_basename)
        if T_true is None:
            # 使用 tqdm.write 代替 print，防止打断进度条显示
            # tqdm.write(f"[跳过] T_true缺失: {query_basename}")
            continue

        filepath = os.path.join(aggregated_dir, agg_file)

        # 实例化 Sampler
        sampler = ProxyStratifiedSampler(
            csv_path=filepath,
            is_multi_predicate=True,
            post_proxy=POST_PROXY_COL,
            comment_proxy=COMMENT_PROXY_COL,
            T_true=T_true
        )

        # --- 运行 Graph Only 基线 ---
        res = sampler.run_baseline_graph_only()
        T_hat = res['T_hat']
        mape = res['Qerror']
        n_post = res.get('n_post', 0)
        n_comment = res.get('n_comment', 0)

        # MPE
        if T_true != 0:
            mpe = (T_hat - T_true) / T_true
        else:
            mpe = T_hat

        node_stats_records.append({
            "query_name": query_basename,
            "method": "baseline_graph_only",
            "post_sampled_cnt": n_post,
            "comment_sampled_cnt": n_comment
        })
        # --- 实时打印结果 ---
        # 使用 tqdm.write 可以在进度条上方打印日志，保持整洁
        # 格式说明: <35 左对齐占35位, >10.1f 右对齐保留1位小数
        tqdm.write(
            f"Query: {query_basename:<30} | T_hat: {T_hat:>9.0f} | Qerr: {mape:.4f} | "
            f"Nodes: P={n_post}, C={n_comment}"
        )
        # 准备记录数据
        try:
            query_index = sorted_query_basenames.index(query_basename)
        except ValueError:
            query_index = -1

        core_nodes = core_nodes_config.get(query_basename, {})
        gt_match_col_str = ";".join([f"u{vid}" for label in core_nodes for vid in core_nodes[label]])

        record = {
            "query_index": query_index,
            "query_basename": query_basename,
            "gt_match_col": gt_match_col_str,
            "T_true": T_true,
            "method": "baseline_graph_only",
            "T_hat_mean": T_hat,
            "T_hat_std": 0.0,
            "Qerror_mean": mape,
            "Qerror_std": 0.0,
            "MPE": mpe
        }
        new_records.append(record)

    # --- 5. 合并并保存 ---
    if not new_records:
        print("[警告] 没有生成任何新记录。")
        return

    new_df = pd.DataFrame.from_records(new_records)
    final_df = pd.concat([existing_df, new_df], ignore_index=True)

    if 'query_index' in final_df.columns and 'method' in final_df.columns:
        final_df.sort_values(by=['query_index', 'method'], inplace=True)

    final_df.to_csv(final_report_path, index=False)
    print(f"\n  已将 [baseline_graph_only] 结果追加到: {final_report_path}")
    print(f"    新增记录数: {len(new_df)}")
    print(f"    总记录数: {len(final_df)}")
    if node_stats_records:
        save_node_counts(node_stats_records)
        print(f"  节点采样统计已追加到: {SAMPLED_COUNT_FILE}")

# 结果保存路径
SAMPLED_COUNT_FILE = "/home/wangshuo/resource/datasets/parler_data/dataset_test/results/efficiency/sampled_node_count.csv"
def save_node_counts(records: List[Dict]):
    """辅助函数：将节点计数追加到 CSV"""
    if not records: return
    df = pd.DataFrame(records)
    # 检查文件是否存在以决定是否写表头
    header = not os.path.exists(SAMPLED_COUNT_FILE)
    try:
        df.to_csv(SAMPLED_COUNT_FILE, mode='a', index=False, header=header)
    except Exception as e:
        print(f"[错误] 写入节点统计失败: {e}")


