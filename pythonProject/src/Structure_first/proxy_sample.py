#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ast # 导入 ast 模块来安全地解析列表字符串
import math
import numpy as np
import pandas as pd
from typing import Tuple, Dict
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import polars as pl
import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List
from pythonProject.src.Structure_first.compute_truth import GroundTruthManager

# ==========================================================
# === 部分 1: 代理分数和结构权重主导的重要性采样和分层采样 ===
# ==========================================================
# ==========================================================
# === xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx ===
# ==========================================================

class ProxyStratifiedSampler:

    def __init__(self, csv_path: str,
                 c_stage: float = 0.2,
                 K: int = 5,
                 total_budget_frac: float = 0.1,
                 T_true: float = 12561,  # 对于多谓词，这个值可能需要重新计算或设为None
                 is_multi_predicate: bool = False,  
                 post_proxy: str = "ML1_proxy4b_probability",  
                 comment_proxy: str = "ML2_proxy4d2_probability",  
                 post_oracle: str = "ML1_oracle2_probability" ,
                 comment_oracle: str = "ML2_oracle2_probability"
                 ):
        # print(f'[Check_total_budget_frac] {total_budget_frac}')
        # print(f'[Check_c_stage] {c_stage}')
        # print(f'[Check_K] {K}')
        # print(f'[Check_is_multi_predicate] {is_multi_predicate}')
        # print(f'[Check_post_proxy] {post_proxy}')
        # print(f'[Check_comment_proxy] {comment_proxy}')
        # print(f'[Check_post_oracle] {post_oracle}')
        # print(f'[Check_comment_oracle] {comment_oracle}')
        """
        is_multi_predicate: 如果为 True，则使用新的多谓词处理逻辑。
        """
        self.csv_path = csv_path
        self.c_stage = c_stage
        self.K = K
        self.total_budget_frac = total_budget_frac
        self.T_true = T_true
        

        # --- 修改这部分逻辑 ---
        df = pd.read_csv(csv_path)

        if is_multi_predicate:
            # 使用新的多谓词预处理函数
            self.posts = self.prepare_instances_from_aggregated(
                df,
                post_proxy_col=post_proxy,
                comment_proxy_col=comment_proxy,
                post_oracle_col=post_oracle,
                comment_oracle_col=comment_oracle,
            )
        else:
            # 使用旧的单谓词预处理函数
            # (为了兼容，我们需要从__init__的参数中获取proxy_model)
            self.posts = self.prepare_posts(df, proxy_model=post_proxy,oracle_model=post_oracle)
        # ==========================================
        # === [新增] 分层结果缓存 ===
        # 结构: { (stratify_mode, K): dataframe_with_stratum_column }
        # ==========================================
        self.stratification_cache = {}

    @staticmethod
    def prepare_instances_from_aggregated(
            df: pd.DataFrame,
            post_proxy_col: str = "ML1_proxy4b1_probability",
            comment_proxy_col: str = "ML2_proxy1_probability",
            post_oracle_col: str = "ML1_oracle1_probability",
            comment_oracle_col: str = "ML2_oracle2_probability"
    ) -> pd.DataFrame:
        """
        为多谓词（核心实例）场景准备数据。
        每一行是一个实例，其中ML列是列表字符串。
        本函数将计算综合的 proxy 和 正确的 oracle。
        """
        if df.empty: return pd.DataFrame()
        df = df.copy()
        # --- 1. 'a' 现在就是 'estimateW' ---
        df.rename(columns={'estimateW': 'a'}, inplace=True)
        df['a'] = pd.to_numeric(df['a'], errors='coerce').fillna(0)

        # --- 安全地解析列表字符串的辅助函数 ---
        def safe_literal_eval(val):
            if pd.isna(val) or not isinstance(val, str) or val == 'nan':
                return []
            try:
                result = ast.literal_eval(val)
                return result if isinstance(result, list) else []
            except (ValueError, SyntaxError):
                return []
        # 如果 CSV 中包含 id_list 列，将其解析为 Python 列表
        if 'post_id_list' in df.columns:
            df['post_ids'] = df['post_id_list'].apply(safe_literal_eval)
        else:
            # 如果没有这一列（旧数据），填充空列表以防报错
            df['post_ids'] = [[] for _ in range(len(df))]

        if 'comment_id_list' in df.columns:
            df['comment_ids'] = df['comment_id_list'].apply(safe_literal_eval)
        else:
            df['comment_ids'] = [[] for _ in range(len(df))]

        # --- 2. 计算综合 proxy (概率乘积) ---

        # --- 检查代理列是否存在 ---
        if post_proxy_col not in df.columns:
            print(f"[警告] 代理列 '{post_proxy_col}' 不存在，将使用 1.0 作为默认值。")
            df[post_proxy_col] = '[]'  # 创建一个空列表字符串列
        if comment_proxy_col not in df.columns:
            print(f"[警告] 代理列 '{comment_proxy_col}' 不存在，将使用 1.0 作为默认值。")
            df[comment_proxy_col] = '[]'

        post_proxy_list = df[post_proxy_col].apply(safe_literal_eval)
        comment_proxy_list = df[comment_proxy_col].apply(safe_literal_eval)

        

        # --- 【关键修复】在这里修改 lambda 函数 ---
        def calculate_prod(lst):
            # 将列表中的元素转换为数值，无法转换的视为 NaN
            numeric_list = [pd.to_numeric(p, errors='coerce') for p in lst]
            # 使用 np.nanprod 会自动忽略 NaN 值，空列表的乘积为 1.0
            return np.nanprod(numeric_list)

        post_proxy_prod = post_proxy_list.apply(calculate_prod)
        comment_proxy_prod = comment_proxy_list.apply(calculate_prod)

        df['proxy'] = post_proxy_prod * comment_proxy_prod

        # --- 3. 计算正确的综合 oracle ---

        if post_oracle_col not in df.columns:
            print(f"[警告] oracle 列 '{post_oracle_col}' 不存在，将按空列表处理。")
            df[post_oracle_col] = '[]'
        if comment_oracle_col not in df.columns:
            print(f"[警告] oracle 列 '{comment_oracle_col}' 不存在，将按空列表处理。")
            df[comment_oracle_col] = '[]'

        # def calculate_instance_oracle(row):
        #     # 检查 Post 的 oracle 条件
        #     post_oracle_list_str = row.get('ML1_oracle1_probability')
        #     post_oracle_list = safe_literal_eval(post_oracle_list_str)
        #     # 如果 post_oracle_list 为空，all([]) 返回 True，这是我们期望的行为
        #     post_oracle_ok = all(pd.to_numeric(p, errors='coerce') > 0.5 for p in post_oracle_list)

        #     # 检查 Comment 的 oracle 条件
        #     comment_oracle_list_str = row.get('ML2_oracle2_probability')
        #     comment_oracle_list = safe_literal_eval(comment_oracle_list_str)
        #     # 如果 comment_oracle_list 为空，all([]) 返回 True
        #     comment_oracle_ok = all(pd.to_numeric(c, errors='coerce') > 0.5 for c in comment_oracle_list)

        #     return 1 if post_oracle_ok and comment_oracle_ok else 0
        def calculate_instance_oracle(row):
            post_oracle_list = safe_literal_eval(row.get(post_oracle_col))
            comment_oracle_list = safe_literal_eval(row.get(comment_oracle_col))
            post_ok = all(pd.to_numeric(p, errors="coerce") > 0.5 for p in post_oracle_list)
            comment_ok = all(pd.to_numeric(c, errors="coerce") > 0.5 for c in comment_oracle_list)
            return 1 if post_ok and comment_ok else 0
        df['oracle'] = df.apply(calculate_instance_oracle, axis=1)

        # --- 4. 'id:ID' 现在就是 'instance_id' ---
        df.rename(columns={'instance_id': 'id:ID'}, inplace=True)

        # --- 5. 筛选和返回 ---
        instances = df[df["a"] > 0].reset_index(drop=True)

        final_cols = ['id:ID', 'a', 'proxy', 'oracle', 'post_ids', 'comment_ids']
        for col in final_cols:
            if col not in instances.columns:
                instances[col] = 0

        return instances[final_cols]

    # --- 统计唯一节点数量的辅助方法 +++ ---
    def _count_unique_nodes(self, sampled_df: pd.DataFrame) -> Tuple[int, int]:
        """
        统计采样结果中唯一的 Post ID 和 Comment ID 数量。
        """
        if sampled_df.empty: return 0, 0

        # 只有多谓词模式（有 post_ids 列）才进行 ID 统计
        if 'post_ids' not in sampled_df.columns:
            # 单谓词模式简单处理：假设每一行是一个独立节点（或根据 id:ID 去重）
            return len(sampled_df), 0

        unique_posts = set()
        unique_comments = set()

        # 遍历采样到的每一个实例，将其中包含的节点 ID 加入集合去重
        for ids in sampled_df['post_ids']:
            if isinstance(ids, list): unique_posts.update(ids)

        for ids in sampled_df['comment_ids']:
            if isinstance(ids, list): unique_comments.update(ids)

        return len(unique_posts), len(unique_comments)

    # ----------------------------
    # 数据预处理
    # ----------------------------
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

    # ----------------------------
    # 分层方法
    # ----------------------------
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
        """
        基于 K-Means 的 1D 聚类分层。
        特征：sqrt(proxy * a) —— 这是针对方差最小化的最佳变换。
        """
        # 1. 构造特征 (使用 sqrt(p*a) 以近似最佳分层边界)
        # 加上 1e-12 防止 log(0) 或其他数值问题，虽然 sqrt 不怕 0
        feature = np.sqrt(posts["proxy"] * posts["a"]).values.reshape(-1, 1)
        
        # 2. 处理 K=1 或 样本过少的情况
        N = len(posts)
        if K <= 1 or N < K:
            posts["stratum"] = 0
            return posts

        # 3. 执行 K-Means
        # n_init='auto' 在 sklearn 新版中是默认值，为了兼容性可显式设置
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
        labels = kmeans.fit_predict(feature)
        
        # 4. 【关键步骤】标签重排序
        # K-Means 的 label 0 不一定是最小的。我们需要按中心点大小排序。
        centers = kmeans.cluster_centers_.flatten()
        # argsort 返回的是：从小到大的中心点对应的原始 label 索引
        sorted_indices = np.argsort(centers)
        
        # 创建映射字典: 旧 label -> 新 label (0=最小, K-1=最大)
        # 例如: centers=[100, 1, 50] -> sorted_indices=[1, 2, 0]
        # map: {1:0, 2:1, 0:2}
        label_map = {old_lbl: new_lbl for new_lbl, old_lbl in enumerate(sorted_indices)}
        
        # 应用映射
        # 使用 numpy 向量化操作加速映射
        # 也就是：labels_new[i] = label_map[labels[i]]
        # 既然 label_map 是 0..K-1 的置换，可以用数组索引代替字典查找
        # 构建一个 lookup table
        lookup = np.zeros(K, dtype=int)
        for old, new in label_map.items():
            lookup[old] = new
            
        posts["stratum"] = lookup[labels]
        
        return posts
    
    # ----------------------------
    # Pilot 阶段分配
    # ----------------------------
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

    # ----------------------------
    # Pilot 采样与统计 (修改版)
    # ----------------------------
    def pilot_stats(
            self,
            posts: pd.DataFrame,
            pilot_alloc: Dict[int, int],
            pilot_sampling_method: str = "uniform"  # "uniform" 或 "importance"
    ):
        """
        pilot_sampling_method:
            "uniform" —— 原逻辑，层内均匀采样
            "importance" —— 按 sqrt(proxy * a) 做近似重要性采样（无放回）
        """
        stats, pilots = {}, {}

        for k, grp in posts.groupby("stratum"):
            Nk = len(grp)
            n1 = int(min(pilot_alloc.get(k, 0), Nk))

            if n1 <= 0:
                stats[k] = {"W_k": float(grp["a"].sum()), "p_hat": 0.0,
                            "sigma_hat": 0.0, "N_k": Nk, "n1": 0}
                pilots[k] = pd.DataFrame(columns=posts.columns)
                continue

            # ========================================================
            # 第一阶段采样方式选择：uniform 或 importance
            # ========================================================
            if pilot_sampling_method == "uniform":
                # -----------------------
                # 层内均匀采样（你原来的逻辑）
                # -----------------------
                sample = grp.sample(
                    n1,
                    replace=False,
                    random_state=np.random.randint(1 << 30)
                ).copy()

            elif pilot_sampling_method == "importance":
                # -----------------------------------------------
                # 层内重要性采样（无放回），权重 ∝ sqrt(proxy * a)
                # -----------------------------------------------
                print('[check importance sampling in pilot stage]')
                eps = 1e-8
                imp_weights = np.sqrt((grp["proxy"] * grp["a"]).clip(lower=0)) + eps
                prob = imp_weights / imp_weights.sum()

                # 近似无偏的无放回重要性采样（抽样比例小可视为无偏）
                sample_indices = np.random.choice(
                    grp.index,
                    size=n1,
                    replace=False,
                    p=prob.values
                )
                sample = grp.loc[sample_indices].copy()

            else:
                raise ValueError(f"Unknown pilot_sampling_method: {pilot_sampling_method}")

            # ========================================================
            # 计算 pilot 阶段的估计量（与采样方式无关）
            # ========================================================
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

    # ----------------------------
    # 新增：基于 Proxy 和 Weight 的启发式分配 (不需要 Pilot 统计)
    # ----------------------------
    @staticmethod
    def allocate_second_stage_heuristic(posts: pd.DataFrame, N2: int, strategy: str = "root_wp") -> Dict[int, int]:
        """
        根据全局的 proxy 和 a (weight) 直接计算分配权重，不需要 Pilot 的采样方差。
        """
        alloc_weights = {}
        
        # 遍历每一层
        for k, grp in posts.groupby("stratum"):
            if grp.empty:
                alloc_weights[k] = 0.0
                continue
                
            a_vals = grp["a"].values
            p_vals = grp["proxy"].values
            
            if strategy == "root_wp":
                # 方法 1: sum( w * sqrt(p) )
                # 对应理论上的 sum( sqrt(Var_i) )，假设 Var_i ∝ w^2 * p * (1-p) ≈ w^2 * p
                # 开根号后即 w * sqrt(p)
                w_h = np.sum(a_vals * np.sqrt(p_vals + 1e-12))
            elif strategy == "sqrt_wp": 
                # [改进策略 1] sum( sqrt(w * p) )
                # 与层内重要性采样的权重完全匹配 (Matched IS)
                # 解决了 w 跨度过大导致分配不均的问题
                w_h = np.sum(np.sqrt(a_vals * p_vals + 1e-12))
            elif strategy == "neyman_bernoulli":
                # [改进策略 2] sum( w * sqrt(p * (1-p)) )
                # 经典的方差最小化分配，关注不确定性 (p=0.5)
                # 加上 1e-6 防止 p=1 或 p=0 时权重为 0 (导致无法分配)
                sigma = np.sqrt(p_vals * (1 - p_vals) + 1e-6)
                w_h = np.sum(a_vals * sigma)
            elif strategy == "prop_value":
                # [改进策略 3] sum( w * p )
                # 直接按期望值分配，适合 Proxy 极准且分布极度偏斜的情况
                w_h = np.sum(a_vals * p_vals)
            elif strategy == "w_root_mean_p":
                # 方法 2: sum(w) * sqrt( mean(p) )
                sum_w = np.sum(a_vals)
                mean_p = np.mean(p_vals)
                w_h = sum_w * np.sqrt(mean_p + 1e-12)
            else:
                w_h = 0.0
            
            alloc_weights[k] = w_h

        # 标准化并分配 N2
        total_w = sum(alloc_weights.values())
        if total_w <= 0:
            # 如果全是0，均匀分配
            n_strata = len(alloc_weights)
            return {k: int(N2 / n_strata) for k in alloc_weights}

        final_alloc = {}
        remainder = N2
        
        # 第一次向下取整分配
        for k in alloc_weights:
            share = (alloc_weights[k] / total_w) * N2
            count = int(math.floor(share))
            final_alloc[k] = max(1, count) # 确保每层至少分1个(如果有预算)
            remainder -= final_alloc[k]
            
        # 如果预算超了（因为max(1)），简单扣减；如果少了，补给权重最大的
        # 这里简化处理：按权重排序补齐剩余
        if remainder > 0:
            sorted_strata = sorted(alloc_weights.keys(), key=lambda x: alloc_weights[x], reverse=True)
            for i in range(remainder):
                k = sorted_strata[i % len(sorted_strata)]
                final_alloc[k] += 1
        
        return final_alloc



    # ----------------------------
    # 第二阶段分配
    # ----------------------------
    # V1.0
    @staticmethod
    def allocate_second_stage(stats: Dict[int, dict], N2: int) -> Dict[int, int]:
        weights = {k: math.sqrt(max(1e-12, st["p_hat"]) * max(1e-12, st["sigma_hat"])) for k, st in stats.items()}
        total_w = sum(weights.values()) or 1e-12
        alloc = {k: max(1, int(N2 * weights[k] / total_w)) for k in stats}
        return alloc

    # ----------------------------
    # 第二阶段采样与估计
    # ----------------------------
  
    # #  v0.0: 重要性采样无放回抽样
    # @staticmethod
    # def second_stage_and_estimate(posts: pd.DataFrame, pilots: Dict[int, pd.DataFrame],
    #                               alloc: Dict[int, int], sampling: str = "uniform") -> Dict:
    #     combined = {}
    #     summaries = {}
    #     all_sampled_frames = []

    #     for k, grp in posts.groupby("stratum"):
    #         pilot = pilots.get(k, pd.DataFrame(columns=posts.columns))
    #         pilot_ids = set(pilot["id:ID"].tolist()) if not pilot.empty else set()
    #         remaining = grp[~grp["id:ID"].isin(pilot_ids)]
    #         n2 = alloc.get(k, 0)
    #         if sampling == "uniform":
    #             add_sample = remaining.sample(min(n2, len(remaining)), replace=False)
    #             pi = len(add_sample) / len(remaining)
    #             add_sample["pi"] = pi
    #         else:  # importance sampling √(proxy*a)
    #             weights = np.sqrt(remaining["proxy"].values * remaining["a"].values + 1e-10)
    #             # if len(weights) > 0: print(
    #             #     f"Stratum {k} Weights -> Min: {np.min(weights):.6f}, Max: {np.max(weights):.6f}, Mean: {np.mean(weights):.6f}")
    #             probs = weights / weights.sum() if weights.sum() > 0 else np.ones(len(remaining)) / len(remaining)
    #             # if len(probs) > 0:
    #             #     print(
    #             #         f"[stratum={k}] probs stats: "
    #             #         f"min={probs.min():.6g}, max={probs.max():.6g}, "
    #             #         f"mean={probs.mean():.6g}, std={probs.std():.6g}"
    #             #     )
    #             rng = np.random.default_rng()
    #             idx = rng.choice(len(remaining), size=min(n2, len(remaining)), replace=False, p=probs)
    #             add_sample = remaining.iloc[idx].copy()
    #             pi = np.minimum(1.0, n2 * probs[idx]) if len(probs) > 0 else np.array([])
    #             add_sample["pi"] = pi

    #         final = pd.concat([pilot, add_sample], ignore_index=True, sort=False)
    #         final["Y"] = final["a"] * final["oracle"]
    #         if "pi" not in final:
    #             pi_pilot = len(pilot) / len(grp) if len(grp) > 0 else 1.0
    #             final["pi"] = pi_pilot
    #         T_hat_k = np.sum(final["Y"] / final["pi"])
    #         summaries[k] = {"T_hat": T_hat_k}
    #         combined[k] = final
    #         all_sampled_frames.append(final)

    #     T_hat = sum(v["T_hat"] for v in summaries.values())
    #     full_sample = pd.concat(all_sampled_frames) if all_sampled_frames else pd.DataFrame()
    #     return {"T_hat": T_hat, "full_sample": full_sample}

    # v1.0: 重要性采样有放回抽样 + 预算去重优化
    @staticmethod
    def second_stage_and_estimate(posts: pd.DataFrame, pilots: Dict[int, pd.DataFrame],
                                  alloc: Dict[int, int], sampling: str = "uniform") -> Dict:
        """
        Stage 2 采样。
        修改：如果是 'importance' 采样，使用有放回 / 无放回 + 预算去重优化，以保证无偏性并减小方差。
        """
        combined = {}
        summaries = {}
        all_sampled_frames = []
        # print('[WS check second stage sampling unbaised]')
        for k, grp in posts.groupby("stratum"):
            # 1. 准备 Pilot 数据
            pilot = pilots.get(k, pd.DataFrame(columns=posts.columns))
            pilot_ids = set(pilot["id:ID"].tolist()) if not pilot.empty else set()
            
            # 2. 准备剩余集合 (Remaining)用于 Stage 2
            # 逻辑：Stage 2 补充采样，不应该与 Pilot 重复。
            # 严格双阶段独立采样：Stage 2 应该在由 Remaining 组成的总集中独立采样。
            remaining = grp[~grp["id:ID"].isin(pilot_ids)]
            n2_budget = alloc.get(k, 0) # 这是分配给 Stage 2 的物理预算 (Unique Count)

            add_sample = pd.DataFrame()
            T_hat_stage2 = 0.0
            
            # 如果没有预算或没有数据，Stage 2 贡献为 0
            if n2_budget > 0 and not remaining.empty:
                if sampling == "uniform":
                    # 均匀采样保持原样 (简单随机无放回是无偏的，只要用 N/n 加权)
                    actual_n2 = min(n2_budget, len(remaining))
                    add_sample = remaining.sample(actual_n2, replace=False)
                    # HT 权重 = N_rem / n2
                    weight = len(remaining) / actual_n2
                    
                    # 暂存 pi 以兼容旧逻辑 (均匀采样下 pi = 1/weight)
                    add_sample["pi"] = 1.0 / weight
                    
                    # 计算 HT 估计值
                    add_sample["Y"] = add_sample["a"] * add_sample["oracle"]
                    T_hat_stage2 = add_sample["Y"].sum() * weight
                elif sampling == "importance_nrs":
                    # print('[check systematic sampling in second stage]')
                    # 1. 准备基础权重
                    eps = 1e-10
                    # p ∝ sqrt(proxy * a)
                    w = np.sqrt(remaining["proxy"].values * remaining["a"].values + eps)
                    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # 2. 确定采样预算 n
                    n_target = min(n2_budget, len(remaining))
                    
                    if n_target > 0:
                        sum_w = w.sum()
                        if sum_w <= 0:
                            probs = np.ones(len(remaining)) / len(remaining)
                        else:
                            probs = w / sum_w

                        # === 系统采样 (Systematic Sampling) 无校准版 ===
                        
                        # A. 计算名义包含概率 (Nominal Inclusion Probability)
                        # 这是每个样本在采样数轴上占据的长度，可能大于 1.0
                        nominal_pi = n_target * probs
                        
                        # B. 计算实际用于估计的概率 (Actual Pi)
                        # 在无放回采样中，概率上限为 1.0
                        # 如果 nominal_pi > 1，说明该样本占据长度 > 1，必然被系统采样击中
                        pi_used = np.minimum(1.0, nominal_pi)
                        
                        # C. 随机打乱 (Random Permutation)
                        # 必须打乱，以消除原始数据排序可能带来的周期性偏差
                        rng = np.random.default_rng()
                        perm_indices = rng.permutation(len(remaining))
                        
                        # 按照打乱后的顺序排列名义概率 (构建数轴段落)
                        perm_nominal_pi = nominal_pi[perm_indices]
                        
                        # D. 构建累积分布 (数轴)
                        # cumsum[-1] 理论上应等于 n_target (忽略浮点微小误差)
                        cumsum = np.cumsum(perm_nominal_pi)
                        total_length = cumsum[-1]
                        
                        # E. 生成采样点 (Fixed Interval = 1.0)
                        # 随机起始点 u ~ [0, 1)
                        u = rng.uniform(0, 1)
                        # 生成点: u, u+1, u+2 ... 直到数轴结束
                        sample_points = np.arange(u, total_length, 1.0)
                        
                        # F. 确定被选中的区间索引
                        # searchsorted 找出采样点落在哪一段
                        selected_positions = np.searchsorted(cumsum, sample_points)
                        
                        # 边界保护 (防止浮点误差导致越界)
                        selected_positions = np.clip(selected_positions, 0, len(remaining) - 1)
                        
                        # G. 映射回原始索引并去重
                        # 关键：由于 nominal_pi 可能 > 1，同一个样本可能覆盖了多个采样点
                        # unique 去重实现了“无放回”逻辑，只关心“是否被选中”
                        sampled_perm_indices = np.unique(selected_positions)
                        final_sample_idx = perm_indices[sampled_perm_indices]
                        
                        # === 估计 ===
                        # 提取被选中的行
                        add_sample = remaining.iloc[final_sample_idx].copy()
                        pi_vals = pi_used[final_sample_idx]
                        
                        # Horvitz-Thompson 估计: sum( y_i / pi_i )
                        y_vals = add_sample["a"].values * add_sample["oracle"].values
                        # 避免除以极小值
                        estimate_terms = y_vals / (pi_vals + 1e-12)
                        T_hat_stage2 = np.sum(estimate_terms)
                        
                        add_sample["pi"] = pi_vals
                    
                    else:
                        T_hat_stage2 = 0.0
                else:  # importance sampling
                    # --- 【关键修改】有放回 + 预算优化 ---
                    
                    # 1. 计算分布
                    # weights = sqrt(proxy * a) 推荐
                    w = np.sqrt(remaining["proxy"].values * remaining["a"].values + 1e-10)
                    sum_w = w.sum()
                    if sum_w == 0:
                        probs = np.ones(len(remaining)) / len(remaining)
                    else:
                        probs = w / sum_w

                    # 2. 循环采样 (有放回)
                    rng = np.random.default_rng()
                    unique_indices = set()
                    sampled_indices = [] # 记录所有 trial
                    
                    # 批量采样优化
                    # 初始批量稍微大一点，假设有一半是新的
                    batch_size = max(50, int(n2_budget * 1.5))
                    
                    while len(unique_indices) < n2_budget:
                        # 还需要多少个 unique
                        needed = n2_budget - len(unique_indices)
                        curr_batch = max(needed, 50)
                        
                        # 有放回抽取
                        raw_idx = rng.choice(len(remaining), size=curr_batch, replace=True, p=probs)
                        
                        for idx in raw_idx:
                            # 预算逻辑：只对新样本扣费
                            if idx not in unique_indices:
                                if len(unique_indices) >= n2_budget:
                                    break # 预算满，该样本不能算入
                                unique_indices.add(idx)
                            
                            # 统计逻辑：所有试(Trial)都算 (Hansen-Hurwitz)
                            # 只要预算没被截断，这个样本就是有效的统计点
                            sampled_indices.append(idx)
                        
                        # 这次循环如果预算满了 break 出去的，最外层 while 也会检测到并退出
                        
                        # 简单防死循环 (如剩余权重全为0或仅剩极少有效样本)
                        if len(sampled_indices) > n2_budget * 200 and len(unique_indices) < n2_budget:
                            break

                    # 3. 构造样本 DataFrame (包含重复行)
                    add_sample = remaining.iloc[sampled_indices].copy()
                    sample_probs = probs[sampled_indices]
                    
                    # 4. 计算 Hansen-Hurwitz 估计量 (仅针对 Remaining 部分！)
                    # (1/n) * sum( y_i / p_i )
                    Y_vals = add_sample["a"].values * add_sample["oracle"].values
                    n_trials = len(sampled_indices)
                    
                    if n_trials > 0:
                        # 这是对 "Remaining集合总值" 的估计
                        T_hat_stage2 = np.sum(Y_vals / sample_probs) / n_trials
                    else:
                        T_hat_stage2 = 0.0
                    
                    # 为了兼容后续可能的 pi 访问 (虽然 HH 不需要 pi)
                    add_sample["pi"] = 1.0 

            # ==========================================
            # 合并估计结果 (Stratified Estimator)
            # Total = Total_Pilot + Total_Stage2
            # ==========================================
            
            # 1. Pilot 部分的贡献 (Census of Pilot samples)
            # Pilot 是无放回采的，且我们后续不回采，所以它也是这一层总量的一部分
            Y_pilot_sum = 0.0
            if not pilot.empty:
                # 确保计算正确
                pilot["Y"] = pilot["a"] * pilot["oracle"]
                Y_pilot_sum = pilot["Y"].sum()
                
            # 2. 层总估计
            T_hat_k = Y_pilot_sum + T_hat_stage2
            
            summaries[k] = {"T_hat": T_hat_k}
            
            # 3. 合并样本数据用于返回 (只用于分析，不用于重算 T_hat)
            # 注意：add_sample 可能包含重复行，这对于后续分析 (unique nodes) 没问题，会自动去重
            final = pd.concat([pilot, add_sample], ignore_index=True, sort=False)
            
            combined[k] = final
            all_sampled_frames.append(final)

        # 汇总所有层
        T_hat = sum(v["T_hat"] for v in summaries.values())
        full_sample = pd.concat(all_sampled_frames) if all_sampled_frames else pd.DataFrame()
        
        return {"T_hat": T_hat, "full_sample": full_sample}


    # ----------------------------
    # 核心执行函数
    # ----------------------------
    def run(self, stratify_mode: str = "proxy", sampling: str = "uniform", alloc_strategy: str = "neyman_pilot",
            force_oracle: bool = False) -> Dict:
        """
        alloc_strategy: 
          - "neyman_pilot": 使用 Pilot 样本的方差估计 (原方法)
          - "root_wp": 使用 sum(w * sqrt(p))
          - "w_root_mean_p": 使用 sum(w) * sqrt(mean(p))
        """
        if self.posts.empty:
            return {"T_hat": 0.0, "T_true": self.T_true, "Qerror": 0.0, "n_post": 0, "n_comment": 0} # +++ 返回 0 计数
        posts = self.posts.copy()

        # ==========================================
        # === [逻辑分支 1] 强制全量 Oracle (针对小数据集) ===
        # ==========================================
        if force_oracle:
             # 不做分层，不做采样，直接计算总和
             # 相当于对所有行都进行了 Oracle 检查
             T_hat = (self.posts["a"] * self.posts["oracle"]).sum()
             
             # 计算 Qerror (如果 T_true 已知)
             if self.T_true is not None and self.T_true != 0:
                 Qerror = abs(T_hat - self.T_true) / self.T_true
             else:
                 Qerror = 0.0
             
             # 统计所有唯一节点 (全量开销)
             n_post, n_comment = self._count_unique_nodes(self.posts)
             
             # 返回结果 (pi 设为 1.0 表示全采)
             pi_stats = {"pi_min": 1.0, "pi_max": 1.0, "pi_mean": 1.0}
             return {"T_hat": T_hat, "T_true": self.T_true, "Qerror": Qerror, 
                     "n_post": n_post, "n_comment": n_comment, **pi_stats}

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
        # if stratify_mode == "proxy":
        #     posts = self.stratify_by_proxy(posts, actual_K)
        # elif stratify_mode == "proxyE":
        #     posts = self.stratify_by_expected_contrib(posts, actual_K)
        # elif stratify_mode == "cluster":  # <--- 新增分支
        #     posts = self.stratify_by_clustering_1d(posts, actual_K)
        # else:
        #     raise ValueError("Unsupported stratify_mode")
        
        # 3. Pilot 采样 (第一阶段)
        stats_init = {k: {"N_k": len(g), "W_k": g["a"].sum()} for k, g in posts.groupby("stratum")}
        pilot_alloc = self.allocate_pilot_budget(stats_init, N1_total)
        stats, pilots = self.pilot_stats(posts, pilot_alloc)

        # 4. 第二阶段分配 (核心修改点)
        if alloc_strategy == "neyman_pilot":
            # 原方法：基于 Pilot 的 stats 计算
            alloc2 = self.allocate_second_stage(stats, N2)
        else:
            # 新方法：基于全局 Proxy 和 Weight 计算
            alloc2 = self.allocate_second_stage_heuristic(posts, N2, strategy=alloc_strategy)

        
        res = self.second_stage_and_estimate(posts, pilots, alloc2, sampling=sampling)

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
        return {"T_hat": T_hat, "T_true": self.T_true, "Qerror": Qerror, "n_post": n_post, "n_comment": n_comment,**pi_stats}

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
        if self.total_budget_frac < 0.15:
            print(f"[POSSA] Budget={self.total_budget_frac:.2f} -> Mode: NRS (Without Replacement)")
            return self.run_proxyE_alloc_sqrt_wp_nrs()
        else:
            # print(f"[POSSA] Budget={self.total_budget_frac:.2f} -> Mode: WR (With Replacement)")
            return self.run_proxyE_alloc_sqrt_wp()
    
    def run_mab_sampling(self, K: int = 10, budget_frac: float = None, batch_size: int = 10, ucb_scale: float = 1.0):
        """
        基于多臂赌博机 (MAB) 的自适应分层采样。
        改进：
        1. 使用有放回采样 + Hansen-Hurwitz 估计器 (无偏)。
        2. 预算 (Budget) 定义为“唯一 Post 数量” (模拟 Oracle 缓存机制)。
           重复采样的样本不消耗预算，但有助于降低方差。
        """
        # 1. 准备数据与分层
        if self.posts.empty:
            return {"T_hat": 0.0, "T_true": self.T_true, "Qerror": 0.0, "n_post": 0, "n_comment": 0}
        
        posts = self.posts.copy()
        posts = self.stratify_by_expected_contrib(posts, K)
        
        N = len(posts)
        # 这里的 budget 现在指的是 "Target Unique Posts"
        target_unique_budget = int(budget_frac * N) if budget_frac else int(self.total_budget_frac * N)
        target_unique_budget = max(10, target_unique_budget) # 至少采一点
        
        # 2. 初始化状态
        arm_state = {}
        groups = posts.groupby("stratum")
        eps = 1e-10
        
        for k, grp in groups:
            # 计算层内全局概率 (p_i)
            weights = np.sqrt(grp["proxy"].values * grp["a"].values + eps)
            weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            total_w = weights.sum()
            if total_w > 0:
                probs = weights / total_w
            else:
                probs = np.ones(len(grp)) / len(grp)
            
            arm_state[k] = {
                "N_k": len(grp),
                "data": {
                    "a": grp["a"].values,
                    "oracle": grp["oracle"].values,
                    "prob": probs,
                    "orig_id": grp.index.tolist()
                },
                "n_k": 0,         # 采样次数 (Trials)
                "sum_z": 0.0,
                "sum_sq_z": 0.0,
                "mean": 0.0,
                "std": 0.0
            }

        # 3. MAB 循环
        global_unique_ids = set() # 全局已消耗的 Oracle 预算 (去重)
        total_trials = 0          # 总尝试次数 (用于 UCB 计算)
        
        # 3.1 初始预热：每个臂采一点
        init_samples = 2
        for k in arm_state:
            new_ids = self._mab_sample_batch(arm_state[k], init_samples)
            global_unique_ids.update(new_ids)
            total_trials += init_samples

        # 3.2 循环直到唯一节点数达到预算
        # 为了防止死循环 (如 budget > N)，设置由 N 决定的最大尝试上限
        max_trials = N * 50 

        while len(global_unique_ids) < target_unique_budget and total_trials < max_trials:
            best_arm = -1
            max_score = -1.0
            
            # UCB 选择
            for k, state in arm_state.items():
                if state["n_k"] == 0: # 避免除零
                    score = float('inf')
                else:
                    # UCB Score = N_k * (sigma_hat + exploration)
                    # exploration 项随 total_trials 衰减
                    exploration = ucb_scale * np.sqrt(2 * np.log(total_trials) / state["n_k"])
                    sigma_hat = state["std"]
                    score = state["N_k"] * (sigma_hat + exploration)
                
                if score > max_score:
                    max_score = score
                    best_arm = k
            
            if best_arm == -1: break
            
            # 批量采样 (默认每次采一小批，例如 10 个)
            # 注意：即便只需要补充 1 个 budget，这里也可以由于"免费"性质多采几个，
            # 但为了精确控制 budget，我们保持 batch_size 适中。
            self._mab_sample_batch(arm_state[best_arm], batch_size) # 更新 state
            
            # 这里需要一点黑客手段获取刚才采样的 IDs 来更新 global_unique_ids
            # 更干净的做法是 _mab_sample_batch 返回 IDs，但我已修改 _mab_sample_batch 返回新 IDs (见下文)
            
            # 为了避免频繁修改 _mab_sample_batch 的签名导致如果不改下面会报错，
            # 这里建议直接使用下面更新过的 _mab_sample_batch
            
            # 修正逻辑：必须让 _mab_sample_batch 返回采到的 ID
            # 如果不修改 _mab_sample_batch，我们无法知道哪些是新采的。
            # 下面假设已修改 _mab_sample_batch 返回 indices 或 orig_ids。
            
            # --- 重新调用逻辑 (修正) ---
            # 回滚上面的一行，改为：
            sampled_ids = self._mab_sample_batch(arm_state[best_arm], batch_size)
            global_unique_ids.update(sampled_ids)
            total_trials += batch_size

        # 4. 最终估计
        T_hat = 0.0
        for k, state in arm_state.items():
            if state["n_k"] > 0:
                # Hansen-Hurwitz: 每一层的 Total 估计 = mean(z)
                # z = y / p. E[z] = Y_total. state["mean"] 存储的就是 mean(z)
                T_hat += state["mean"] 

        # 5. 统计开销 (使用实际消耗的 Unique Counts)
        n_post = len(global_unique_ids)
        # 这里的 n_comment 只是近似，如果需要精确，需要去查这些 post 有多少 comment
        # 简单起见，从原数据中提取
        if n_post > 0:
            full_sample = posts.loc[list(global_unique_ids)]
            _, n_comment = self._count_unique_nodes(full_sample)
        else:
            n_comment = 0
            
        Qerror = abs(T_hat - self.T_true) / (self.T_true if self.T_true != 0 else 1.0)
        
        return {
            "T_hat": T_hat, 
            "T_true": self.T_true, 
            "Qerror": Qerror, 
            "n_post": n_post, 
            "n_comment": n_comment,
            "trials": total_trials  # 可选：查看实际采样了多少次
        }

    def _mab_sample_batch(self, state, n_batch, replace=True):
        """辅助函数：执行批量采样并更新状态，返回本次采样的原始 ID 列表"""
        data = state["data"]
        N_pool = len(data["a"])
        
        # 有放回采样 (关键：保证无偏)
        indices = np.random.choice(N_pool, size=n_batch, replace=True, p=data["prob"])
        
        # 提取数据
        a_vals = data["a"][indices]
        oracle_vals = data["oracle"][indices]
        p_vals = data["prob"][indices]
        orig_ids = np.array(data["orig_id"])[indices]
        
        # 计算 Hansen-Hurwitz 变量 z = y / p
        y_vals = a_vals * oracle_vals
        z_vals = y_vals / (p_vals + 1e-12)
        
        # 更新统计量
        state["n_k"] += n_batch
        state["sum_z"] += np.sum(z_vals)
        state["sum_sq_z"] += np.sum(z_vals ** 2)
        
        # 更新均值和标准差
        z_bar = state["sum_z"] / state["n_k"]
        state["mean"] = z_bar 
        
        if state["n_k"] > 1:
            var_z = (state["sum_sq_z"] - state["n_k"] * (z_bar ** 2)) / (state["n_k"] - 1)
            state["std"] = np.sqrt(max(1e-12, var_z))
        else:
            state["std"] = state["mean"] # 启发式
            
        return orig_ids
    
    # ==========================================================
    # === 🧩 四种基线方法（Uniform / sqrt(Proxy) / sqrt(Proxy×a) /a * sqrt(proxy)===
    # ==========================================================

    def run_baseline_uniform(self, budget_frac: float = None):
        if self.posts.empty: return {"T_hat": 0.0, "T_true": 0, "Qerror": 0.0, "n_post": 0, "n_comment": 0}
        """均匀采样 HT 估计"""
        posts = self.posts.copy()
        N = len(posts)
        budget = int(budget_frac * N) if budget_frac else int(self.total_budget_frac * N)
        n = min(budget, N)
        sample = posts.sample(n=n, random_state=np.random.randint(1 << 30))
        pi = n / N
        T_hat = ((sample["a"] * sample["oracle"]) / pi).sum()
        Qerror = abs(T_hat - self.T_true) / (self.T_true if self.T_true != 0 else 1.0)
        # +++ 统计节点 +++
        n_post, n_comment = self._count_unique_nodes(sample)
        pi_stats = {"pi_min": pi, "pi_max": pi, "pi_mean": pi}

        return {"T_hat": T_hat, "T_true": self.T_true, "Qerror": Qerror, "n_post": n_post, "n_comment": n_comment,
                **pi_stats}

    def run_baseline_proxy(self, budget_frac: float = None, eps: float = 1e-10):
        """proxy-only 采样"""
        if self.posts.empty:
            return {"T_hat": 0.0, "T_true": self.T_true, "Qerror": 0.0, "n_post": 0, "n_comment": 0}

        """proxy-only 采样 (p ∝ sqrt(proxy))"""
        posts = self.posts.copy()
        N = len(posts)
        budget = int(budget_frac * N) if budget_frac else int(self.total_budget_frac * N)
        n = min(budget, N)

        weights = np.sqrt(posts["proxy"].values + eps)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        probs = weights / (weights.sum() or 1e-12)

        # --- 输出统计 ---
        # print(f"[Proxy] weights: min={weights.min():.6f}, max={weights.max():.6f}, mean={weights.mean():.6f}")
        # print(f"[Proxy] probs  : min={probs.min():.6f},  max={probs.max():.6f},  mean={probs.mean():.6f}")

        rng = np.random.default_rng(np.random.randint(1 << 30))
        sample_idx = rng.choice(N, size=n, replace=False, p=probs)
        sample = posts.iloc[sample_idx]
        pi = np.minimum(1.0, n * probs[sample_idx])
        T_hat = np.sum((sample["a"].values * sample["oracle"].values) / pi)
        Qerror = abs(T_hat - self.T_true) / (self.T_true if self.T_true != 0 else 1.0)
        # 统计节点
        n_post, n_comment = self._count_unique_nodes(sample)
        pi_stats = self._calc_pi_stats(weights)
        return {"T_hat": T_hat, "T_true": self.T_true, "Qerror": Qerror, "n_post": n_post, "n_comment": n_comment, **pi_stats}

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
        # --- 输出统计 ---
        # print(f"[Proxy×A] weights: min={weights.min():.6f}, max={weights.max():.6f}, mean={weights.mean():.6f}")
        # print(f"[Proxy×A] probs  : min={probs.min():.6f},  max={probs.max():.6f},  mean={probs.mean():.6f}")

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

    def run_baseline_proxy_a_unbiased(self, budget_frac: float = None, eps: float = 1e-10):
        """
        [无偏版本] proxy×a 采样 (有放回 + Hansen-Hurwitz)
        """
        if self.posts.empty:
            return {"T_hat": 0.0, "T_true": self.T_true, "Qerror": 0.0, "n_post": 0, "n_comment": 0}

        posts = self.posts.copy()
        N = len(posts)
        
        # 计算样本量
        budget = int(budget_frac * N) if budget_frac else int(self.total_budget_frac * N)
        n = max(1, budget) # 确保至少采样1次

        # 1. 计算权重 (p ∝ sqrt(proxy * a)) - 这是方差最优分布
        weights = np.sqrt(posts["proxy"].values * posts["a"].values + eps)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        
        sum_weights = weights.sum()
        if sum_weights == 0:
             return {"T_hat": 0.0, "T_true": self.T_true, "Qerror": 0.0, 
                "n_post": 0, "n_comment": 0, "pi_min": 0, "pi_max": 0, "pi_mean": 0}
             
        probs = weights / sum_weights

        # 2. 【修改点1】有放回采样 (replace=True)
        rng = np.random.default_rng(np.random.randint(1 << 30))
        # 注意：这里可能会抽到重复的索引
        sample_idx = rng.choice(N, size=n, replace=True, p=probs)
        
        sample = posts.iloc[sample_idx].copy()
        
        # 3. 【修改点2】获取对应的单次选中概率 p_i
        sample_probs = probs[sample_idx]

        # 4. 【修改点3】Hansen-Hurwitz 无偏估计
        # 公式: (1/n) * sum( y_i / p_i )
        # y_i = a * oracle
        y_values = sample["a"].values * sample["oracle"].values
        
        # 这里的 mean() 等价于 (1/n) * sum(...)
        # 注意：即使有重复样本，也要重复计算，不能去重，否则就引入偏差了
        estimate_terms = y_values / sample_probs
        T_hat = np.mean(estimate_terms) # 或者 np.sum(estimate_terms) / n

        Qerror = abs(T_hat - self.T_true) / (self.T_true if self.T_true != 0 else 1.0)

        # 5. 统计开销 (物理开销去重，数学计算不去重)
        # 虽然数学上我们利用了重复样本来保证无偏，但实际上Oracle只需要对唯一节点跑一次
        n_post, n_comment = self._count_unique_nodes(sample)
        
        # 统计 pi (这里展示期望选中次数 n * p_i)
        pi_stats = self._calc_pi_stats(sample_probs * n)

        return {"T_hat": T_hat, "T_true": self.T_true, "Qerror": Qerror, 
                "n_post": n_post, "n_comment": n_comment, **pi_stats}

    def run_baseline_proxy_a_unbiased_test1(self, budget_frac: float = None, eps: float = 1e-10):
        """
        [无偏版本 - 预算优化] proxy×a 采样 (有放回 + Hansen-Hurwitz)
        优化逻辑：当抽到重复样本时，不消耗 Oracle 预算，从而允许进行更多的采样尝试 (Trials)，
        以在相同预算下获得更低的方差。
        """
        # print('[Check_running_baseline_proxy_a_unbiased_optimized]')
        if self.posts.empty:
            return {"T_hat": 0.0, "T_true": self.T_true, "Qerror": 0.0, "n_post": 0, "n_comment": 0}

        posts = self.posts.copy()
        N = len(posts)
        
        # 1. 确定预算 (这里指的是唯一 Oracle 调用的上限)
        budget = int(budget_frac * N) if budget_frac else int(self.total_budget_frac * N)
        if budget <= 0:
             return {"T_hat": 0.0, "T_true": self.T_true, "Qerror": 0.0, "n_post": 0, "n_comment": 0}

        # 2. 计算权重 (p ∝ sqrt(proxy * a))
        weights = np.sqrt(posts["proxy"].values * posts["a"].values + eps)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        
        sum_weights = weights.sum()
        if sum_weights == 0:
             return {"T_hat": 0.0, "T_true": self.T_true, "Qerror": 0.0, 
                "n_post": 0, "n_comment": 0, "pi_min": 0, "pi_max": 0, "pi_mean": 0}
             
        probs = weights / sum_weights

        # 3. 执行自适应采样循环
        rng = np.random.default_rng(np.random.randint(1 << 30))
        
        unique_indices = set()
        all_sampled_indices = [] # 记录每一次采样尝试（包括重复的）
        current_trials = 0
        
        # 优化：批量采样以减少循环开销
        # 初始批量稍微大一点，假设有一半是新的
        batch_size = max(100, int(budget * 1.5))
        
        while len(unique_indices) < budget:
            # 动态调整步长
            remaining_budget = budget - len(unique_indices)
            next_batch = max(remaining_budget, 100)
            
            # 批量生成随机索引
            new_idx_batch = rng.choice(N, size=next_batch, replace=True, p=probs)
            
            for idx in new_idx_batch:
                # 无论是否重复，都是一次有效的统计Trial
                # 只有当样本是新的时，才检查预算是否超标
                if idx not in unique_indices:
                    if len(unique_indices) >= budget:
                        # 预算已满，产生的新样本无法被"支付"，停止采样
                        # 注意：这次尝试因为没有完成（被丢弃），不计入 all_sampled_indices
                        break
                    else:
                        # 预算未满，支付预算，接纳新样本
                        unique_indices.add(idx)
                
                # 如果是旧样本（免费），或者新样本且预算未超（付费），都记录
                all_sampled_indices.append(idx)
                current_trials += 1

            # 防止死循环：如果 probs 分布极其集中，可能所有非零概率的样本都采完了还填不满预算
            # 简单保护：如果连续采了很多都没有新样本，可能已经遍历完有效集
            if current_trials > budget * 50 and len(unique_indices) < budget:
                # 检查是否还有漏网之鱼太耗时，直接break
                break

        # 4. 计算估计量
        # 必须使用完整的 trial 序列，包含重复项
        if current_trials == 0:
            return {"T_hat": 0.0, "T_true": self.T_true, "Qerror": 0.0, "n_post": 0, "n_comment": 0}

        final_sample_idx = np.array(all_sampled_indices)
        sample = posts.iloc[final_sample_idx].copy()
        sample_probs = probs[final_sample_idx]

        # Hansen-Hurwitz 估计量: (1/n) * sum( y_i / p_i )
        y_values = sample["a"].values * sample["oracle"].values
        estimate_terms = y_values / sample_probs
        T_hat = np.mean(estimate_terms) # sum / current_trials

        Qerror = abs(T_hat - self.T_true) / (self.T_true if self.T_true != 0 else 1.0)

        # 5. 统计开销 (基于 Unique 样本)
        # 用 unique_indices 对应的子集来统计准确的节点开销
        unique_sample_df = posts.iloc[list(unique_indices)]
        n_post, n_comment = self._count_unique_nodes(unique_sample_df)

        # 统计 pi 信息 (基于 budget的等效 n)
        # 这里只是为了输出日志对齐，其实数学上真正生效的 n 是 current_trials
        # 我们可以输出实际 trials 的统计
        pi_stats = self._calc_pi_stats(sample_probs * current_trials)

        return {"T_hat": T_hat, "T_true": self.T_true, "Qerror": Qerror, 
                "n_post": n_post, "n_comment": n_comment, **pi_stats}


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


    # ==========================================================
    # === 🧩 主基线1（Fastest估计节点或核心集的频率 + Oralce） ===
    # ==========================================================

    def run_baseline_graph_only(self):
        """
        基线方法：仅图采样 (Graph Only)。
        不进行第二阶段采样，直接计算所有 estimateW > 0 的实例的 sum(estimateW * oracle)。
        此结果是确定性的（对于给定的 aggregated csv），方差为 0。
        """
        # 健壮性检查
        if self.posts.empty:
            return {"T_hat": 0.0, "T_true": self.T_true, "Qerror": 1.0 if self.T_true != 0 else 0.0}

        # 直接计算加权和：estimateW (即列 'a') * oracle
        # 因为不采样，相当于对所有样本进行了检查
        T_hat = (self.posts["a"] * self.posts["oracle"]).sum()

        # 计算 Qerror (MAPE)
        Qerror = abs(T_hat - self.T_true) / (self.T_true if self.T_true != 0 else 1.0)

        # 统计所有候选集中的节点
        n_post, n_comment = self._count_unique_nodes(self.posts)
        pi_stats = {"pi_min": 1.0, "pi_max": 1.0, "pi_mean": 1.0}

        return {"T_hat": T_hat, "T_true": self.T_true, "Qerror": Qerror, "n_post": n_post, "n_comment": n_comment,**pi_stats}

    

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



    # ==========================================================
    # === 🧩 用于测试效率和误差曲线 ===
    # ==========================================================
    def run_baseline_proxy_a_checkpoints_origin(self, budget_fracs, eps: float = 1e-10, seed: int = None):
        """
        FOIS_nrs 的检查点版本：
        - 在最大预算下先一次采样顺序
        - 对不同 budget_frac 取前缀估计
        """
        if self.posts.empty:
            return []

        posts = self.posts.copy()
        N = len(posts)

        # 预算序列处理
        budget_fracs = sorted(list(set(budget_fracs)))
        max_frac = max(budget_fracs)
        max_n = min(int(max_frac * N), N)
        if max_n <= 0:
            return []

        # 重要性分布 (FOIS_nrs: sqrt(proxy * a))
        weights = np.sqrt(posts["proxy"].values * posts["a"].values + eps)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        probs = weights / (weights.sum() or 1e-12)

        rng = np.random.default_rng(seed if seed is not None else np.random.randint(1 << 30))
        sample_idx = rng.choice(N, size=max_n, replace=False, p=probs)

        # 预取数组
        a_vals = posts["a"].values[sample_idx]
        oracle_vals = posts["oracle"].values[sample_idx]
        p_vals = probs[sample_idx]

        results = []
        for frac in budget_fracs:
            n = max(1, int(frac * N))
            n = min(n, max_n)

            # 前缀估计
            pi = np.minimum(1.0, n * p_vals[:n])
            T_hat = np.sum((a_vals[:n] * oracle_vals[:n]) / (pi + 1e-12))

            Qerror = abs(T_hat - self.T_true) / (self.T_true if self.T_true != 0 else 1.0)

            # oracle 预算（唯一 Post + Comment）
            sample_prefix = posts.iloc[sample_idx[:n]]
            n_post, n_comment = self._count_unique_nodes(sample_prefix)
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
    
    def run_baseline_proxy_a_checkpoints(self, budget_fracs, eps: float = 1e-10, seed: int = None):
        """
        [修正版] FOIS_nrs (无放回):
        使用 【随机系统采样 (Randomized Systematic Sampling)】。
        
        特性：
        1. 严格无放回 (UPSWOR)。
        2. 严格无偏 (Strictly Unbiased)。
        3. 包含概率 pi = min(1, n * p_i) 严格成立。
        
        注意：由于系统采样依赖于具体的 n 来构建累积概率轴，无法简单地通过切片前缀来模拟不同预算，
        因此代码会对每个 budget_frac 重新运行一次采样过程。
        """
        # print('[Check_running_baseline_proxy_a_checkpoints]')
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


def compute_T_true(
        gt_path: str,
        id_mapping_path: str,
        post_csv_path: str,
        gt_match_col: str = "u1",  # structure_match_gt.csv 中对应 Post 节点的列名
        prob_col: str = "ML1_oracle1_probability",
        prob_threshold: float = 0.5
) -> float:
    """
    根据精确子图匹配结果计算 T_true，并统计 a>0 的 post 数量。
    """

    # === 1️⃣ 读取文件 ===
    gt_df = pd.read_csv(gt_path)
    idmap_df = pd.read_csv(id_mapping_path)
    post_df = pd.read_csv(post_csv_path)

    # === 2️⃣ 提取 id_mapping 中的 Post 节点映射表 ===
    post_map = idmap_df[idmap_df["type"].str.lower() == "post"][["internal_id", "orig_id"]].copy()

    # === 3️⃣ 结构匹配结果中，选取与 Post 节点对应的列 ===
    if gt_match_col not in gt_df.columns:
        raise ValueError(f"❌ {gt_match_col} 不存在于 structure_match_gt.csv 中的列 {list(gt_df.columns)}")

    gt_df = gt_df[[gt_match_col]].rename(columns={gt_match_col: "internal_id"})

    # === 4️⃣ 将内部ID映射到原始Post ID ===
    merged = gt_df.merge(post_map, on="internal_id", how="left")

    # === 5️⃣ 统计每个Post节点出现次数 ===
    post_counts = merged.groupby("orig_id").size().reset_index(name="st_truth")

    # === 6️⃣ 连接 post.csv，补充 ML1_oracle1_probability 字段 ===
    post_df = post_df.merge(post_counts, left_on="id:ID", right_on="orig_id", how="left")
    post_df["st_truth"] = post_df["st_truth"].fillna(0)

    # === 7️⃣ 计算 oracle 是否成立 ===
    post_df["oracle"] = (post_df[prob_col] > prob_threshold).astype(int)

    # === 8️⃣ 聚合得到 T_true ===
    post_df["true_contrib"] = post_df["st_truth"] * post_df["oracle"]
    T_true = post_df["true_contrib"].sum()

    # === 9️⃣ 统计信息 ===
    total_posts = len(post_df)
    nonzero_a_posts = (post_df["st_truth"] > 0).sum()
    oracle_posts = (post_df["oracle"] > 0).sum()
    both_nonzero = ((post_df["st_truth"] > 0) & (post_df["oracle"] > 0)).sum()

    print(f"\n===== 📊 T_true 统计 =====")
    print(f"总 post 数量: {total_posts}")
    print(f"a>0 (st_truth>0) 的 post 数: {nonzero_a_posts}")
    print(f"oracle=1 (prob>{prob_threshold}) 的 post 数: {oracle_posts}")
    print(f"同时满足 a>0 且 oracle=1 的 post 数: {both_nonzero}")
    print(f"✅ 计算完成: T_true = {T_true:.3f}")

    return T_true


def compute_T_true_polars(
        gt_path: str,
        id_mapping_path: str,
        post_csv_path: str,
        gt_match_col: str = "u1",
        prob_col: str = "ML1_oracle1_probability",
        prob_threshold: float = 0.5,
):
    """
    使用 Polars 高效计算 T_true，逻辑与 Pandas 版本对齐。
    """
    # === Step 1. 惰性读取所有 CSV ===
    try:
        gt_df = pl.scan_csv(gt_path)
        idmap_df = pl.scan_csv(id_mapping_path)
        post_df = pl.scan_csv(post_csv_path)
    except Exception as e:
        print(f"❌ 错误: Polars 无法扫描输入文件。请检查路径和文件权限。")
        print(f"   具体错误: {e}")
        return 0.0

    # === Step 2. 仅保留 type='post' 的映射关系 ===
    post_map = (
        idmap_df
        .filter(pl.col("type").str.to_lowercase() == "post")
        .select(["internal_id", "orig_id"])
    )

    # === Step 3. 选取 gt 中的匹配列并重命名 ===
    gt_df = gt_df.select(
        pl.col(gt_match_col).alias("internal_id")
    )

    # === Step 4. join gt 与 post_map ===
    merged = gt_df.join(post_map, on="internal_id", how="left")

    # === Step 5. 统计每个 orig_id 的出现次数 ===
    post_counts = (
        merged
        .group_by("orig_id")
        .agg(pl.count().alias("st_truth"))
    )

    # === Step 6. join 回 post_df，并计算 T_true ===
    # 注意：这里我们只计算最终的 T_true 值
    # 如果还需要像 Pandas 版本那样的详细统计，需要执行更多聚合操作
    final_lazy_df = (
        post_df
        .join(post_counts, left_on="id:ID", right_on="orig_id", how="left")
        .fill_null(0)
        .with_columns(
            (pl.col(prob_col) > prob_threshold).cast(pl.Int8).alias("oracle"),
        )
        .with_columns(
            (pl.col("st_truth") * pl.col("oracle")).alias("true_contrib")
        )
    )

    # === Step 7. 触发执行并获取结果 ===
    # 我们直接在 collect 内部进行聚合，效率最高
    result = final_lazy_df.select(
        pl.sum("true_contrib").alias("T_true")
    ).collect()

    # 从结果 DataFrame 中提取 T_true 的值
    T_true_value = result["T_true"][0] if result is not None and len(result) > 0 else 0.0

    # <--- 关键修改: 在此处添加打印语句 --->
    print(f"✅ Polars 计算完成: T_true = {T_true_value:.3f}")

    return T_true_value



# ==========================================================
# === 部分 2: 采样评估与报告生成 ===
# ==========================================================
# ==========================================================
# === 主函数：遍历 aggregated_results 目录并执行评估 ===
# ==========================================================

def run_evaluation_for_query(
        aggregated_csv_path: str,
        T_true: float,
        post_proxy: str,
        comment_proxy: str,
        post_oracle: str = "ML1_oracle1_probability",
        comment_oracle: str = "ML2_oracle2_probability",
        runs: int = 50,           # ✅ 修改默认值为 50
        summarys_dir: str = None, # ✅ 新增参数：分次结果保存目录
        query_index: int = -1,    # ✅ 新增参数：查询索引
        gt_match_col: str = "",   # ✅ 新增参数：GT匹配列名
):
    """
    对单个聚合后的查询文件运行所有采样方法并打印结果。
    """
    query_basename = os.path.basename(aggregated_csv_path).replace("aggregated_list_", "").replace(".csv", "")
    print(f"\n\n{'=' * 20} 评估查询: {query_basename} {'=' * 20}")
    print(f"  (使用 T_true = {T_true})")

    # --- 将 T_true 传递给 Sampler ---
    sampler = ProxyStratifiedSampler(
        csv_path=aggregated_csv_path,
        is_multi_predicate=True,
        post_proxy=post_proxy,
        comment_proxy=comment_proxy,
        post_oracle=post_oracle,
        comment_oracle=comment_oracle,
        T_true=T_true
    )

    if sampler.posts.empty:
        print("没有可供采样的数据 (所有实例的 a <= 0)。")
        return None

    # 如果运行次数小于等于2，去极值就没有意义了。
    if runs <= 2:
        trim_extremes = False
    else:
        trim_extremes = True

    all_methods = {
        # "proxy_importance": sampler.run_proxy_importance,
        # "proxy_uniform": sampler.run_proxy_uniform,
        
        # "proxyE_uniform": sampler.run_proxyE_uniform,
        # "UN": sampler.run_baseline_uniform,
        # "PO": sampler.run_baseline_proxy,
        # "MAB": sampler.run_mab_sampling,

        # "FaSTestO": sampler.run_baseline_graph_only,

        "FOIS_nrs": sampler.run_baseline_proxy_a,
        "FOIS_rs": sampler.run_baseline_proxy_a_unbiased_test1,
        "POSS": sampler.run_proxyE_importance,
        "POSSA": sampler.run_possa,
    }

    results = {}
    print(f"将为 {len(all_methods)} 种方法，每种运行 {runs} 次...")

    node_stats_records = []
    
    for name, func in all_methods.items():
        T_list, Q_list = [], []
        post_cnts, comment_cnts = [], []
        
        # --- ✅ 修改点：分次执行循环 ---
        for t in range(runs):
            try:
                out = func()
                
                # 收集用于计算汇总均值的数据
                T_hat = out["T_hat"]
                Qerror = out["Qerror"]
                n_post = out.get("n_post", 0)
                n_comment = out.get("n_comment", 0)
                
                T_list.append(T_hat)
                Q_list.append(Qerror)
                post_cnts.append(n_post)
                comment_cnts.append(n_comment)

                # --- ✅ 修改点：保存单次运行结果到对应文件 ---
                if summarys_dir:
                    run_file = os.path.join(summarys_dir, f"results_summary_run_{t+1}.csv")
                    # 使用追加模式 'a'
                    with open(run_file, "a") as f:
                        # 格式: query_index,query_basename,gt_match_col,T_true,method,T_hat,Qerror,n_post,n_comment
                        line = f"{query_index},{query_basename},{gt_match_col},{T_true},{name},{T_hat},{Qerror},{n_post},{n_comment}\n"
                        f.write(line)
                        
            except Exception as e:
                print(f"[ERROR] 方法 {name} 第 {t+1} 次运行失败: {e}")
                T_list.append(0.0)
                Q_list.append(1.0)
        
        # --- 下面是原来的汇总统计逻辑 ---
        T_list_trimmed = list(T_list)
        Q_list_trimmed = list(Q_list)
        if trim_extremes and len(T_list_trimmed) > 2:
            T_list_trimmed.sort()
            Q_list_trimmed.sort()
            T_list_trimmed = T_list_trimmed[1:-1]
            Q_list_trimmed = Q_list_trimmed[1:-1]
            
        avg_post = int(np.mean(post_cnts)) if post_cnts else 0
        avg_comment = int(np.mean(comment_cnts)) if comment_cnts else 0
        
        results[name] = {
            "T_hat_mean": np.mean(T_list_trimmed),
            "T_hat_std": np.std(T_list_trimmed),
            "Qerror_mean": np.mean(Q_list_trimmed),
            "Qerror_std": np.std(Q_list_trimmed),
            "n_post_mean": avg_post,
            "n_comment_mean": avg_comment
        }
        node_stats_records.append({
            "query_name": query_basename,
            "method": name,
            "post_sampled_cnt": avg_post,
            "comment_sampled_cnt": avg_comment
        })

    # 打印估算结果总结 (保持不变)
    print("\n--- 估算结果总结 ---")
    print(f"{'Method':20s} | {'T_hat (mean ± std)':25s} | {'Qerror (mean ± std)':25s} | {'Samples(Post/Cmt)':18s}")
    print("-" * 100)
    for name, res in results.items():
        print(
            f"{name:20s} | "
            f"{res['T_hat_mean']:>10.3f} ± {res['T_hat_std']:<10.3f} | "
            f"{res['Qerror_mean']:>10.4f} ± {res['Qerror_std']:<10.4f} | "
            f"{res['n_post_mean']:>6d} / {res['n_comment_mean']:<6d}"
        )
    save_node_counts(node_stats_records)
    return results

def multi_predicate_evaluation(dataset_name: str, run_times: int = 20,
    post_proxy_col="ML1_proxy4b1_probability",
    comment_proxy_col="ML2_proxy1_probability",
    post_oracle_col="ML1_oracle2_probability",
    comment_oracle_col="ML2_oracle2_probability"): # ✅ 增加 run_times 参数
    """
    主评估流程：加载/计算T_true，运行采样，生成最终的详细CSV报告。
    """
    print(f"\n{'=' * 10} 开始对数据集 '{dataset_name}' 进行多谓词采样评估 {'=' * 10}")

    # --- 1. 获取 T_true (从缓存或通过计算) ---
    print("\n>>> 步骤 1: 获取所有查询的 T_true...")
    gt_manager = GroundTruthManager(dataset_name=dataset_name,
                                    post_oracle_col=post_oracle_col,
                                    comment_oracle_col=comment_oracle_col,)
    all_T_true_results = gt_manager.get_all()
    if not all_T_true_results:
        print("[错误] 未能获取 T_true，评估无法继续。")
        return

    # --- 路径配置 ---
    base_path = f"/home/wangshuo/resource/datasets/parler_data/{dataset_name}"
    aggregated_dir = os.path.join(base_path, "results", "aggregated_results")
    core_config_path = os.path.join(base_path, "data_graph", "core_nodes_config.json")
    final_report_path = os.path.join(base_path, "results", "results_summary2.csv")

    # --- 代理模型配置 ---
    POST_PROXY_COL = post_proxy_col
    COMMENT_PROXY_COL = comment_proxy_col

    # ✅ --- 准备保存分次结果的目录与文件初始化 ---
    safe_post_name = POST_PROXY_COL.replace("/", "_")
    safe_comment_name = COMMENT_PROXY_COL.replace("/", "_")
    folder_name = f"{safe_post_name}_{safe_comment_name}"
    summarys_dir = os.path.join(base_path, "results", "result_summarys", folder_name)
    os.makedirs(summarys_dir, exist_ok=True)
    print(f"[INFO] 分次实验结果将保存至: {summarys_dir}")

    # 初始化 run_times 个文件 (写表头)
    csv_headers = "query_index,query_basename,gt_match_col,T_true,method,T_hat,Qerror,n_post,n_comment\n"
    for t in range(run_times):
        run_file = os.path.join(summarys_dir, f"results_summary_run_{t+1}.csv")
        with open(run_file, "w") as f:
            f.write(csv_headers)

    # --- 加载核心节点配置 ---
    try:
        with open(core_config_path, 'r') as f:
            core_nodes_config = json.load(f)
    except FileNotFoundError:
        print(f"[错误] 核心节点配置文件不存在: {core_config_path}")
        return

    # --- 2. 遍历【真实存在】的聚合文件并进行评估 ---
    print("\n>>> 步骤 2: 运行采样评估...")
    if not os.path.exists(aggregated_dir):
        print(f"[错误] 聚合结果目录不存在: {aggregated_dir}")
        return

    agg_files = [f for f in os.listdir(aggregated_dir) if f.endswith('.csv')]
    if not agg_files:
        print(f"[警告] 在目录 {aggregated_dir} 中没有找到任何聚合结果文件。")
        return

    final_report_records = []
    sorted_query_basenames = sorted(list(all_T_true_results.keys()))

    for agg_file in sorted(agg_files):
        # 解析 query_basename
        if agg_file.startswith("aggregated_list_"):
            base = agg_file.replace("aggregated_list_", "")
        elif agg_file.startswith("aggregated_wide_"):
            base = agg_file.replace("aggregated_wide_", "")
        else:
            base = agg_file
        query_basename = base.replace(".csv", "") + ".graph"

        # 查找 T_true
        T_true_for_query = all_T_true_results.get(query_basename)
        if T_true_for_query is None:
            print(f"[警告] 在 T_true 缓存中没有找到查询 '{query_basename}' 的值，跳过评估。")
            continue

        filepath = os.path.join(aggregated_dir, agg_file)

        # ✅ --- 提前计算 query_index 和 gt_match_col 以便传给子函数 ---
        try:
            query_index = sorted_query_basenames.index(query_basename)
        except ValueError:
            query_index = -1
        core_nodes = core_nodes_config.get(query_basename, {})
        gt_match_col_str = ";".join([f"u{vid}" for label in core_nodes for vid in core_nodes[label]])

        # 调用修改后的 run_evaluation_for_query
        query_results = run_evaluation_for_query(
            aggregated_csv_path=filepath,
            T_true=T_true_for_query,
            post_proxy=POST_PROXY_COL,
            comment_proxy=COMMENT_PROXY_COL,
            post_oracle=post_oracle_col,
            comment_oracle=comment_oracle_col,
            runs=run_times,              # ✅ 传入次数
            summarys_dir=summarys_dir,   # ✅ 传入保存目录
            query_index=query_index,     # ✅ 传入索引
            gt_match_col=gt_match_col_str # ✅ 传入GT列
        )

        if query_results:
            for method_name, metrics in query_results.items():
                record = {
                    "query_index": query_index,
                    "query_basename": query_basename,
                    "gt_match_col": gt_match_col_str,
                    "T_true": T_true_for_query,
                    "method": method_name,
                    "T_hat_mean": metrics["T_hat_mean"],
                    "T_hat_std": metrics["T_hat_std"],
                    "Qerror_mean": metrics["Qerror_mean"],
                    "Qerror_std": metrics["Qerror_std"],
                }
                final_report_records.append(record)

    # --- 3. 生成并保存最终的CSV报告 ---
    if not final_report_records:
        print("\n[完成] 没有生成任何评估结果。")
        return

    report_df = pd.DataFrame.from_records(final_report_records)
    report_df.sort_values(by=['query_index', 'method'], inplace=True)
    report_df.to_csv(final_report_path, index=False)

    print(f"\n\n{'=' * 15} 最终评估报告 {'=' * 15}")
    print(f"✅ 详细评估报告已保存到: {final_report_path}")
    print(f"✅ 分次详细数据已保存至: {summarys_dir} (共 {run_times} 个 run_*.csv 文件)")

from tqdm import tqdm  # 导入进度条库

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
    print(f"\n✅ 已将 [baseline_graph_only] 结果追加到: {final_report_path}")
    print(f"    新增记录数: {len(new_df)}")
    print(f"    总记录数: {len(final_df)}")
    if node_stats_records:
        save_node_counts(node_stats_records)
        print(f"✅ 节点采样统计已追加到: {SAMPLED_COUNT_FILE}")

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


# ==========================================================
# === 部分 3: 误差与oracle预算曲线评估测试 ===
# ==========================================================
# ==========================================================
# === xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx ===
# ==========================================================



def run_budget_curve_multi_predicate_fast(
    dataset_name: str,
    budget_fracs: List[float],
    run_times: int = 5,
    post_proxy_col: str = "ML1_proxy4b_probability",
    comment_proxy_col: str = "ML2_proxy1_probability",
    post_oracle_col: str = "ML1_oracle2_probability",
    comment_oracle_col: str = "ML2_oracle2_probability",
):
    """
    同时生成 FOIS_nrs / FOIS_rs / POSS 的预算曲线
    并输出每种方法针对每个 Query 的中间平均误差结果。
    """
    print(f"\n====== Budget Curve (FOIS_nrs / FOIS_rs / POSS): {dataset_name} ======")

    gt_manager = GroundTruthManager(dataset_name=dataset_name,
                                    post_oracle_col=post_oracle_col,
                                    comment_oracle_col=comment_oracle_col)
    all_T_true_results = gt_manager.get_all()
    if not all_T_true_results:
        print("[Error] 未获取到 T_true")
        return None

    base_path = f"/home/wangshuo/resource/datasets/parler_data/{dataset_name}"
    aggregated_dir = os.path.join(base_path, "results", "aggregated_results")
    if not os.path.exists(aggregated_dir):
        print(f"[Error] 聚合目录不存在: {aggregated_dir}")
        return None

    agg_files = [f for f in os.listdir(aggregated_dir) if f.endswith(".csv")]
    if not agg_files:
        print("[Warn] 没有聚合文件")
        return None

    records = []
    
    # 按照文件名排序，保证输出顺序稳定
    for agg_file in sorted(agg_files):
        if agg_file.startswith("aggregated_list_"):
            base = agg_file.replace("aggregated_list_", "")
        else:
            base = agg_file
        query_basename = base.replace(".csv", "") + ".graph"

        T_true = all_T_true_results.get(query_basename)
        if T_true is None or T_true == 0:
            continue

        filepath = os.path.join(aggregated_dir, agg_file)
        sampler = ProxyStratifiedSampler(
            csv_path=filepath,
            is_multi_predicate=True,
            post_proxy=post_proxy_col,
            comment_proxy=comment_proxy_col,
            post_oracle=post_oracle_col,
            comment_oracle=comment_oracle_col,
            T_true=T_true,
            total_budget_frac=max(budget_fracs)
        )

        if sampler.posts.empty:
            continue

        # --- [新增模块 1] 初始化当前查询的临时统计容器 ---
        # 结构: { "FOIS_nrs": {0.01: [], 0.05: []}, "FOIS_rs": {...}, "POSS": {...} }
        method_names = ["FOIS_nrs", "FOIS_rs", "POSS"]
        temp_method_errors = {
            m: {b: [] for b in budget_fracs} 
            for m in method_names
        }

        for run_id in range(run_times):
            # 1. 运行 FOIS_nrs
            res_nrs = sampler.run_baseline_proxy_a_checkpoints(budget_fracs)
            # 2. 运行 FOIS_rs
            res_rs = sampler.run_baseline_proxy_a_unbiased_checkpoints(budget_fracs)
            # 3. 运行 POSS
            # res_poss = sampler.run_proxyE_importance_checkpoints(budget_fracs)

            # --- [新增模块 2] 收集数据 & 填充临时统计 ---
            
            # 处理 FOIS_nrs
            for rec in res_nrs:
                records.append({
                    "query_basename": query_basename,
                    "run_id": run_id + 1,
                    "budget_frac": rec["budget_frac"],
                    "budget_n": rec["budget_n"],
                    "T_true": T_true,
                    "T_hat": rec["T_hat"],
                    "Qerror": rec["Qerror"],
                    "n_post": rec["n_post"],
                    "n_comment": rec["n_comment"],
                    "oracle_cost": rec["oracle_cost"],
                    "method": "FOIS_nrs"
                })
                # 记录误差用于打印
                if rec["budget_frac"] in temp_method_errors["FOIS_nrs"]:
                    temp_method_errors["FOIS_nrs"][rec["budget_frac"]].append(rec["Qerror"])

            # 处理 FOIS_rs
            for rec in res_rs:
                records.append({
                    "query_basename": query_basename,
                    "run_id": run_id + 1,
                    "budget_frac": rec["budget_frac"],
                    "budget_n": rec["budget_n"],
                    "T_true": T_true,
                    "T_hat": rec["T_hat"],
                    "Qerror": rec["Qerror"],
                    "n_post": rec["n_post"],
                    "n_comment": rec["n_comment"],
                    "oracle_cost": rec["oracle_cost"],
                    "method": "FOIS_rs"
                })
                # 记录误差用于打印
                if rec["budget_frac"] in temp_method_errors["FOIS_rs"]:
                    temp_method_errors["FOIS_rs"][rec["budget_frac"]].append(rec["Qerror"])

            # # 处理 POSS
            # for rec in res_poss:
            #     records.append({
            #         "query_basename": query_basename,
            #         "run_id": run_id + 1,
            #         "budget_frac": rec["budget_frac"],
            #         "budget_n": rec["budget_n"],
            #         "T_true": T_true,
            #         "T_hat": rec["T_hat"],
            #         "Qerror": rec["Qerror"],
            #         "n_post": rec["n_post"],
            #         "n_comment": rec["n_comment"],
            #         "oracle_cost": rec["oracle_cost"],
            #         "method": "POSS"
            #     })
            #     # 记录误差用于打印
            #     if rec["budget_frac"] in temp_method_errors["POSS"]:
            #         temp_method_errors["POSS"][rec["budget_frac"]].append(rec["Qerror"])

        # --- [新增模块 3] 当前查询的所有 Run 结束后，格式化打印每种方法的平均误差 ---
        print(f"Query: {query_basename:<35} | T_true: {int(T_true)}")
        
        for m in method_names:
            # 构建该方法在不同 budget 下的误差字符串
            stats_str_list = []
            for b in budget_fracs:
                errs = temp_method_errors[m][b]
                if errs:
                    avg_err = np.mean(errs)
                    stats_str_list.append(f"B={b:.2f}: {avg_err:.4f}")
            
            # 打印单行: [Method Name] B=0.01: 0.xxx | B=0.05: 0.xxx
            final_str = " | ".join(stats_str_list)
            print(f"  [{m:<8}] {final_str}")
            
        print("-" * 80) # 分隔线

    if not records:
        print("[Warn] 无结果生成")
        return None

    df = pd.DataFrame(records)
    out_dir = os.path.join(base_path, "results", "efficiency")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "FOIS_rs_POSS_budget_curve.csv")
    file_exists = os.path.exists(out_path)
    df.to_csv(out_path, index=False, mode='a', header=not file_exists)
    action_str = "追加到" if file_exists else "创建新文件"
    print(f"\n[OK] 结果已{action_str}: {out_path}")
    return df

def run_budget_curve_multi_predicate(
    dataset_name: str,
    budget_fracs: List[float],
    run_times: int = 5,
    post_proxy_col: str = "ML1_proxy4b_probability",
    comment_proxy_col: str = "ML2_proxy1_probability",
    post_oracle_col: str = "ML1_oracle2_probability",
    comment_oracle_col: str = "ML2_oracle2_probability",
):
    """
    同时生成 FOIS_nrs / FOIS_rs / POSS 的预算曲线
    并输出每种方法针对每个 Query 的中间平均误差结果。
    【改进】：每处理完一个 Query，立即将结果追加写入文件。
    """
    print(f"\n====== Budget Curve (FOIS_nrs / FOIS_rs / POSS): {dataset_name} ======")

    # 1. 准备 Ground Truth
    gt_manager = GroundTruthManager(dataset_name=dataset_name,
                                    post_oracle_col=post_oracle_col,
                                    comment_oracle_col=comment_oracle_col)
    all_T_true_results = gt_manager.get_all()
    if not all_T_true_results:
        print("[Error] 未获取到 T_true")
        return None

    # 2. 准备路径
    base_path = f"/home/wangshuo/resource/datasets/parler_data/{dataset_name}"
    aggregated_dir = os.path.join(base_path, "results", "aggregated_results")
    if not os.path.exists(aggregated_dir):
        print(f"[Error] 聚合目录不存在: {aggregated_dir}")
        return None

    agg_files = [f for f in os.listdir(aggregated_dir) if f.endswith(".csv")]
    if not agg_files:
        print("[Warn] 没有聚合文件")
        return None

    # --- [修改点 1] 提前定义输出路径，以便在循环中访问 ---
    out_dir = os.path.join(base_path, "results", "efficiency")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "FOIS_rs_POSS_budget_curve.csv")
    
    # 如果你想每次重新跑都清空旧文件，取消下面注释：
    # if os.path.exists(out_path):
    #     os.remove(out_path)

    all_records = [] # 用于最后返回完整 DataFrame
    
    # 3. 开始遍历查询
    for agg_file in sorted(agg_files):
        # 解析文件名
        if agg_file.startswith("aggregated_list_"):
            base = agg_file.replace("aggregated_list_", "")
        else:
            base = agg_file
        query_basename = base.replace(".csv", "") + ".graph"

        T_true = all_T_true_results.get(query_basename)
        if T_true is None or T_true == 0:
            continue

        filepath = os.path.join(aggregated_dir, agg_file)
        sampler = ProxyStratifiedSampler(
            csv_path=filepath,
            is_multi_predicate=True,
            post_proxy=post_proxy_col,
            comment_proxy=comment_proxy_col,
            post_oracle=post_oracle_col,
            comment_oracle=comment_oracle_col,
            T_true=T_true,
            total_budget_frac=max(budget_fracs)
        )

        if sampler.posts.empty:
            continue

        # 初始化统计容器
        method_names = ["FOIS_nrs", "FOIS_rs", "POSS"]
        temp_method_errors = {
            m: {b: [] for b in budget_fracs} 
            for m in method_names
        }
        
        # --- [修改点 2] 当前查询的临时结果列表 ---
        current_records = []

        for run_id in range(run_times):
            # 1. 运行 FOIS_nrs
            res_nrs = sampler.run_baseline_proxy_a_checkpoints(budget_fracs)
            # 2. 运行 FOIS_rs
            res_rs = sampler.run_baseline_proxy_a_unbiased_checkpoints(budget_fracs)
            # 3. 运行 POSS (目前注释掉)

            # --- 收集 FOIS_nrs ---
            for rec in res_nrs:
                row = {
                    "query_basename": query_basename,
                    "run_id": run_id + 1,
                    "budget_frac": rec["budget_frac"],
                    "budget_n": rec["budget_n"],
                    "T_true": T_true,
                    "T_hat": rec["T_hat"],
                    "Qerror": rec["Qerror"],
                    "n_post": rec["n_post"],
                    "n_comment": rec["n_comment"],
                    "oracle_cost": rec["oracle_cost"],
                    "method": "FOIS_nrs"
                }
                current_records.append(row)
                if rec["budget_frac"] in temp_method_errors["FOIS_nrs"]:
                    temp_method_errors["FOIS_nrs"][rec["budget_frac"]].append(rec["Qerror"])

            # --- 收集 FOIS_rs ---
            for rec in res_rs:
                row = {
                    "query_basename": query_basename,
                    "run_id": run_id + 1,
                    "budget_frac": rec["budget_frac"],
                    "budget_n": rec["budget_n"],
                    "T_true": T_true,
                    "T_hat": rec["T_hat"],
                    "Qerror": rec["Qerror"],
                    "n_post": rec["n_post"],
                    "n_comment": rec["n_comment"],
                    "oracle_cost": rec["oracle_cost"],
                    "method": "FOIS_rs"
                }
                current_records.append(row)
                if rec["budget_frac"] in temp_method_errors["FOIS_rs"]:
                    temp_method_errors["FOIS_rs"][rec["budget_frac"]].append(rec["Qerror"])

            # --- 收集 POSS (如果启用) ---
            # for rec in res_poss:
            #     row = { ... }
            #     current_records.append(row)
            #     ...

        # 将当前查询结果加入总结果
        all_records.extend(current_records)

        # 打印控制台统计
        print(f"Query: {query_basename:<35} | T_true: {int(T_true)}")
        for m in method_names:
            stats_str_list = []
            for b in budget_fracs:
                errs = temp_method_errors[m][b]
                if errs:
                    avg_err = np.mean(errs)
                    stats_str_list.append(f"B={b:.2f}: {avg_err:.4f}")
            final_str = " | ".join(stats_str_list)
            # 只有当该方法有数据时才打印 (避免 POSS 被注释掉时打印空行)
            if final_str:
                print(f"  [{m:<8}] {final_str}")
        
        # --- [修改点 3] 立即写入文件 (追加模式) ---
        if current_records:
            df_chunk = pd.DataFrame(current_records)
            # 检查文件是否存在，决定是否写表头
            file_exists = os.path.exists(out_path)
            # mode='a' 表示追加，header=not file_exists 表示仅在文件新建时写表头
            df_chunk.to_csv(out_path, index=False, mode='a', header=not file_exists)
            print(f"[Saved] appended to {out_path}")
            
        print("-" * 80)

    if not all_records:
        print("[Warn] 无结果生成")
        return None

    print(f"\n[Done] 所有查询处理完毕。最终文件: {out_path}")
    return pd.DataFrame(all_records)


def run_adaptive_sampling_experiment(
    dataset_name: str = "dataset_test",
    run_times: int = 5
):
    """
    按照指定的 budget_frac 列表，对 run_proxy_importance 和 run_proxyE_importance 
    进行两阶段自适应采样评估。结果包含每次运行的详细数据。
    """
    # === 1. 配置参数与路径 ===
    TARGET_TICKS = [0.05, 0.1, 0.2, 0.15, 0.3, 0.4]
    # 为了逻辑清晰，我们在内部排个序，或者保持原样遍历皆可。这里保持原样遍历。
    
    base_path = f"/home/wangshuo/resource/datasets/parler_data/{dataset_name}"
    aggregated_dir = os.path.join(base_path, "results", "aggregated_results")
    
    # T_true JSON 路径
    t_true_path = os.path.join(base_path, "results", "T_true_ML1_oracle2_probability_ML2_oracle2_probability.json")
    
    # 输出 CSV 路径
    output_dir = os.path.join(base_path, "results", "efficiency")
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "two_stage_adaptive_results.csv")
    
    # 代理列名配置 (根据 Dataset_test 的常规配置)
    POST_PROXY = "ML1_proxy4b_probability"
    COMMENT_PROXY = "ML2_proxy1_probability"
    POST_ORACLE = "ML1_oracle2_probability"
    COMMENT_ORACLE = "ML2_oracle2_probability"

    print(f"\n{'='*10} 开始两阶段自适应采样评估 (Adaptive Sampling) {'='*10}")
    print(f"Target Ticks: {TARGET_TICKS}")
    print(f"Output File: {output_csv}")

    # === 2. 加载 T_true ===
    if not os.path.exists(t_true_path):
        print(f"[Error] T_true 文件未找到: {t_true_path}")
        return

    with open(t_true_path, 'r') as f:
        all_t_true = json.load(f)
    print(f"成功加载 {len(all_t_true)} 个查询的 T_true 值。")

    # === 3. 准备结果文件头 ===
    # 如果文件不存在，写入表头
    headers = ["query_basename", "run_id", "budget_frac", "budget_n", "T_true", "T_hat", "Qerror", "n_post", "n_comment", "oracle_cost", "method"]
    if not os.path.exists(output_csv):
        pd.DataFrame(columns=headers).to_csv(output_csv, index=False)

    # === 4. 遍历聚合文件 ===
    if not os.path.exists(aggregated_dir):
        print(f"[Error] 聚合目录不存在: {aggregated_dir}")
        return

    agg_files = sorted([f for f in os.listdir(aggregated_dir) if f.endswith(".csv")])
    
    for file_idx, agg_file in enumerate(agg_files):
        # --- 解析文件名得到 query_basename ---
        # 逻辑：去除前缀，去除后缀，加上 .graph
        if agg_file.startswith("aggregated_list_"):
            base = agg_file.replace("aggregated_list_", "")
        elif agg_file.startswith("aggregated_wide_"):
            base = agg_file.replace("aggregated_wide_", "")
        else:
            base = agg_file
        query_basename = base.replace(".csv", "") + ".graph"

        # 获取 T_true
        T_true = all_t_true.get(query_basename)
        if T_true is None:
            # print(f"[Skip] {query_basename} 在 JSON 中没有对应的 T_true")
            continue

        print(f"\n[{file_idx+1}/{len(agg_files)}] Processing: {query_basename} (T_true={T_true})")

        # --- 初始化 Sampler (只加载一次 CSV) ---
        filepath = os.path.join(aggregated_dir, agg_file)
        # 初始 budget 设为 1.0 (全量)，后续我们会动态修改
        sampler = ProxyStratifiedSampler(
            csv_path=filepath,
            is_multi_predicate=True,
            post_proxy=POST_PROXY,
            comment_proxy=COMMENT_PROXY,
            post_oracle=POST_ORACLE,
            comment_oracle=COMMENT_ORACLE,
            T_true=T_true,
            total_budget_frac=1.0 
        )

        if sampler.posts.empty:
            print("   -> Data Empty, skipping.")
            continue

        total_instances = len(sampler.posts)
        
        # 定义要运行的方法映射
        # 方法名 -> (函数引用)
        methods_map = {
            "run_proxy_importance": sampler.run_proxy_importance,
            "run_proxyE_importance": sampler.run_proxyE_importance
        }

        # === 5. 遍历采样率 (Ticks) ===
        for tick in TARGET_TICKS:
            # 计算对应的 budget_n (物理行数预算)
            budget_n = int(math.floor(tick * total_instances))
            
            # --- !!! 关键步骤：动态更新 Sampler 的预算比例 !!! ---
            # 因为 run_proxy_importance 内部使用 self.total_budget_frac
            sampler.total_budget_frac = tick

            # === 6. 遍历方法 ===
            for method_name, run_func in methods_map.items():
                
                batch_records = []
                qerrors = []

                # === 7. 重复运行 n 次 ===
                for r in range(run_times):
                    run_id = r + 1
                    
                    try:
                        # 执行采样
                        res = run_func()
                        
                        T_hat = res["T_hat"]
                        Qerror = res["Qerror"]
                        n_post = res.get("n_post", 0)
                        n_comment = res.get("n_comment", 0)
                        
                        # 计算 oracle_cost (这里定义为唯一节点之和)
                        oracle_cost = n_post + n_comment

                        # 记录数据
                        record = {
                            "query_basename": query_basename,
                            "run_id": run_id,
                            "budget_frac": tick,
                            "budget_n": budget_n,
                            "T_true": T_true,
                            "T_hat": T_hat,
                            "Qerror": Qerror,
                            "n_post": n_post,
                            "n_comment": n_comment,
                            "oracle_cost": oracle_cost,
                            "method": method_name
                        }
                        batch_records.append(record)
                        qerrors.append(Qerror)
                        
                    except Exception as e:
                        print(f"   [Error] {method_name} run {run_id} failed: {e}")

                # === 8. 保存当前 batch (5次运行) 到 CSV ===
                if batch_records:
                    df_batch = pd.DataFrame(batch_records)
                    # 追加模式写入，不写表头
                    df_batch.to_csv(output_csv, mode='a', header=False, index=False)

                # === 9. 打印平均误差 (作为控制台反馈) ===
                avg_q = np.mean(qerrors) if qerrors else 0.0
                print(f"   -> {method_name:<22} | Tick: {tick:<4} | Avg Qerror: {avg_q:.4f}")

    print(f"\n[Done] 所有实验完成。结果已保存至: {output_csv}")

# ==========================================================
# === 部分 4: 多线程优化执行 ===
# ==========================================================
# ==========================================================
# === xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx ===
# ==========================================================

def _process_budget_curve_worker(
    agg_file: str,
    base_path: str,
    aggregated_dir: str,
    all_t_true: dict,
    budget_fracs: List[float],
    run_times: int,
    config: dict
):
    """
    [Worker 函数] 处理单个聚合文件，运行 FOIS_nrs / FOIS_rs 等基线方法。
    """
    # 1. 解析文件名
    if agg_file.startswith("aggregated_list_"):
        base = agg_file.replace("aggregated_list_", "")
    elif agg_file.startswith("aggregated_wide_"):
        base = agg_file.replace("aggregated_wide_", "")
    else:
        base = agg_file
    query_basename = base.replace(".csv", "") + ".graph"

    # 2. 获取 T_true
    T_true = all_t_true.get(query_basename)
    if T_true is None or T_true == 0:
        return []  # Skip

    filepath = os.path.join(aggregated_dir, agg_file)
    
    # 3. 初始化 Sampler
    try:
        sampler = ProxyStratifiedSampler(
            csv_path=filepath,
            is_multi_predicate=True,
            post_proxy=config["post_proxy"],
            comment_proxy=config["comment_proxy"],
            post_oracle=config["post_oracle"],
            comment_oracle=config["comment_oracle"],
            T_true=T_true,
            total_budget_frac=max(budget_fracs) # 初始化时只需给最大预算
        )
    except Exception as e:
        # print(f"[Worker Error] Init failed for {agg_file}: {e}")
        return []

    if sampler.posts.empty:
        return []

    file_records = []

    # 4. 循环运行多次
    for r in range(run_times):
        run_id = r + 1
        
        # --- A. 运行 FOIS_nrs (无放回) ---
        try:
            res_nrs = sampler.run_baseline_proxy_a_checkpoints(budget_fracs)
            for rec in res_nrs:
                file_records.append({
                    "query_basename": query_basename,
                    "run_id": run_id,
                    "budget_frac": rec["budget_frac"],
                    "budget_n": rec["budget_n"],
                    "T_true": T_true,
                    "T_hat": rec["T_hat"],
                    "Qerror": rec["Qerror"],
                    "n_post": rec["n_post"],
                    "n_comment": rec["n_comment"],
                    "oracle_cost": rec["oracle_cost"],
                    "method": "FOIS_nrs"
                })
        except Exception as e:
            pass # 忽略单个方法错误

        # --- B. 运行 FOIS_rs (有放回/无偏) ---
        try:
            res_rs = sampler.run_baseline_proxy_a_unbiased_checkpoints(budget_fracs)
            for rec in res_rs:
                file_records.append({
                    "query_basename": query_basename,
                    "run_id": run_id,
                    "budget_frac": rec["budget_frac"],
                    "budget_n": rec["budget_n"],
                    "T_true": T_true,
                    "T_hat": rec["T_hat"],
                    "Qerror": rec["Qerror"],
                    "n_post": rec["n_post"],
                    "n_comment": rec["n_comment"],
                    "oracle_cost": rec["oracle_cost"],
                    "method": "FOIS_rs"
                })
        except Exception as e:
            pass
            

    return file_records

def run_budget_curve_multi_predicate_fast(
    dataset_name: str,
    budget_fracs: List[float],
    run_times: int = 5,
    post_proxy_col: str = "ML1_proxy4b_probability",
    comment_proxy_col: str = "ML2_proxy1_probability",
    post_oracle_col: str = "ML1_oracle2_probability",
    comment_oracle_col: str = "ML2_oracle2_probability",
    max_workers: int = None
):
    """
    [多进程加速版] Budget Curve Generator (FOIS_nrs / FOIS_rs)
    """
    print(f"\n====== [Fast MP] Budget Curve (FOIS): {dataset_name} ======")

    # 1. 准备 Ground Truth
    # 注意：这里在主进程统一加载一次 T_true，然后传给子进程，避免子进程重复计算
    print("Loading Ground Truth...")
    gt_manager = GroundTruthManager(dataset_name=dataset_name,
                                    post_oracle_col=post_oracle_col,
                                    comment_oracle_col=comment_oracle_col)
    all_T_true_results = gt_manager.get_all()
    
    if not all_T_true_results:
        print("[Error] 未获取到 T_true")
        return None

    # 2. 准备路径
    base_path = f"/home/wangshuo/resource/datasets/parler_data/{dataset_name}"
    aggregated_dir = os.path.join(base_path, "results", "aggregated_results")
    
    if not os.path.exists(aggregated_dir):
        print(f"[Error] 聚合目录不存在: {aggregated_dir}")
        return None

    agg_files = sorted([f for f in os.listdir(aggregated_dir) if f.endswith(".csv")])
    if not agg_files:
        print("[Warn] 没有聚合文件")
        return None

    out_dir = os.path.join(base_path, "results", "efficiency")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "FOIS_rs_FOSS_nrs_budget_curve_fast.csv")
    
    # 3. 初始化 CSV (写入表头)
    headers = ["query_basename", "run_id", "budget_frac", "budget_n", "T_true", "T_hat", "Qerror", "n_post", "n_comment", "oracle_cost", "method"]
    pd.DataFrame(columns=headers).to_csv(out_path, index=False)

    # 4. 封装配置
    config = {
        "post_proxy": post_proxy_col,
        "comment_proxy": comment_proxy_col,
        "post_oracle": post_oracle_col,
        "comment_oracle": comment_oracle_col
    }

    # 5. 多进程执行
    if max_workers is None:
        max_workers = max(1, os.cpu_count() - 2)
    
    print(f"Starting Process Pool with {max_workers} workers...")
    print(f"Total files to process: {len(agg_files)}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for agg_file in agg_files:
            futures.append(
                executor.submit(
                    _process_budget_curve_worker,
                    agg_file, base_path, aggregated_dir, all_T_true_results,
                    budget_fracs, run_times, config
                )
            )
        
        # 收集结果
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Queries"):
            try:
                records = future.result()
                if records:
                    # 立即写入文件
                    df_chunk = pd.DataFrame(records)
                    # 确保列顺序一致
                    df_chunk = df_chunk[headers] 
                    df_chunk.to_csv(out_path, mode='a', header=False, index=False)
            except Exception as e:
                print(f"Job failed: {e}")

    print(f"\n[Done] 加速运行结束。结果已保存至: {out_path}")



def _process_single_query_file(
    agg_file: str,
    base_path: str,
    aggregated_dir: str,
    all_t_true: dict,
    target_ticks: list,
    run_times: int,
    config: dict
):
    """
    内部工作函数：处理单个聚合文件，完成所有 tick 和 run 的计算。
    返回该文件生成的所有结果记录列表。
    """
    # --- 解析文件名 ---
    if agg_file.startswith("aggregated_list_"):
        base = agg_file.replace("aggregated_list_", "")
    elif agg_file.startswith("aggregated_wide_"):
        base = agg_file.replace("aggregated_wide_", "")
    else:
        base = agg_file
    query_basename = base.replace(".csv", "") + ".graph"

    # 获取 T_true
    T_true = all_t_true.get(query_basename)
    if T_true is None:
        return []  # Skip

    filepath = os.path.join(aggregated_dir, agg_file)
    
    # --- 初始化 Sampler ---
    # 注意：这里在子进程中初始化，避免跨进程传递大对象
    try:
        sampler = ProxyStratifiedSampler(
            csv_path=filepath,
            is_multi_predicate=True,
            post_proxy=config["POST_PROXY"],
            comment_proxy=config["COMMENT_PROXY"],
            post_oracle=config["POST_ORACLE"],
            comment_oracle=config["COMMENT_ORACLE"],
            T_true=T_true,
            total_budget_frac=1.0 # 初始值，后面会覆盖
        )
    except Exception as e:
        # print(f"[Warn] Init failed for {agg_file}: {e}")
        return []

    if sampler.posts.empty:
        return []

    total_instances = len(sampler.posts)
    file_records = []
    
    methods_map = {
        "run_proxy_importance": sampler.run_proxy_importance,
        "run_proxyE_importance": sampler.run_proxyE_importance
    }

    # === 遍历采样率 (Ticks) ===
    for tick in target_ticks:
        budget_n = int(math.floor(tick * total_instances))
        sampler.total_budget_frac = tick  # 动态更新预算

        # === 遍历方法 ===
        for method_name, run_func in methods_map.items():
            # === 重复运行 n 次 ===
            for r in range(run_times):
                run_id = r + 1
                try:
                    res = run_func()
                    
                    oracle_cost = res.get("n_post", 0) + res.get("n_comment", 0)
                    
                    record = {
                        "query_basename": query_basename,
                        "run_id": run_id,
                        "budget_frac": tick,
                        "budget_n": budget_n,
                        "T_true": T_true,
                        "T_hat": res["T_hat"],
                        "Qerror": res["Qerror"],
                        "n_post": res.get("n_post", 0),
                        "n_comment": res.get("n_comment", 0),
                        "oracle_cost": oracle_cost,
                        "method": method_name
                    }
                    file_records.append(record)
                except Exception:
                    # 忽略单个错误的运行，避免整个进程崩溃
                    pass
                    
    return file_records

def run_adaptive_sampling_experiment_fast(
    dataset_name: str = "dataset_test",
    run_times: int = 5,
    max_workers: int = None  # 默认使用 CPU 核心数
):
    """
    [多进程加速版] 
    按照指定的 budget_frac 列表，对 run_proxy_importance 和 run_proxyE_importance 
    进行两阶段自适应采样评估。
    """
    # === 1. 配置参数与路径 ===
    TARGET_TICKS = [0.05, 0.1, 0.2, 0.15, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    base_path = f"/home/wangshuo/resource/datasets/parler_data/{dataset_name}"
    aggregated_dir = os.path.join(base_path, "results", "aggregated_results")
    t_true_path = os.path.join(base_path, "results", "T_true_ML1_oracle2_probability_ML2_oracle2_probability.json")
    
    output_dir = os.path.join(base_path, "results", "efficiency")
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "two_stage_adaptive_results_fast.csv")
    
    # 封装配置字典传给子进程
    config = {
        "POST_PROXY": "ML1_proxy4b_probability",
        "COMMENT_PROXY": "ML2_proxy1_probability",
        "POST_ORACLE": "ML1_oracle2_probability",
        "COMMENT_ORACLE": "ML2_oracle2_probability"
    }

    print(f"\n{'='*10} 开始两阶段自适应采样评估 (多进程加速版) {'='*10}")
    
    # === 2. 加载 T_true ===
    if not os.path.exists(t_true_path):
        print(f"[Error] T_true 文件未找到: {t_true_path}")
        return
    with open(t_true_path, 'r') as f:
        all_t_true = json.load(f)

    # === 3. 准备文件列表 ===
    if not os.path.exists(aggregated_dir):
        print(f"[Error] 聚合目录不存在: {aggregated_dir}")
        return
    agg_files = sorted([f for f in os.listdir(aggregated_dir) if f.endswith(".csv")])
    print(f"待处理文件数: {len(agg_files)}")

    # === 4. 初始化输出文件 ===
    headers = ["query_basename", "run_id", "budget_frac", "budget_n", "T_true", "T_hat", "Qerror", "n_post", "n_comment", "oracle_cost", "method"]
    # 覆盖模式（如果不想覆盖，改为 'a' 并添加 header 判断逻辑）
    pd.DataFrame(columns=headers).to_csv(output_csv, index=False)

    # === 5. 多进程并行执行 ===
    # max_workers 建议设置为 CPU 核心数 - 2，防止系统卡顿
    if max_workers is None:
        max_workers = max(1, os.cpu_count() - 2)

    print(f"启动进程池: {max_workers} workers")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        futures = []
        for agg_file in agg_files:
            futures.append(
                executor.submit(
                    _process_single_query_file,
                    agg_file, base_path, aggregated_dir, all_t_true, 
                    TARGET_TICKS, run_times, config
                )
            )
        
        # 使用 tqdm 显示进度并收集结果
        # as_completed 会在任务完成时立刻返回，不用等所有都做完
        for future in tqdm(as_completed(futures), total=len(futures), desc="Sampling Progress"):
            try:
                result_records = future.result()
                if result_records:
                    # === 批量写入结果 ===
                    # 每次完成一个文件，就将该文件的所有结果追加写入 CSV
                    df_chunk = pd.DataFrame(result_records)
                    df_chunk.to_csv(output_csv, mode='a', header=False, index=False)
            except Exception as e:
                print(f"Worker Error: {e}")

    print(f"\n[Done] 加速评估完成。结果已保存至: {output_csv}")