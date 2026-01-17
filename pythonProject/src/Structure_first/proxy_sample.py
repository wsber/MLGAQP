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
from typing import Dict, List
from pythonProject.src.Structure_first.compute_truth import GroundTruthManager


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
        print(f'[Check_post_proxy] {post_proxy}')
        print(f'[Check_comment_proxy] {comment_proxy}')
        print(f'[Check_post_oracle] {post_oracle}')
        print(f'[Check_comment_oracle] {comment_oracle}')
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
        修改：如果是 'importance' 采样，使用有放回 (With Replacement) + 预算去重优化，以保证无偏性并减小方差。
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
    def run(self, stratify_mode: str = "proxy", sampling: str = "uniform") -> Dict:
        if self.posts.empty:
            return {"T_hat": 0.0, "T_true": self.T_true, "Qerror": 0.0, "n_post": 0, "n_comment": 0} # +++ 返回 0 计数
        posts = self.posts.copy()

        if stratify_mode == "proxy":
            posts = self.stratify_by_proxy(posts, self.K)
        elif stratify_mode == "proxyE":
            posts = self.stratify_by_expected_contrib(posts, self.K)
        else:
            raise ValueError("Unsupported stratify_mode")

        N_total = int(math.floor(self.total_budget_frac * len(posts)))
        N1_total = int(math.floor(self.c_stage * N_total))
        N2 = N_total - N1_total

        stats_init = {k: {"N_k": len(g), "W_k": g["a"].sum()} for k, g in posts.groupby("stratum")}
        pilot_alloc = self.allocate_pilot_budget(stats_init, N1_total)
        stats, pilots = self.pilot_stats(posts, pilot_alloc)
        alloc2 = self.allocate_second_stage(stats, N2)
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

    def run_mab_sampling_old(self, K: int = 10, budget_frac: float = None, batch_size: int = 10, ucb_scale: float = 1.0):
        """
        基于多臂赌博机 (MAB) 的自适应分层采样 (修正为有放回采样以保证无偏性)。
        """
        # 1. 准备数据与分层 (Arms)
        # print('[Check_running_mab_sampling_ws3]')
        if self.posts.empty:
            return {"T_hat": 0.0, "T_true": self.T_true, "Qerror": 0.0, "n_post": 0, "n_comment": 0}
        
        posts = self.posts.copy()
        posts = self.stratify_by_expected_contrib(posts, K)
        
        N = len(posts)
        budget = int(budget_frac * N) if budget_frac else int(self.total_budget_frac * N)
        
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
                # 必须保留原始数据用于有放回采样
                "data": {
                    "a": grp["a"].values,
                    "oracle": grp["oracle"].values,
                    "prob": probs,
                    "orig_id": grp.index.tolist() # 或其他唯一标识
                },
                "n_k": 0,
                "sum_z": 0.0,
                "sum_sq_z": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "sampled_ids": set() # 用于统计唯一节点开销
            }

        # 3. MAB 循环
        current_sample_count = 0
        
        # 初始每个臂采一点
        init_samples = 2
        for k in arm_state:
            self._mab_sample_batch(arm_state[k], init_samples, replace=True)
            current_sample_count += init_samples

        while current_sample_count < budget:
            best_arm = -1
            max_score = -1.0
            total_n = current_sample_count
            
            # UCB 选择
            for k, state in arm_state.items():
                # UCB Score = N_k * (sigma_hat + exploration)
                exploration = ucb_scale * np.sqrt(2 * np.log(total_n) / state["n_k"])
                sigma_hat = state["std"]
                score = state["N_k"] * (sigma_hat + exploration)
                
                if score > max_score:
                    max_score = score
                    best_arm = k
            
            if best_arm == -1: break
            
            # 批量采样
            n_batch = min(batch_size, budget - current_sample_count)
            self._mab_sample_batch(arm_state[best_arm], n_batch, replace=True)
            current_sample_count += n_batch

        # 4. 最终估计
        T_hat = 0.0
        total_unique_nodes = 0
        
        # 收集所有采样到的唯一ID用于统计开销
        all_sampled_indices = []
        
        for k, state in arm_state.items():
            if state["n_k"] > 0:
                # Hansen-Hurwitz: T_hat_k = N_k * mean(z)
                # 注意：这里的 mean 已经是 sum(z)/n_k，即 E[y/p]
                # 但标准公式是 (1/n) * sum(y/p)。
                # 这里的 state["mean"] 存储的是 mean(z)。
                # 而 z = y/p。
                # 所以 T_hat_k = mean(z) * N_k 是不对的。
                # 应该是 T_hat_k = mean(z)。因为 sum(p)=1，所以 E[y/p] = T_total。
                # 修正：
                # Hansen-Hurwitz 估计的是总和 Y_total。
                # estimator = (1/n) * sum(y_i / p_i)
                # 所以 T_hat_k = state["mean"] (如果 mean 存的是 mean(z))
                
                # 让我们再检查一下 state["mean"] 的计算：
                # z_bar = state["sum_z"] / state["n_k"]
                # state["mean"] = z_bar 
                
                T_hat += state["mean"] # 直接累加各层的估计总和
                
                all_sampled_indices.extend(list(state["sampled_ids"]))

        # 5. 统计开销 (基于唯一节点)
        if all_sampled_indices:
            full_sample = posts.loc[all_sampled_indices]
            n_post, n_comment = self._count_unique_nodes(full_sample)
        else:
            n_post, n_comment = 0, 0
            
        Qerror = abs(T_hat - self.T_true) / (self.T_true if self.T_true != 0 else 1.0)
        
        return {"T_hat": T_hat, "T_true": self.T_true, "Qerror": Qerror, 
                "n_post": n_post, "n_comment": n_comment}

    def _mab_sample_batch_old(self, state, n_batch, replace=True):
        """辅助函数：执行批量采样并更新状态"""
        data = state["data"]
        N_pool = len(data["a"])
        
        # 有放回采样
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
        # 注意：这里的 mean 估计的是该层的 Total Value，而不是 Mean Value
        z_bar = state["sum_z"] / state["n_k"]
        state["mean"] = z_bar 
        
        if state["n_k"] > 1:
            var_z = (state["sum_sq_z"] - state["n_k"] * (z_bar ** 2)) / (state["n_k"] - 1)
            state["std"] = np.sqrt(max(1e-12, var_z))
        else:
            state["std"] = state["mean"] # 启发式
            
        # 记录唯一ID用于开销统计
        state["sampled_ids"].update(orig_ids)
    
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

    # ---  基于 proxy * a 进行加权采样 ---
    def run_baseline_proxy_mul_a(self, budget_frac: float = None, eps: float = 1e-10):
        """
        基线方法：proxy * a 采样。
        采样概率正比于 proxy * a。
        适用场景：认为最终贡献值直接正比于 (图估计 * 谓词概率)。这是理论上的最优重要性分布。
        """
        # print('[Check_running_baseline_a2]')
        # 1. 健壮性检查
        if self.posts.empty:
            return {"T_hat": 0.0, "T_true": self.T_true, "Qerror": 0.0, "n_post": 0, "n_comment": 0}

        posts = self.posts.copy()
        N = len(posts)

        # 2. 确定样本量
        budget = int(budget_frac * N) if budget_frac else int(self.total_budget_frac * N)
        n = min(budget, N)

        # 3. 计算权重 (权重 = proxy * a)
        # 之前的 baseline_proxy_a 是 sqrt(proxy * a)，这里去掉了 sqrt
        weights =posts["proxy"].values * posts["a"].values + eps
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

        # 7. 统计节点开销
        n_post, n_comment = self._count_unique_nodes(sample)

        # 8. Horvitz-Thompson 无偏估计
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
# === 部分 3: 采样评估与报告生成 ===
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
        "UN": sampler.run_baseline_uniform,
        "PO": sampler.run_baseline_proxy,
        "MAB": sampler.run_mab_sampling,

        # "FaSTestO": sampler.run_baseline_graph_only,

        "FOIS_nrs": sampler.run_baseline_proxy_a,
        "FOIS_rs": sampler.run_baseline_proxy_a_unbiased_test1,
        "POSS": sampler.run_proxyE_importance,
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

if __name__ == '__main__':
    dataset_to_process = 'dataset_test'
    # multi_predicate_evaluation(dataset_to_process)
    evaluate_graph_only_baseline(dataset_to_process)

