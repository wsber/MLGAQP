# ablation_proxy_vs_w.py
import math
import numpy as np
import pandas as pd
from typing import Tuple, Dict
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

np.random.seed(47005)

# ----------------------------
# Helper: prepare posts from raw rows
# ----------------------------
def prepare_posts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # ensure numeric
    df["comment_upvotes"] = pd.to_numeric(df["comment_upvotes"], errors="coerce").fillna(0).astype(float)
    g = df.groupby("postId", sort=False)
    posts = pd.DataFrame({
        "postId": list(g.groups.keys()),
        # w: number of matches (occurrence count)
        "w": g.size().values.astype(float),
        # a: total contribution (sum of comment_upvotes)
        "a": g["comment_upvotes"].sum().values.astype(float),
        # proxy and oracle take first (assume same per post)
        "proxy": g["post_proxy4b1"].first().values.astype(float),
        "oracle_val": g["post_oracle1"].first().values.astype(float)
    })
    posts["oracle"] = (posts["oracle_val"] > 0.5).astype(int)
    return posts

# ----------------------------
# Base stratified sampler utilities
# ----------------------------
def stratify_by_proxy(posts: pd.DataFrame, K: int) -> pd.DataFrame:
    posts = posts.copy()
    # try qcut, fallback to rank-based
    try:
        posts["stratum"] = pd.qcut(posts["proxy"], K, labels=False, duplicates="drop")
    except Exception:
        posts["stratum"] = pd.cut(posts["proxy"].rank(method="first"), bins=K, labels=False)
    if posts["stratum"].isnull().any():
        posts["stratum"] = posts["stratum"].fillna(0)
    posts["stratum"] = posts["stratum"].astype(int)
    return posts


def stratify_by_clustering(posts: pd.DataFrame, K: int, use_features=("proxy", "a")) -> pd.DataFrame:
    """
    用 KMeans 在指定特征上聚类。默认使用 ('proxy','a')（对 a 做 log1p 并标准化）。
    备注：返回的 posts 带有整型 'stratum' 字段。
    """
    posts = posts.copy()
    Xcols = []
    X = []
    if "proxy" in use_features:
        Xcols.append("proxy")
        X.append(posts["proxy"].values.reshape(-1,1))
    if "a" in use_features:
        Xcols.append("log1p_a")
        X.append(np.log1p(posts["a"].values).reshape(-1,1))
    if "w" in use_features:
        Xcols.append("log1p_w")
        X.append(np.log1p(posts["w"].values).reshape(-1,1))

    if len(X) == 0:
        raise ValueError("No features selected for clustering")

    X = np.hstack(X)
    # 标准化（重要）
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=K, random_state=42, n_init=10)
    posts["stratum"] = km.fit_predict(Xs).astype(int)
    return posts

def stratify_by_expected_contrib(posts: pd.DataFrame, K: int) -> pd.DataFrame:
    """
    简单且语义清晰的做法：按 proxy * a（期望贡献）做分位切分（qcut）。
    适合 SUM 估计，且解释性好。
    """
    posts = posts.copy()
    # expected contribution
    posts["exp_contrib"] = posts["proxy"] * posts["a"]
    # 若 exp_contrib 全部为 0 或重复较多，qcut 可能抛错 -> fallback to rank-based cut
    try:
        posts["stratum"] = pd.qcut(posts["exp_contrib"], K, labels=False, duplicates="drop")
    except Exception:
        # use rank-based cut
        posts["stratum"] = pd.cut(posts["exp_contrib"].rank(method="first"), bins=K, labels=False)
    posts["stratum"] = posts["stratum"].fillna(0).astype(int)
    return posts

# ----------------------------
# Pilot allocation helper (把 pilot 总预算 N1_total 分配到每个层)
# ----------------------------
def allocate_pilot_budget_per_stratum(stats: Dict[int, dict], N1_total: int, min_per_stratum: int = 1) -> Dict[int, int]:
    """
    将 pilot 总预算 N1_total 按层大小 N_k 比例分配，结果为每层 n1_k（整数）。
    策略：
      - 首先按 N_k / N_total 计算连续比例 cont_k
      - 取 floor(cont_k) 作为初始分配
      - 将剩余名额按小数部分从大到小分配
      - 保证每层至少 min_per_stratum（如果可能）且不超过该层容量 N_k
    """
    if N1_total <= 0:
        return {k: 0 for k in stats}
    Nks = {k: st["N_k"] for k, st in stats.items()}
    total_N = sum(Nks.values())
    if total_N == 0:
        return {k: 0 for k in stats}
    # preliminary continuous allocation
    cont = {k: (N1_total * Nks[k] / total_N) for k in Nks}
    floored = {k: int(math.floor(v)) for k, v in cont.items()}
    assigned = sum(floored.values())
    rem = N1_total - assigned
    # fractional parts sorted descending
    fracs = sorted(((k, cont[k] - floored[k]) for k in cont), key=lambda x: x[1], reverse=True)
    alloc = floored.copy()
    idx = 0
    while rem > 0 and idx < len(fracs):
        k = fracs[idx][0]
        alloc[k] += 1
        rem -= 1
        idx += 1
    # ensure min_per_stratum where possible
    for k in alloc:
        if Nks[k] == 0:
            alloc[k] = 0
        else:
            alloc[k] = max(0, min(alloc[k], Nks[k]))  # cap at N_k
    # If some strata got 0 but we want min_per_stratum, try to enforce (if budget allows)
    if min_per_stratum > 0:
        want = {k: max(min_per_stratum, alloc[k]) if Nks[k] > 0 else 0 for k in alloc}
        sum_want = sum(want.values())
        if sum_want <= N1_total:
            alloc = want
        else:
            # cannot satisfy min for all; keep current alloc (already sum==N1_total)
            pass
    return alloc

# ----------------------------
# Pilot stats using per-stratum pilot allocation
# ----------------------------
def pilot_stats_with_alloc(posts: pd.DataFrame, pilot_alloc: Dict[int,int]) -> Tuple[Dict[int, dict], Dict[int, pd.DataFrame]]:
    stats = {}
    pilots = {}
    for k, grp in posts.groupby("stratum"):
        Nk = len(grp)
        n1 = int(min(pilot_alloc.get(k, 0), Nk))
        if n1 <= 0:
            stats[k] = {"W_k": float(grp["a"].sum()), "p_hat": 0.0, "sigma_hat": 0.0, "N_k": Nk, "n1": 0}
            pilots[k] = pd.DataFrame(columns=posts.columns)
            continue
        sample = grp.sample(n1, replace=False, random_state=np.random.randint(1<<30))
        sample = sample.copy()
        sample["Y"] = sample["a"] * sample["oracle"]  # observed contribution
        W_k_sample = sample["a"].sum()
        W_pos = sample.loc[sample["oracle"] == 1, "a"].sum()
        p_hat = (W_pos / W_k_sample) if W_k_sample > 0 else (sample["oracle"].mean() if len(sample)>0 else 0.0)
        sigma_hat = sample["Y"].std(ddof=1) if len(sample) > 1 else 0.0
        W_k = grp["a"].sum()
        stats[k] = {"W_k": float(W_k), "p_hat": float(p_hat), "sigma_hat": float(sigma_hat), "N_k": Nk, "n1": n1}
        pilots[k] = sample
    return stats, pilots

# ----------------------------
# Allocation functions but scaled to a given second-stage budget N2
# ----------------------------

def allocate_proxy_only_given_N2(stats: Dict[int, dict], N2: int,
                                 alpha: float=1.0, beta: float=1.0, min_per_stratum: int=1) -> Dict[int,int]:
    """
    Allocation without W_k: weight_k = sqrt(p_tilde * sigma_hat), scaled to N2.
    """
    if N2 <= 0:
        return {k: 0 for k in stats}
    weights = {}
    for k, st in stats.items():
        n1 = st.get("n1", 0)
        if n1 > 0:
            p_tilde = (st["p_hat"] * n1 + alpha) / (n1 + alpha + beta)
        else:
            p_tilde = alpha / (alpha + beta)
        val = math.sqrt(max(1e-12, p_tilde) * max(1e-12, st["sigma_hat"]))
        weights[k] = val
    total_w = sum(weights.values())
    alloc = {}
    if total_w <= 0:
        Nk_total = sum(st["N_k"] for st in stats.values())
        for k, st in stats.items():
            alloc[k] = int(round(N2 * st["N_k"] / Nk_total)) if Nk_total>0 else 0
    else:
        cont = {k: N2 * weights[k] / total_w for k in stats}
        floored = {k: int(math.floor(v)) for k, v in cont.items()}
        assigned = sum(floored.values())
        rem = N2 - assigned
        frac = sorted(((k, cont[k]-floored[k]) for k in cont), key=lambda x: x[1], reverse=True)
        alloc = floored.copy()
        idx = 0
        while rem>0 and idx<len(frac):
            alloc[frac[idx][0]] += 1
            rem -= 1
            idx += 1
        ks = list(cont.keys())
        i = 0
        while rem>0:
            alloc[ks[i%len(ks)]] += 1
            rem -= 1
            i += 1
    for k, st in stats.items():
        if st["N_k"] > 0:
            alloc[k] = max(alloc.get(k,0), min_per_stratum)
        max_add = max(0, st["N_k"] - st.get("n1",0))
        if alloc.get(k,0) > max_add:
            alloc[k] = max_add
    return alloc

# ----------------------------
# Second stage + estimation (same for both methods)
# ----------------------------
def second_stage_and_estimate(posts: pd.DataFrame, pilots: Dict[int,pd.DataFrame], alloc: Dict[int,int]) -> Tuple[Dict,int, Dict]:
    """
    For each stratum, sample alloc[k] additional posts (without replacement among remaining),
    combine with pilot, compute Y=a*oracle for final sample, compute per-stratum summaries,
    then compute T_hat and Var_hat (stratified total estimator).
    Return (summaries, total_pulled_count, combined_samples_dict)
    """
    combined = {}
    summaries = {}
    total_pulled = 0
    for k, grp in posts.groupby("stratum"):
        pilot = pilots.get(k, pd.DataFrame(columns=posts.columns))
        pilot_ids = set(pilot["postId"].tolist()) if not pilot.empty else set()
        remaining_grp = grp[~grp["postId"].isin(pilot_ids)]
        add_n = alloc.get(k, 0)
        add_sample = pd.DataFrame(columns=posts.columns)
        if add_n > 0 and len(remaining_grp)>0:
            add_sample = remaining_grp.sample(min(add_n, len(remaining_grp)), replace=False, random_state=np.random.randint(1<<30))
        final = pd.concat([pilot, add_sample], ignore_index=True, sort=False)
        final = final.copy()
        if final.empty:
            summaries[k] = {"n_k":0, "N_k": len(grp), "ybar":0.0, "s2":0.0}
            combined[k] = final
            continue
        final["Y"] = final["a"] * final["oracle"]
        n_k = len(final)
        total_pulled += n_k
        ybar = float(final["Y"].mean())
        s2 = float(final["Y"].var(ddof=1)) if n_k>1 else 0.0
        combined[k] = final
        summaries[k] = {"n_k": n_k, "N_k": len(grp), "ybar": ybar, "s2": s2}
    # compute T_hat and Var_hat
    T_hat = 0.0
    Var_hat = 0.0
    for k, v in summaries.items():
        N_k = v["N_k"]
        n_k = v["n_k"]
        ybar = v["ybar"]
        s2 = v["s2"]
        T_hat += N_k * ybar
        if n_k > 1 and N_k>0:
            Var_hat += (N_k**2) * (1.0 - n_k / N_k) * s2 / n_k
    se = math.sqrt(max(0.0, Var_hat))
    ci_low = T_hat - 1.96 * se
    ci_high = T_hat + 1.96 * se
    return summaries, total_pulled, {"T_hat": float(T_hat), "Var_hat": float(Var_hat), "CI": (ci_low, ci_high)}

# ----------------------------
# Driver: run both methods and compare (现在支持 c_stage 控制)
# ----------------------------
def run_ablation(csv_path: str, stratify_mode="proxy",
                 K: int=5,
                 c_stage: float=0.4, total_budget_frac:float=0.15, min_pilot_per_stratum:int=1):
    """
    c_stage: pilot stage 占总预算的比例 (0..1). 例如 c_stage=0.4 表示 pilot 使用总预算的 40%（整数计数）。
    total_budget_frac: 总预算占总 posts 的比例（两个阶段之和）。
    min_pilot_per_stratum: pilot 分配时对有容量层的最小保障（若预算充足）。
    """
    df = pd.read_csv(csv_path)
    posts = prepare_posts(df)
    print(stratify_mode)
    # Stratify
    if stratify_mode == "proxy":
        posts = stratify_by_proxy(posts, K=K)
    elif stratify_mode == "clustering":
        # posts = stratify_by_clustering(posts, K=K, use_features=("proxy", "a"))
        posts = stratify_by_expected_contrib(posts, K=K)
    else:
        raise ValueError("Unknown stratify_mode")

    # compute overall budgets
    N_total = int(math.floor(total_budget_frac * len(posts)))
    N1_total = int(math.floor(c_stage * N_total))
    N2 = max(0, N_total - N1_total)

    # Safety: ensure budgets are within total posts
    N_total = min(N_total, len(posts))
    N1_total = min(N1_total, N_total)
    N2 = N_total - N1_total

    # ----------------------------
    # Pilot: 将 N1_total 按层分配
    # ----------------------------
    # need initial stats only for N_k (we can compute N_k per stratum from posts)
    stats_init = {}
    for k, grp in posts.groupby("stratum"):
        stats_init[k] = {"N_k": len(grp), "W_k": float(grp["a"].sum())}

    pilot_alloc = allocate_pilot_budget_per_stratum(stats_init, N1_total, min_per_stratum=min_pilot_per_stratum)
    # print pilot allocation
    print(f"Total posts: {len(posts)}, Total budget: {N_total} (pilot {N1_total} + second {N2})")
    print("Pilot allocation per stratum:", pilot_alloc)

    # draw pilot samples according to pilot_alloc
    stats, pilots = pilot_stats_with_alloc(posts, pilot_alloc)

    # ----------------------------
    # Second-stage: 使用 stats（含 pilot 信息）计算 allocation，且把第二阶段预算限制为 N2
    # ----------------------------
    # Method B: Proxy-only allocation using N2
    alloc_p = allocate_proxy_only_given_N2(stats, N2, alpha=1.0, beta=1.0, min_per_stratum=1)

    # Run second stage and estimation
    summaries_p, total_pulled_p, res_p = second_stage_and_estimate(posts, pilots, alloc_p)

    T_true = float((posts["a"] * posts["oracle"]).sum())
    re_p = abs(res_p["T_hat"] - T_true) / (T_true if T_true != 0 else 1.0)

    # Print comparison
    print("\n-- Method  (Proxy-only) --")
    print("Second-stage alloc:", alloc_p)
    print("Pulled total (pilot + second):", total_pulled_p)
    print("T_hat:", res_p["T_hat"], "Var_hat:", res_p["Var_hat"], "95%CI:", res_p["CI"])
    print("Relative error:", re_p)
    print("\n-- Ground truth --")
    print("True total T:", T_true)

    return {
        "posts": posts,
        "stats": stats,
        "pilots": pilots,
        "pilot_alloc": pilot_alloc,
        "proxy_method": {"alloc": alloc_p, "result": res_p, "summaries": summaries_p},
        "T_true": T_true,
        "N_total": N_total,
        "N1_total": N1_total,
        "N2": N2
    }


def compare_strat_methods(csv_path: str, Ks=range(2, 11),
                          c_stage: float=0.4, total_budget_frac: float=0.15):
    """
    比较 proxy-stratify 和 clustering-stratify 在不同 K 下的 proxy-only 误差表现。
    返回 DataFrame，并绘制折线图。
    """
    results = []
    for K in Ks:
        for mode in ["proxy", "clustering"]:
            out = run_ablation(
                csv_path, stratify_mode=mode, K=K,
                c_stage=c_stage, total_budget_frac=total_budget_frac,
                min_pilot_per_stratum=1
            )
            T_true = out["T_true"]
            T_hat = out["proxy_method"]["result"]["T_hat"]
            re = abs(T_hat - T_true) / (T_true if T_true != 0 else 1.0)
            results.append({"K": K, "mode": mode, "RelativeError": re})

    df = pd.DataFrame(results)

    # ---- 绘制折线图 ----
    plt.figure(figsize=(8,6))
    for mode, g in df.groupby("mode"):
        plt.plot(g["K"], g["RelativeError"], marker="o", label=mode)
        # 在点上标注 4 位小数
        for x, y in zip(g["K"], g["RelativeError"]):
            plt.text(x, y, f"{y:.4f}", ha="center", va="bottom", fontsize=8)

    plt.xlabel("Number of strata K")
    plt.ylabel("Relative Error")
    plt.title("Proxy-only method: Proxy vs Clustering stratification")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    return df


if __name__ == "__main__":
    csv_path = "/home/wangshuo/projects/Neo4j_Exp/pythonProject/output/query_results.csv"
    df_res = compare_strat_methods(csv_path, Ks=range(5, 11),
                                   c_stage=0.2, total_budget_frac=0.1)
    print(df_res)


