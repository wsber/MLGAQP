import math
import numpy as np
import pandas as pd
from typing import Tuple

class PostStratifiedSampler:
    def __init__(self, K: int = 5, N1: int = 5, total_budget_frac: float = 0.15, random_state: int = 42):
        """
        K: number of strata
        N1: pilot sample size per stratum (minimum)
        total_budget_frac: fraction of total posts to use as total oracle budget (e.g. 0.15)
        """
        self.K = K
        self.N1 = N1
        self.total_budget_frac = total_budget_frac
        np.random.seed(random_state)

    def prepare_posts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate rows into posts:
        - w = number of rows per post (occurrence count)
        - a = sum(comment_upvotes) for that post (post's total contribution)
        - proxy = post_proxy4b1 (take first)
        - oracle = (post_oracle1 > 0.5) (take first)
        """
        # Ensure numeric
        df = df.copy()
        df["comment_upvotes"] = pd.to_numeric(df["comment_upvotes"], errors="coerce").fillna(0).astype(float)
        # Group by postId
        g = df.groupby("postId", sort=False)
        posts = pd.DataFrame({
            "postId": list(g.groups.keys()),
            "w": g.size().values,  # count of rows per post
            "a": g["comment_upvotes"].sum().values,  # total upvotes per post
            "proxy": g["post_proxy4b1"].first().values,
            "oracle_val": g["post_oracle1"].first().values
        })
        posts["oracle"] = (posts["oracle_val"] > 0.5).astype(int)
        # If a post has a==0 and w>0, it's fine (contribution zero)
        return posts

    def stratify(self, posts: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        Assign stratum by quantiles of proxy into K strata (0..K-1)
        Return posts with 'stratum' column and stratum_info dict with N_k
        """
        posts = posts.copy()
        # If proxies have many identical values, use rank-based qcut fallback
        try:
            posts["stratum"] = pd.qcut(posts["proxy"], self.K, labels=False, duplicates="drop")
            # qcut can return NaN if constant proxy; fallback handled below
        except Exception:
            posts["stratum"] = pd.Series(pd.cut(posts["proxy"].rank(method="first"), bins=self.K, labels=False))
        # In case some NaN strata (e.g., constant proxy), assign all to 0
        if posts["stratum"].isnull().any():
            posts["stratum"] = posts["stratum"].fillna(0).astype(int)
        posts["stratum"] = posts["stratum"].astype(int)
        # gather stratum info
        stratum_info = {}
        for k in range(posts["stratum"].nunique()):
            Nk = int((posts["stratum"] == k).sum())
            stratum_info[k] = {"N_k": Nk}
        return posts, stratum_info

    def pilot(self, posts: pd.DataFrame) -> Tuple[dict, dict]:
        """
        Pilot sampling: sample up to N1 per stratum.
        Estimate per-stratum:
          - p_hat (weighted by a): sum_{sample, oracle=1} a / sum_{sample} a
          - sigma_hat: sample std of Y = a * oracle
          - W_k: sum of a in stratum (total weight)
        Return stats dict and pilot_samples (for merging later)
        """
        stats = {}
        pilot_samples = {}
        for k, grp in posts.groupby("stratum"):
            Nk = len(grp)
            n1 = min(self.N1, Nk)
            if n1 == 0:
                stats[k] = {"W_k": 0.0, "p_hat": 0.0, "sigma_hat": 0.0, "N_k": 0}
                pilot_samples[k] = pd.DataFrame(columns=posts.columns)
                continue
            sample = grp.sample(n1, replace=False)
            # Y = a * oracle
            sample = sample.copy()
            sample["Y"] = sample["a"] * sample["oracle"]
            W_k_sample = sample["a"].sum()
            W_pos = sample.loc[sample["oracle"] == 1, "a"].sum()
            p_hat = (W_pos / W_k_sample) if W_k_sample > 0 else 0.0
            sigma_hat = sample["Y"].std(ddof=1) if len(sample) > 1 else 0.0
            W_k = grp["a"].sum()  # total a in the stratum
            stats[k] = {"W_k": float(W_k), "p_hat": float(p_hat), "sigma_hat": float(sigma_hat), "N_k": Nk, "n1": n1}
            pilot_samples[k] = sample
        return stats, pilot_samples

    def allocate(self, stats: dict, total_posts: int) -> dict:
        """
        Compute total budget N_total = floor(total_budget_frac * total_posts)
        Deduct pilot samples used, remaining N2 distributed using:
            weight_k = sqrt(W_k * p_hat * sigma_hat)
        If all weights zero, allocate proportional to N_k.
        Return allocation per stratum (additional samples beyond pilot).
        """
        N_total = max(1, int(math.floor(self.total_budget_frac * total_posts)))
        # compute pilot used
        pilot_used = sum(st.get("n1", 0) for st in stats.values())
        remaining = max(0, N_total - pilot_used)
        if remaining == 0:
            return {k: 0 for k in stats.keys()}, N_total

        weights = {}
        for k, st in stats.items():
            w = math.sqrt(max(0.0, st["W_k"]) * max(0.0, st["p_hat"]) * max(0.0, st["sigma_hat"]))
            weights[k] = w

        total_weight = sum(weights.values())
        alloc = {}
        if total_weight <= 0.0:
            # fallback: proportional to N_k
            Nk_total = sum(st["N_k"] for st in stats.values())
            for k, st in stats.items():
                alloc[k] = int(round(remaining * st["N_k"] / Nk_total)) if Nk_total > 0 else 0
        else:
            # continuous allocation then integerize by largest remainder
            cont = {k: (remaining * weights[k] / total_weight) for k in stats}
            floored = {k: int(math.floor(v)) for k, v in cont.items()}
            assigned = sum(floored.values())
            rem = remaining - assigned
            # assign remaining by largest fractional parts
            frac = sorted(((k, cont[k] - floored[k]) for k in cont), key=lambda x: x[1], reverse=True)
            alloc = floored.copy()
            idx = 0
            while rem > 0 and idx < len(frac):
                alloc[frac[idx][0]] += 1
                rem -= 1
                idx += 1
            # if still rem due to rounding edge, distribute arbitrarily
            ks = list(cont.keys())
            i = 0
            while rem > 0:
                alloc[ks[i % len(ks)]] += 1
                rem -= 1
                i += 1

        # ensure we don't allocate more than available posts in a stratum minus pilot
        for k, st in stats.items():
            max_add = max(0, st["N_k"] - st.get("n1", 0))
            if alloc.get(k, 0) > max_add:
                alloc[k] = max_add
        return alloc, N_total

    def allocate_smooth(self, stats: dict, total_posts: int, alpha: float = 1.0, beta: float = 1.0, min_per_stratum: int = 1) -> tuple:
        """
        Compute allocation with two enhancements:
        1) Smoothed p_hat (Laplace smoothing) to avoid zero estimates.
        2) Minimum sample guarantee per stratum.

        Args:
            stats: dict from pilot() with W_k, p_hat, sigma_hat, N_k, n1
            total_posts: total number of posts
            alpha, beta: smoothing parameters for p_hat (default=1 Laplace smoothing)
            min_per_stratum: minimum number of samples per stratum (default=1)

        Returns:
            alloc: dict {k: n_k_additional}
            N_total: total budget
        """
        # 总预算（按比例取总post数）
        N_total = max(1, int(math.floor(self.total_budget_frac * total_posts)))
        pilot_used = sum(st.get("n1", 0) for st in stats.values())
        remaining = max(0, N_total - pilot_used)

        if remaining == 0:
            return {k: 0 for k in stats.keys()}, N_total

        weights = {}
        for k, st in stats.items():
            # 平滑 p_hat
            W_k_sample = st.get("n1", 0)
            if W_k_sample > 0:
                p_hat_smooth = (st["p_hat"] * W_k_sample + alpha) / (W_k_sample + alpha + beta)
            else:
                p_hat_smooth = alpha / (alpha + beta)

            w = math.sqrt(max(0.0, st["W_k"]) * p_hat_smooth * max(0.0, st["sigma_hat"]))
            weights[k] = w

        total_weight = sum(weights.values())
        alloc = {}

        if total_weight <= 0.0:
            # fallback: proportional to N_k
            Nk_total = sum(st["N_k"] for st in stats.values())
            for k, st in stats.items():
                alloc[k] = int(round(remaining * st["N_k"] / Nk_total)) if Nk_total > 0 else 0
        else:
            # continuous allocation then integerize by largest remainder
            cont = {k: (remaining * weights[k] / total_weight) for k in stats}
            floored = {k: int(math.floor(v)) for k, v in cont.items()}
            assigned = sum(floored.values())
            rem = remaining - assigned
            frac = sorted(((k, cont[k] - floored[k]) for k in cont), key=lambda x: x[1], reverse=True)
            alloc = floored.copy()
            idx = 0
            while rem > 0 and idx < len(frac):
                alloc[frac[idx][0]] += 1
                rem -= 1
                idx += 1

        # 最小保障：每层至少分到 min_per_stratum
        for k, st in stats.items():
            if st["N_k"] > 0:  # 层非空
                alloc[k] = max(alloc.get(k, 0), min_per_stratum)

            # 不超过剩余可采post数
            max_add = max(0, st["N_k"] - st.get("n1", 0))
            if alloc[k] > max_add:
                alloc[k] = max_add

        return alloc, N_total
    def second_stage(self, posts: pd.DataFrame, pilot_samples: dict, alloc: dict) -> Tuple[dict, dict]:
        """
        For each stratum, sample alloc[k] additional posts (without replacement among remaining),
        then combine with pilot sample to form the final sample for that stratum.
        Return combined_samples and per-stratum summaries (Ybar, s^2, n_k_total).
        """
        combined = {}
        summaries = {}
        for k, grp in posts.groupby("stratum"):
            pilot = pilot_samples.get(k)
            pilot_ids = set(pilot["postId"].tolist()) if not pilot.empty else set()
            remaining_grp = grp[~grp["postId"].isin(pilot_ids)]
            add_n = alloc.get(k, 0)
            add_sample = pd.DataFrame(columns=posts.columns)
            if add_n > 0 and len(remaining_grp) > 0:
                add_sample = remaining_grp.sample(min(add_n, len(remaining_grp)), replace=False)
            # combine
            final = pd.concat([pilot, add_sample], ignore_index=True, sort=False)
            if final.empty:
                summaries[k] = {"n_k": 0, "N_k": len(grp), "ybar": 0.0, "s2": 0.0}
                combined[k] = final
                continue
            final = final.copy()
            final["Y"] = final["a"] * final["oracle"]
            n_k = len(final)
            ybar = final["Y"].mean()
            s2 = final["Y"].var(ddof=1) if n_k > 1 else 0.0
            combined[k] = final
            summaries[k] = {"n_k": n_k, "N_k": len(grp), "ybar": float(ybar), "s2": float(s2)}
        return combined, summaries

    def estimate_total(self, summaries: dict) -> Tuple[float, float, Tuple[float, float]]:
        """
        Compute T_hat = sum_k N_k * ybar_k
        Var_hat = sum_k N_k^2 * (1 - n_k/N_k) * s2_k / n_k
        Return T_hat, Var_hat, 95% CI
        """
        T_hat = 0.0
        Var_hat = 0.0
        for k, v in summaries.items():
            N_k = v["N_k"]
            n_k = v["n_k"]
            ybar = v["ybar"]
            s2 = v["s2"]
            T_hat += N_k * ybar
            if n_k > 1:
                Var_hat += (N_k**2) * (1.0 - n_k / N_k) * s2 / n_k
        se = math.sqrt(max(0.0, Var_hat))
        ci_low = T_hat - 1.96 * se
        ci_high = T_hat + 1.96 * se
        return float(T_hat), float(Var_hat), (ci_low, ci_high)

    def run(self, df: pd.DataFrame):
        posts = self.prepare_posts(df)
        posts, stratum_info = self.stratify(posts)
        stats, pilot_samples = self.pilot(posts)
        alloc, N_total = self.allocate(stats, total_posts=len(posts))
        combined, summaries = self.second_stage(posts, pilot_samples, alloc)
        T_hat, Var_hat, CI = self.estimate_total(summaries)
        # compute true total from full data (oracle known)
        T_true = float((posts["a"] * posts["oracle"]).sum())
        return {
            "T_hat": T_hat,
            "Var_hat": Var_hat,
            "CI": CI,
            "T_true": T_true,
            "N_total_budget": N_total,
            "pilot_used": sum(st.get("n1", 0) for st in stats.values()),
            "alloc": alloc,
            "summaries": summaries,
            "stats": stats,
            "posts_df": posts
        }


if __name__ == "__main__":
    # replace with your real CSV path
    csv_path = "/home/wangshuo/projects/Neo4j_Exp/pythonProject/output/query_results.csv"
    df = pd.read_csv(csv_path)

    sampler = PostStratifiedSampler(K=5, N1=200, total_budget_frac=0.1, random_state=2025)
    out = sampler.run(df)

    print("Total budget (posts):", out["N_total_budget"])
    print("Pilot used:", out["pilot_used"])
    print("Allocation per stratum (additional samples):", out["alloc"])
    print("Estimated T_hat:", out["T_hat"])
    print("Variance estimate Var_hat:", out["Var_hat"])
    print("95% CI:", out["CI"])
    print("True total (using full oracle):", out["T_true"])
