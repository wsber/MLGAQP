import numpy as np
import pandas as pd
from typing import Callable, List, Dict


class StratifiedSampler:
    def __init__(self, K: int, N1: int, N2: int, oracle: Callable, f_func: Callable):
        """
        K: 分层数
        N1: pilot 样本数（每层）
        N2: 第二阶段总样本预算
        oracle: 高成本判定函数 O(x) ∈ {0,1}
        f_func: 聚合函数 f(H)，传入子图结果返回数值
        """
        self.K = K
        self.N1 = N1
        self.N2 = N2
        self.oracle = oracle
        self.f_func = f_func

    def stratify(self, D: pd.DataFrame) -> List[pd.DataFrame]:
        """
        按 proxy score 分位数划分 K 个层
        D: DataFrame with columns ["id", "proxy", "w", "fbar"]
        """
        D = D.copy()
        D["stratum"] = pd.qcut(D["proxy"], self.K, labels=False, duplicates="drop")
        strata = [D[D["stratum"] == k] for k in range(self.K)]
        return strata

    def pilot_sampling(self, strata: List[pd.DataFrame]) -> Dict[int, Dict]:
        """
        阶段1 pilot采样：每层均匀采 N1，估计 p_k, mu_k, sigma_k
        """
        stats = {}
        for k, S in enumerate(strata):
            sample = S.sample(min(self.N1, len(S)), replace=False)
            sample["oracle"] = sample["id"].apply(self.oracle)

            # 估计加权正例率
            W_total = sample["w"].sum()
            W_pos = sample.loc[sample["oracle"] == 1, "w"].sum()
            p_hat = W_pos / W_total if W_total > 0 else 0.0

            # 估计层内均值 mu_hat
            numerator = (sample.loc[sample["oracle"] == 1, "w"] * sample["fbar"]).sum()
            mu_hat = numerator / W_pos if W_pos > 0 else 0.0

            # 估计加权标准差 sigma_hat
            contrib = sample["w"] * sample["fbar"] * sample["oracle"]
            sigma_hat = contrib.std(ddof=1) if len(contrib) > 1 else 0.0

            stats[k] = {
                "W_k": S["w"].sum(),
                "N_k": len(S),
                "p_hat": p_hat,
                "mu_hat": mu_hat,
                "sigma_hat": sigma_hat
            }
        return stats

    def allocate_budget(self, stats: Dict[int, Dict]) -> Dict[int, int]:
        """
        阶段2: Neyman分配
        n_k ∝ sqrt(W_k * p_hat_k * sigma_hat_k)
        """
        weights = []
        for k, st in stats.items():
            val = np.sqrt(st["W_k"] * st["p_hat"] * st["sigma_hat"])
            weights.append(val)
        total = sum(weights)
        alloc = {k: int(self.N2 * weights[k] / total) if total > 0 else 0
                 for k in stats}
        return alloc

    def second_stage_sampling(self, strata: List[pd.DataFrame], alloc: Dict[int, int]) -> Dict[int, Dict]:
        """
        阶段2: 层内均匀采样 + oracle
        """
        results = {}
        for k, S in enumerate(strata):
            n_k = alloc.get(k, 0)
            if n_k == 0 or len(S) == 0:
                continue
            sample = S.sample(min(n_k, len(S)), replace=False)
            sample["oracle"] = sample["id"].apply(self.oracle)

            # 更新 mu_hat
            W_pos = sample.loc[sample["oracle"] == 1, "w"].sum()
            numerator = (sample.loc[sample["oracle"] == 1, "w"] * sample["fbar"]).sum()
            mu_hat = numerator / W_pos if W_pos > 0 else 0.0

            results[k] = {
                "W_k": S["w"].sum(),
                "p_hat": W_pos / sample["w"].sum() if sample["w"].sum() > 0 else 0.0,
                "mu_hat": mu_hat
            }
        return results

    def estimate(self, results: Dict[int, Dict]) -> float:
        """
        Horvitz–Thompson 加权估计
        T_hat = sum_k W_k * p_hat_k * mu_hat_k
        """
        T_hat = sum(st["W_k"] * st["p_hat"] * st["mu_hat"] for st in results.values())
        return T_hat

    def run(self, D: pd.DataFrame):
        # Step 1: stratify
        strata = self.stratify(D)

        # Step 2: pilot
        stats = self.pilot_sampling(strata)

        # Step 3: allocate
        alloc = self.allocate_budget(stats)

        # Step 4: second stage
        results = self.second_stage_sampling(strata, alloc)

        # Step 5: estimation
        T_hat = self.estimate(results)
        return T_hat

def main():
    # 模拟数据：1000个节点
    np.random.seed(42)
    N = 1000
    D = pd.DataFrame({
        "id": range(N),
        "proxy": np.random.rand(N),  # proxy score
        "w": np.random.randint(1, 10, size=N),  # 出现次数
        "fbar": np.random.rand(N) * 5  # 节点平均贡献
    })

    # 模拟oracle（真实判定）
    def oracle(x_id):
        return np.random.binomial(1, 0.3)  # 30% 概率为正例

    # f(H)可以直接包含在fbar中，这里不额外写

    sampler = StratifiedSampler(K=5, N1=10, N2=100, oracle=oracle, f_func=None)
    T_hat = sampler.run(D)
    print("Estimated T =", T_hat)


if __name__ == "__main__":
    main()
