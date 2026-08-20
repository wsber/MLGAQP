import os
import json
import numpy as np
import pandas as pd

class ResultAverager:
    def __init__(self, parent_data, dataset_name, table1_oracle, table2_oracle):
        """
        初始化路径与参数配置
        """
        self.parent_data = parent_data
        self.dataset_name = dataset_name
        self.table1_oracle = table1_oracle
        self.table2_oracle = table2_oracle
        
        self.base_eff_dir = f"/home/wangshuo/resource/datasets/{parent_data}/{dataset_name}/results/efficiency"
        self.base_res_dir = f"/home/wangshuo/resource/datasets/{parent_data}/{dataset_name}/results"
        
        self.eps = 1e-12

    def process_csv(self, file_prefix, default_method=None):
        """
        通用 CSV 平均值计算函数 (sum / count -> avg)
        完全兼容 Amazon/Parler 列名风格，并处理可能的列缺失问题
        """
        sum_csv_path = os.path.join(self.base_eff_dir, f"{file_prefix}_sum.csv")
        count_csv_path = os.path.join(self.base_eff_dir, f"{file_prefix}_count.csv")
        avg_csv_path = os.path.join(self.base_eff_dir, f"{file_prefix}_avg.csv")

        if not os.path.exists(sum_csv_path):
            print(f"[Warning] sum CSV 不存在: {sum_csv_path}，跳过此项。")
            return None
        if not os.path.exists(count_csv_path):
            print(f"[Warning] count CSV 不存在: {count_csv_path}，跳过此项。")
            return None

        print(f"\n[Processing CSV] 开始计算 {file_prefix} 的 AVG 结果...")
        df_sum = pd.read_csv(sum_csv_path)
        df_count = pd.read_csv(count_csv_path)

        # 1. 动态检测数据集风格列名 (Amazon 还是 Parler)
        is_amazon_style = (
            "n_product" in df_sum.columns or 
            "n_product" in df_count.columns or 
            "n_review" in df_sum.columns or 
            "n_review" in df_count.columns or
            self.parent_data == "amazon_data"
        )

        if is_amazon_style:
            col_v1, col_v2 = "n_product", "n_review"
            print(f"  -> 检测到 Amazon 风格，使用列名: {col_v1}, {col_v2}")
        else:
            col_v1, col_v2 = "n_post", "n_comment"
            print(f"  -> 检测到 Parler 风格，使用列名: {col_v1}, {col_v2}")

        # 2. 补齐缺失的 method 列
        for df in [df_sum, df_count]:
            if "method" not in df.columns:
                df["method"] = default_method if default_method is not None else "Unknown"

        # 3. 补齐可选列，避免报错
        optional_cols = {
            "budget_n": 0,
            "Qerror": np.nan,
            col_v1: 0,
            col_v2: 0,
            "oracle_cost": 0
        }
        
        for df in [df_sum, df_count]:
            for col, default_val in optional_cols.items():
                if col not in df.columns:
                    df[col] = default_val

        # 4. 关键列校验
        key_cols = ["query_basename", "budget_frac", "run_id", "method"]
        essential_numeric_cols = ["T_true", "T_hat"]
        
        for c in key_cols + essential_numeric_cols:
            if c not in df_count.columns:
                raise ValueError(f"count 文件缺少基础必需列: {c}")
            if c not in df_sum.columns:
                raise ValueError(f"sum 文件缺少基础必需列: {c}")

        # 5. DataFrame merge
        use_cols = key_cols + ["budget_n", "T_true", "T_hat", "Qerror", col_v1, col_v2, "oracle_cost"]
        cdf = df_count[use_cols].copy()
        sdf = df_sum[use_cols].copy()

        merged = pd.merge(
            sdf, cdf,
            on=key_cols,
            how="inner",
            suffixes=("_sum", "_count")
        )

        if merged.empty:
            print(f"[Warning] merge 后交集为空! {os.path.basename(sum_csv_path)} 与 count 无共同查询。")
            return pd.DataFrame()

        # 6. 计算均值及 Qerror (ARE)
        denom_hat = merged["T_hat_count"].astype(float).replace(0.0, np.nan)
        denom_true = merged["T_true_count"].astype(float).replace(0.0, np.nan)

        merged["T_hat_avg"] = merged["T_hat_sum"].astype(float) / denom_hat
        merged["T_true_avg"] = merged["T_true_sum"].astype(float) / denom_true

        denom_q = merged["T_true_avg"].abs().replace(0.0, np.nan)
        merged["Qerror_avg"] = (merged["T_hat_avg"] - merged["T_true_avg"]).abs() / denom_q

        # 7. 构建输出
        out_df = pd.DataFrame({
            "query_basename": merged["query_basename"],
            "run_id": merged["run_id"],
            "budget_frac": merged["budget_frac"],
            "budget_n": merged["budget_n_count"],
            "T_true": merged["T_true_avg"],
            "T_hat": merged["T_hat_avg"],
            "Qerror": merged["Qerror_avg"],
            col_v1: merged[f"{col_v1}_count"],
            col_v2: merged[f"{col_v2}_count"],
            "oracle_cost": merged["oracle_cost_count"],
            "method": merged["method"]
        })

        out_df = out_df.sort_values(["query_basename", "budget_frac", "run_id", "method"]).reset_index(drop=True)
        out_df.to_csv(avg_csv_path, index=False)
        
        print(f"  -> [DONE] 生成 CSV: {avg_csv_path} (行数: {len(out_df)})")
        return out_df

    def process_json(self):
        """
        处理 T_true 的 JSON 文件 (sum / count -> avg)
        兼容 count/cout 拼写错误情况
        """
        print("\n[Processing JSON] 开始生成 Ground Truth JSON...")
        base_name = f"T_true_{self.table1_oracle}_{self.table2_oracle}"
        
        json_sum_path = os.path.join(self.base_res_dir, f"{base_name}_sum.json")
        json_avg_path = os.path.join(self.base_res_dir, f"{base_name}_avg.json")
        
        json_count_candidate_1 = os.path.join(self.base_res_dir, f"{base_name}_count.json")
        json_count_candidate_2 = os.path.join(self.base_res_dir, f"{base_name}_cout.json") # 兼容拼写错误

        if os.path.exists(json_count_candidate_1):
            json_count_path = json_count_candidate_1
        elif os.path.exists(json_count_candidate_2):
            json_count_path = json_count_candidate_2
        else:
            raise FileNotFoundError(
                f"count/cout 真值文件都没找到:\n- {json_count_candidate_1}\n- {json_count_candidate_2}"
            )

        with open(json_sum_path, "r") as f:
            t_sum = json.load(f)
        with open(json_count_path, "r") as f:
            t_count = json.load(f)

        common_keys = sorted(set(t_sum.keys()) & set(t_count.keys()))
        t_avg = {}

        for k in common_keys:
            denom = float(t_count[k])
            if abs(denom) < self.eps:
                t_avg[k] = None
            else:
                t_avg[k] = float(t_sum[k]) / denom

        with open(json_avg_path, "w") as f:
            json.dump(t_avg, f, indent=4, ensure_ascii=False)

        print(f"  -> [DONE] 生成 JSON: {json_avg_path} (键总数: {len(t_avg)})")

    def run_all(self):
        """
        一键运行所有需处理的任务，等价于原先的三个脚本叠加
        """
        # 1. 基础对照: allocation_strategy_comparison
        self.process_csv("allocation_strategy_comparison", default_method=None)
        
        # 2. FastestO 曲线
        self.process_csv("FastestO_budget_curve", default_method="FastestO")
        
        # 3. Exact_structureO 曲线
        self.process_csv("Exact_structureO_budget_curve", default_method="Exact_structureO")
        
        # 4. JSON 真值均值
        self.process_json()


# =========================
# 执行入口（按需修改此处配置即可）
# =========================
if __name__ == "__main__":
    
    # 示例: Parler 数据集配置
    processor_parler = ResultAverager(
        parent_data="parler_data",
        dataset_name="dataset_test",
        table1_oracle="ML1_oracle2_probability",
        table2_oracle="ML2_oracle2_probability"
    )
    
    # 运行所有任务
    processor_parler.run_all()
    
    # ---------------------------------------------------------
    # 如果之后需要跑 Amazon 的数据，只需这样调用：
    # ---------------------------------------------------------
    # processor_amazon = ResultAverager(
    #     parent_data="amazon_data",
    #     dataset_name="amazon_extend",
    #     table1_oracle="ML3_oracle2_probability",
    #     table2_oracle="ML2_oracle1_probability"
    # )
    # processor_amazon.run_all()