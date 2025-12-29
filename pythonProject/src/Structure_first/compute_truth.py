import os
import json
import pandas as pd
import polars as pl
from typing import Dict, List
from tqdm import tqdm

class GroundTruthManager:
    """
    一个用于管理和计算查询真实值 (T_true) 的类。

    它封装了多种计算引擎（Pandas, Polars, DuckDB），支持单谓词和多谓词场景，
    并内置了缓存机制以避免重复计算。
    """

    def __init__(self, dataset_name: str):
        """
        初始化 GroundTruthManager。

        Args:
            dataset_name (str): 数据集的名称，用于构建所有相关文件路径。
        """
        self.dataset_name = dataset_name
        # --- 集中管理所有路径 ---
        self.base_path = f"/home/wangshuo/resource/datasets/parler_data/{dataset_name}"
        self.cache_path = os.path.join(self.base_path, "results", "T_true.json")
        self.gt_dir = os.path.join(self.base_path, "ground_truth", "structure_result")
        self.core_config_path = os.path.join(self.base_path, "data_graph", "core_nodes_config.json")
        self.id_mapping_path = os.path.join(self.base_path, "data_graph", "id_mapping.csv")
        self.post_csv_path = os.path.join(self.base_path, "csv_data", "post.csv")
        self.comment_csv_path = os.path.join(self.base_path, "csv_data", "comment.csv")

    # --- 主入口方法 ---
    def get_all(self) -> Dict[str, float]:
        """
        主函数：检查缓存，如果不存在则计算所有查询的多谓词T_true，并保存到缓存。
        这是与外部交互的主要公共接口。
        """
        if os.path.exists(self.cache_path):
            print(f"✅ 找到 T_true 缓存文件: {self.cache_path}")
            with open(self.cache_path, 'r') as f:
                all_T_true_results = json.load(f)
            print("已从缓存加载 T_true 数据。")
            return all_T_true_results

        print(f"⚠️ 未找到 T_true 缓存文件，开始计算...")
        all_T_true_results = {}
        try:
            with open(self.core_config_path, 'r') as f:
                core_nodes_config = json.load(f)

            source_data = self._load_and_prepare_sources()

            gt_files = [f for f in os.listdir(self.gt_dir) if f.endswith('_matches.csv')]
            if not gt_files:
                print(f"[警告] 在目录 {self.gt_dir} 中没有找到任何 '_matches.csv' 文件。")
                return {}

            for gt_file in sorted(gt_files):
                gt_path = os.path.join(self.gt_dir, gt_file)
                T_true = self._compute_multi_predicate_polars(
                    gt_path=gt_path,
                    core_nodes_config=core_nodes_config,
                    source_data=source_data,
                    prob_threshold=0.5
                )
                query_basename = os.path.basename(gt_path).replace("_matches.csv", "")
                all_T_true_results[query_basename] = T_true

        except FileNotFoundError as e:
            print(f"[严重错误] 计算 T_true 时依赖文件未找到: {e}")
            return {}
        except Exception as e:
            print(f"[严重错误] 计算 T_true 时发生未知异常: {e}")
            return {}

        print(f"\n--- T_true 计算完成，结果汇总 ---")
        print(json.dumps(all_T_true_results, indent=4))

        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'w') as f:
            json.dump(all_T_true_results, f, indent=4)
        print(f"✅ T_true 结果已缓存到: {self.cache_path}")

        return all_T_true_results

    # --- 内部辅助和计算方法 ---
    def _load_and_prepare_sources(self) -> Dict[str, pl.LazyFrame]:
        """惰性加载和预处理源文件。"""
        try:
            idmap_df = pl.scan_csv(self.id_mapping_path)
            post_df = pl.scan_csv(self.post_csv_path)
            comment_df = pl.scan_csv(self.comment_csv_path)
        except Exception as e:
            raise FileNotFoundError(f"无法扫描输入文件: {e}")

        idmap_df = idmap_df.select(
            pl.col("internal_id"), pl.col("orig_id"), pl.col("type").str.to_lowercase().alias("type")
        )
        post_df = post_df.select(
            pl.col("id:ID").alias("orig_id"),
            pl.col("ML1_oracle1_probability").cast(pl.Float64).fill_null(0.0).alias("oracle_prob")
        ).with_columns(pl.lit("post").alias("type"))
        comment_df = comment_df.select(
            pl.col("id:ID").alias("orig_id"),
            pl.col("ML2_oracle2_probability").cast(pl.Float64).fill_null(0.0).alias("oracle_prob")
        ).with_columns(pl.lit("comment").alias("type"))

        oracle_source = pl.concat([post_df, comment_df])
        print("已成功预加载 id_mapping, post, 和 comment 数据。")
        return {"idmap": idmap_df, "oracle_source": oracle_source}

    def _compute_multi_predicate_polars(self, gt_path: str, core_nodes_config: Dict, source_data: Dict,
                                        prob_threshold: float = 0.5) -> float:
        """使用 Polars 计算多谓词场景下的 T_true。"""
        query_basename = os.path.basename(gt_path).replace("_matches.csv", "")
        print(f"\n--- 正在为查询 '{query_basename}' 计算 T_true (多谓词) ---")
        if query_basename not in core_nodes_config: return 0.0
        query_config = core_nodes_config[query_basename]
        query_config_int_labels = {int(k): v for k, v in query_config.items()}
        core_node_cols = [f"u{vid}" for label, vids in query_config_int_labels.items() for vid in vids]
        if not core_node_cols: return 0.0
        try:
            gt_df = pl.scan_csv(gt_path).select(core_node_cols)
        except Exception:
            return 0.0
        idmap = source_data["idmap"]
        oracle_source = source_data["oracle_source"]
        current_df = gt_df.with_row_index(name="row_nr")
        all_oracle_cols = []
        for i, col_name in enumerate(core_node_cols):
            temp_df = current_df.select([pl.col(col_name).alias("internal_id")]).with_row_index(name="row_nr_temp")
            temp_df = temp_df.join(idmap, on="internal_id", how="left")
            temp_df = temp_df.join(oracle_source, on=["orig_id", "type"], how="left")
            oracle_col_name = f"oracle_ok_{i}"
            all_oracle_cols.append(oracle_col_name)
            temp_df = temp_df.with_columns(
                (pl.col("oracle_prob").fill_null(0.0) > prob_threshold).alias(oracle_col_name))
            current_df = current_df.join(
                temp_df.select([pl.col("row_nr_temp").alias("row_nr"), pl.col(oracle_col_name)]), on="row_nr",
                how="left")
        valid_matches = current_df.filter(pl.all_horizontal(
            [pl.col(c) for c in all_oracle_cols])) if all_oracle_cols else current_df
        result = valid_matches.select(pl.len()).collect()
        T_true = result[0, 0] if result is not None and not result.is_empty() else 0.0
        print(f"✅ 计算完成: T_true for '{query_basename}' = {T_true}")
        return float(T_true)

    # --- 保留的单谓词计算方法 ---

    def compute_single_predicate_pandas(self, gt_path: str, gt_match_col: str = "u1",
                                        prob_col: str = "ML1_oracle1_probability",
                                        prob_threshold: float = 0.5) -> float:
        """根据精确子图匹配结果计算 T_true (使用 Pandas)。"""
        gt_df = pd.read_csv(gt_path)
        idmap_df = pd.read_csv(self.id_mapping_path)
        post_df = pd.read_csv(self.post_csv_path)
        post_map = idmap_df[idmap_df["type"].str.lower() == "post"][["internal_id", "orig_id"]].copy()
        if gt_match_col not in gt_df.columns:
            raise ValueError(f"❌ {gt_match_col} 不存在于 structure_match_gt.csv 中的列 {list(gt_df.columns)}")
        gt_df = gt_df[[gt_match_col]].rename(columns={gt_match_col: "internal_id"})
        merged = gt_df.merge(post_map, on="internal_id", how="left")
        post_counts = merged.groupby("orig_id").size().reset_index(name="st_truth")
        post_df = post_df.merge(post_counts, left_on="id:ID", right_on="orig_id", how="left")
        post_df["st_truth"] = post_df["st_truth"].fillna(0)
        post_df["oracle"] = (post_df[prob_col] > prob_threshold).astype(int)
        T_true = (post_df["st_truth"] * post_df["oracle"]).sum()
        print(f"✅ Pandas 计算完成: T_true = {T_true:.3f}")
        return T_true

    def compute_single_predicate_polars(self, gt_path: str, gt_match_col: str = "u1",
                                        prob_col: str = "ML1_oracle1_probability",
                                        prob_threshold: float = 0.5) -> float:
        """使用 Polars 高效计算单谓词 T_true。"""
        try:
            gt_df = pl.scan_csv(gt_path)
            idmap_df = pl.scan_csv(self.id_mapping_path)
            post_df = pl.scan_csv(self.post_csv_path)
        except Exception as e:
            raise FileNotFoundError(f"Polars 无法扫描输入文件: {e}")
        post_map = idmap_df.filter(pl.col("type").str.to_lowercase() == "post").select(["internal_id", "orig_id"])
        gt_df = gt_df.select(pl.col(gt_match_col).alias("internal_id"))
        merged = gt_df.join(post_map, on="internal_id", how="left")
        post_counts = merged.group_by("orig_id").agg(pl.count().alias("st_truth"))
        final_lazy_df = post_df.join(post_counts, left_on="id:ID", right_on="orig_id", how="left").fill_null(0) \
            .with_columns((pl.col(prob_col) > prob_threshold).cast(pl.Int8).alias("oracle")) \
            .with_columns((pl.col("st_truth") * pl.col("oracle")).alias("true_contrib"))
        result = final_lazy_df.select(pl.sum("true_contrib").alias("T_true")).collect()
        T_true_value = result["T_true"][0] if result is not None and len(result) > 0 else 0.0
        print(f"✅ Polars 计算完成: T_true = {T_true_value:.3f}")
        return T_true_value

    def save_all_filtered_matches(self, output_dir_name: str = "filtered_structure_result") -> None:
        """
        计算并保存所有满足多谓词 Oracle 条件的匹配结果到指定目录。
        """
        output_dir = os.path.join(self.base_path, "ground_truth", output_dir_name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"📂 准备将过滤后的结果保存到: {output_dir}")

        try:
            with open(self.core_config_path, 'r') as f:
                core_nodes_config = json.load(f)

            source_data = self._load_and_prepare_sources()

            gt_files = [f for f in os.listdir(self.gt_dir) if f.endswith('_matches.csv')]
            if not gt_files:
                print(f"[警告] 在目录 {self.gt_dir} 中没有找到任何 '_matches.csv' 文件。")
                return

            # --- 修改点 1: 使用 tqdm 包裹文件列表，显示进度条 ---
            # desc: 进度条左侧的描述文字
            # unit: 单位名称
            for gt_file in tqdm(sorted(gt_files), desc="过滤查询进度", unit="file"):
                gt_path = os.path.join(self.gt_dir, gt_file)
                # 输出文件名：例如 query1_matches.csv -> query1_filtered.csv
                output_filename = gt_file.replace("_matches.csv", "_filtered.csv")
                output_path = os.path.join(output_dir, output_filename)
                
                self._save_multi_predicate_matches(
                    gt_path=gt_path,
                    output_path=output_path,
                    core_nodes_config=core_nodes_config,
                    source_data=source_data,
                    prob_threshold=0.5
                )

        except Exception as e:
            print(f"[严重错误] 保存过滤结果时发生异常: {e}")

    def _save_multi_predicate_matches(self, gt_path: str, output_path: str, core_nodes_config: Dict, source_data: Dict,
                                        prob_threshold: float = 0.5) -> None:
        """
        过滤并保存单个查询的匹配结果（内部方法）。
        """
        query_basename = os.path.basename(gt_path).replace("_matches.csv", "")
        
        # --- 修改点 2: 将 print 改为 tqdm.write，避免打乱进度条显示 ---
        # tqdm.write(f"正在处理: {query_basename}") # 如果觉得日志太多，可以注释掉这行
        
        if query_basename not in core_nodes_config:
            tqdm.write(f"⚠️ 配置中未找到 {query_basename}，跳过。")
            return

        query_config = core_nodes_config[query_basename]
        query_config_int_labels = {int(k): v for k, v in query_config.items()}
        core_node_cols = [f"u{vid}" for label, vids in query_config_int_labels.items() for vid in vids]
        
        if not core_node_cols:
            tqdm.write(f"⚠️ {query_basename} 没有核心节点，跳过。")
            return

        try:
            # 加载原始匹配数据 (Lazy)
            gt_df = pl.scan_csv(gt_path)
        except Exception as e:
            tqdm.write(f"❌ 读取 {gt_path} 失败: {e}")
            return

        idmap = source_data["idmap"]
        oracle_source = source_data["oracle_source"]
        
        # 添加行号以便后续 Join
        current_df = gt_df.with_row_index(name="row_nr")
        all_oracle_cols = []

        for i, col_name in enumerate(core_node_cols):
            # 提取当前核心节点列，准备验证
            temp_df = current_df.select([pl.col(col_name).alias("internal_id")]).with_row_index(name="row_nr_temp")
            
            # 关联 ID 映射和 Oracle 概率
            temp_df = temp_df.join(idmap, on="internal_id", how="left")
            temp_df = temp_df.join(oracle_source, on=["orig_id", "type"], how="left")
            
            oracle_col_name = f"oracle_ok_{i}"
            all_oracle_cols.append(oracle_col_name)
            
            # 判定是否满足阈值
            temp_df = temp_df.with_columns(
                (pl.col("oracle_prob").fill_null(0.0) > prob_threshold).alias(oracle_col_name)
            )
            
            # 将判定结果 (True/False) 拼回主表
            current_df = current_df.join(
                temp_df.select([pl.col("row_nr_temp").alias("row_nr"), pl.col(oracle_col_name)]), 
                on="row_nr",
                how="left"
            )

        # 联合判定：所有核心节点都必须满足条件 (AND 逻辑)
        if all_oracle_cols:
            valid_matches = current_df.filter(pl.all_horizontal([pl.col(c) for c in all_oracle_cols]))
        else:
            valid_matches = current_df

        # 清理辅助列，只保留原始结构列
        cols_to_drop = ["row_nr"] + all_oracle_cols
        final_df = valid_matches.drop(cols_to_drop)

        # 执行计算并写入 CSV
        final_df.collect().write_csv(output_path)


# def load_and_prepare_sources(id_mapping_path: str, post_csv_path: str, comment_csv_path: str) -> Dict[
#     str, pl.LazyFrame]:
#     """
#     惰性加载 id_mapping, post, 和 comment 文件，并进行初步处理。
#     """
#
#     try:
#         # 惰性扫描所有文件
#         idmap_df = pl.scan_csv(id_mapping_path)
#         post_df = pl.scan_csv(post_csv_path)
#         comment_df = pl.scan_csv(comment_csv_path)
#     except Exception as e:
#         raise FileNotFoundError(f"无法扫描输入文件: {e}")
#
#     # 预处理 id_mapping
#     idmap_df = idmap_df.select(
#         pl.col("internal_id"),
#         pl.col("orig_id"),
#         pl.col("type").str.to_lowercase().alias("type")
#     )
#
#     # 预处理 post.csv
#     post_df = post_df.select(
#         pl.col("id:ID").alias("orig_id"),
#         pl.col("ML1_oracle1_probability").cast(pl.Float64).fill_null(0.0).alias("oracle_prob")
#     ).with_columns(pl.lit("post").alias("type"))
#
#     # 预处理 comment.csv
#     comment_df = comment_df.select(
#         pl.col("id:ID").alias("orig_id"),
#         pl.col("ML2_oracle2_probability").cast(pl.Float64).fill_null(0.0).alias("oracle_prob")
#     ).with_columns(pl.lit("comment").alias("type"))
#
#     # 将 post 和 comment 的 oracle 信息合并成一个大表
#     oracle_source = pl.concat([post_df, comment_df])
#
#     print("已成功预加载 id_mapping, post, 和 comment 数据。")
#     return {"idmap": idmap_df, "oracle_source": oracle_source}
#
#
# def compute_T_true_multi_predicate_polars(
#         gt_path: str,
#         core_nodes_config: Dict[str, Dict[str, List[int]]],
#         source_data: Dict[str, pl.LazyFrame],
#         prob_threshold: float = 0.5
# ) -> float:
#     """
#     使用 Polars 计算多谓词（多标签）场景下的 T_true (修复 'row_nr' 重复问题)。
#     """
#     query_basename = os.path.basename(gt_path).replace("_matches.csv", ".graph")
#     print(f"\n--- 正在为查询 '{query_basename}' 计算 T_true (多谓词) ---")
#
#     # ... (Step 1: 获取配置, 保持不变) ...
#     if query_basename not in core_nodes_config: return 0.0
#     query_config = core_nodes_config[query_basename]
#     query_config_int_labels = {int(k): v for k, v in query_config.items()}
#     core_node_cols = []
#     for label, vids in query_config_int_labels.items():
#         for vid in vids:
#             core_node_cols.append(f"u{vid}")
#     if not core_node_cols: return 0.0
#
#     # ... (Step 2: 惰性读取GT, 保持不变) ...
#     try:
#         gt_df = pl.scan_csv(gt_path).select(core_node_cols)
#     except Exception:
#         return 0.0
#
#     # 3. 逐个连接核心节点列
#     idmap = source_data["idmap"]
#     oracle_source = source_data["oracle_source"]
#
#     # --- 【关键修复 1】---
#     # 在循环开始前，只添加一次行号列
#     current_df = gt_df.with_row_index(name="row_nr")
#
#     all_oracle_cols = []
#
#     for i, col_name in enumerate(core_node_cols):
#         # 只需要为 temp_df 添加行号
#         temp_df = current_df.select([pl.col(col_name).alias("internal_id")]).with_row_index(name="row_nr_temp")
#
#         # 连接 idmap 和 oracle_source
#         temp_df = temp_df.join(idmap, on="internal_id", how="left")
#         temp_df = temp_df.join(oracle_source, on=["orig_id", "type"], how="left")
#
#         # 计算 oracle 条件
#         oracle_col_name = f"oracle_ok_{i}"
#         all_oracle_cols.append(oracle_col_name)
#         temp_df = temp_df.with_columns(
#             (pl.col("oracle_prob").fill_null(0.0) > prob_threshold).alias(oracle_col_name)
#         )
#
#         # --- 【关键修复 2】---
#         # 连接时使用不同的行号列名，或者只选择需要的列
#         # 这里我们连接后只保留 oracle_col_name
#         # 连接 temp_df (它有 'row_nr_temp') 和 current_df (它有 'row_nr')
#         current_df = current_df.join(
#             temp_df.select([pl.col("row_nr_temp").alias("row_nr"), pl.col(oracle_col_name)]),
#             on="row_nr",
#             how="left"
#         )
#
#     # 4. 过滤出所有 oracle 条件都满足的行
#     if not all_oracle_cols:  # 如果没有oracle列（例如核心节点为空），则所有都算有效
#         valid_matches = current_df
#     else:
#         all_true_expr = pl.all_horizontal([pl.col(c) for c in all_oracle_cols])
#         valid_matches = current_df.filter(all_true_expr)
#
#     # 5. 计算 T_true
#     result = valid_matches.select(pl.len()).collect()
#
#     T_true = result[0, 0] if result is not None and not result.is_empty() else 0.0
#
#     print(f"✅ 计算完成: T_true for '{query_basename}' = {T_true}")
#     return float(T_true)
#
#
# def get_or_compute_all_T_true(dataset_name: str) -> Dict[str, float]:
#     """
#     主函数：检查缓存，如果不存在则计算所有查询的多谓词T_true，并保存到缓存。
#     """
#     base_path = f"/home/wangshuo/resource/datasets/parler_data/{dataset_name}"
#     # --- 缓存文件路径 ---
#     cache_path = os.path.join(base_path, "results", "T_true.json")
#
#     # 1. 检查缓存是否存在
#     if os.path.exists(cache_path):
#         print(f"✅ 找到 T_true 缓存文件: {cache_path}")
#         with open(cache_path, 'r') as f:
#             all_T_true_results = json.load(f)
#         print("已从缓存加载 T_true 数据。")
#         return all_T_true_results
#
#     # 2. 如果缓存不存在，则执行计算
#     print(f"⚠️ 未找到 T_true 缓存文件，开始计算...")
#
#     # --- 路径配置 ---
#     gt_dir = os.path.join(base_path, "ground_truth", "structure_result")
#     core_config_path = os.path.join(base_path, "data_graph", "core_nodes_config.json")
#     id_mapping_path = os.path.join(base_path, "data_graph", "id_mapping.csv")
#     post_csv_path = os.path.join(base_path, "csv_data", "post.csv")
#     comment_csv_path = os.path.join(base_path, "csv_data", "comment.csv")
#
#     all_T_true_results = {}
#     try:
#         with open(core_config_path, 'r') as f:
#             core_nodes_config = json.load(f)
#
#         source_data = load_and_prepare_sources(id_mapping_path, post_csv_path, comment_csv_path)
#
#         gt_files = [f for f in os.listdir(gt_dir) if f.endswith('_matches.csv')]
#         if not gt_files:
#             print(f"[警告] 在目录 {gt_dir} 中没有找到任何 '_matches.csv' 文件。")
#             return {}
#
#         for gt_file in sorted(gt_files):
#             gt_path = os.path.join(gt_dir, gt_file)
#             T_true = compute_T_true_multi_predicate_polars(
#                 gt_path=gt_path,
#                 core_nodes_config=core_nodes_config,
#                 source_data=source_data,
#                 prob_threshold=0.5
#             )
#             query_basename = os.path.basename(gt_path).replace("_matches.csv", ".graph")
#             all_T_true_results[query_basename] = T_true
#
#     except FileNotFoundError as e:
#         print(f"[严重错误] 计算 T_true 时依赖文件未找到: {e}")
#         return {}
#     except Exception as e:
#         print(f"[严重错误] 计算 T_true 时发生未知异常: {e}")
#         return {}
#
#     # 3. 将计算结果保存到缓存文件
#     print(f"\n--- T_true 计算完成，结果汇总 ---")
#     print(json.dumps(all_T_true_results, indent=4))
#
#     os.makedirs(os.path.dirname(cache_path), exist_ok=True)
#     with open(cache_path, 'w') as f:
#         json.dump(all_T_true_results, f, indent=4)
#     print(f"✅ T_true 结果已缓存到: {cache_path}")
#
#     return all_T_true_results
#
