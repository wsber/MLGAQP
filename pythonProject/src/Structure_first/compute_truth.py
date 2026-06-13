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

    def __init__(self, dataset_name: str,
                 post_oracle_col: str = "ML1_oracle1_probability",
                 comment_oracle_col: str = "ML2_oracle2_probability",
                 parent_dataset: str = "parler_data",
                 table1: str = "post",      
                 table2: str = "comment"):
        """
        初始化 GroundTruthManager。

        Args:
            dataset_name (str): 数据集的名称，用于构建所有相关文件路径。
        """
        self.dataset_name = dataset_name
        self.post_oracle_col = post_oracle_col
        self.comment_oracle_col = comment_oracle_col
        self.table1 = table1.lower()
        self.table2 = table2.lower()

        # --- 集中管理所有路径 ---
        # self.base_path = f"/home/wangshuo/resource/datasets/parler_data/{dataset_name}"
        self.base_path = f"/home/wangshuo/resource/datasets/{parent_dataset}/{dataset_name}"
        
        safe_post = post_oracle_col.replace("/", "_")
        safe_comment = comment_oracle_col.replace("/", "_")
        self.cache_path = os.path.join(
            self.base_path, "results", 
            f"T_true_{safe_post}_{safe_comment}.json"
        )

        self.gt_dir = os.path.join(self.base_path, "ground_truth", "structure_result")
        self.core_config_path = os.path.join(self.base_path, "data_graph", "core_nodes_config.json")
        self.id_mapping_path = os.path.join(self.base_path, "data_graph", "id_mapping.csv")
        # self.post_csv_path = os.path.join(self.base_path, "csv_data", "post.csv")
        # self.comment_csv_path = os.path.join(self.base_path, "csv_data", "comment.csv")
        self.table1_csv_path = os.path.join(self.base_path, "csv_data", f"{self.table1}.csv")
        self.table2_csv_path = os.path.join(self.base_path, "csv_data", f"{self.table2}.csv")



    def get_all(self,
            agg_mode: str = "count",
            sum_on: str = None,
            sum_col: str = None,
            sum_match_col: str = None) -> Dict[str, float]:
    
        # 主函数：默认计算 count 版 T_true；当 agg_mode='sum' 时计算 SUM 版 T_true。
        # Args:
            # agg_mode: 'count' | 'sum'
            # sum_on:   'post' | 'comment'（仅 agg_mode='sum' 时需要）
            # sum_col:  post.csv/comment.csv 中被求和的列名（仅 agg_mode='sum' 时需要）
            # sum_match_col: *_matches.csv 里对应要取值的节点变量列名（例如 'u3'）。
            #                若不提供且 core_node_cols 只有 1 个，则自动用那一列；否则报错避免歧义。
            
        agg_mode = str(agg_mode).lower()

        safe_post = self.post_oracle_col.replace("/", "_")
        safe_comment = self.comment_oracle_col.replace("/", "_")

        cache_path = self.cache_path
        if agg_mode == "sum":
            if sum_on is None or sum_col is None:
                raise ValueError("agg_mode='sum' 需要同时指定 sum_on=('post'|'comment') 和 sum_col='列名'")
            sum_on = str(sum_on).lower()
            if sum_on not in {"post", "comment"}:
                raise ValueError("sum_on 只能是 'post' 或 'comment'")

            safe_sum_col = str(sum_col).replace("/", "_").replace(":", "_")
            safe_sum_match_col = str(sum_match_col or "auto").replace("/", "_").replace(":", "_")

            cache_path = os.path.join(
                self.base_path, "results",
                f"T_true_sum_{sum_on}_{safe_sum_col}_{safe_sum_match_col}_{safe_post}_{safe_comment}.json"
            )
        elif agg_mode != "count":
            raise ValueError("agg_mode 只能是 'count' 或 'sum'")

        if os.path.exists(cache_path):
            print(f"✅ 找到 T_true 缓存文件: {cache_path}")
            with open(cache_path, "r") as f:
                all_T_true_results = json.load(f)
            print("已从缓存加载 T_true 数据。")
            return all_T_true_results

        print(f"⚠️ 未找到 T_true 缓存文件，开始计算... (agg_mode={agg_mode})")
        all_T_true_results = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    all_T_true_results = loaded
                    print(f"✅ 检测到已有缓存，断点续跑: {cache_path} (已完成 {len(all_T_true_results)} 条)")
                else:
                    print(f"[警告] 缓存文件格式异常，将重新计算: {cache_path}")
            except Exception as e:
                print(f"[警告] 读取缓存失败，将重新计算。原因: {e}")
        else:
            print(f"⚠️ 未找到 T_true 缓存文件，开始计算... (agg_mode={agg_mode})")

        try:
            with open(self.core_config_path, "r") as f:
                core_nodes_config = json.load(f)

            source_data = self._load_and_prepare_sources(
                agg_mode=agg_mode, sum_on=sum_on, sum_col=sum_col
            )

            gt_files = [f for f in os.listdir(self.gt_dir) if f.endswith("_matches.csv")]
            if not gt_files:
                print(f"[警告] 在目录 {self.gt_dir} 中没有找到任何 '_matches.csv' 文件。")
                return {}

            os.makedirs(os.path.dirname(cache_path), exist_ok=True)

            for gt_file in sorted(gt_files):
                gt_path = os.path.join(self.gt_dir, gt_file)
                query_basename = os.path.basename(gt_path).replace("_matches.csv", "")

                # 已有结果则跳过，支持断点续跑
                if query_basename in all_T_true_results:
                    continue

                T_true = self._compute_multi_predicate_polars(
                    gt_path=gt_path,
                    core_nodes_config=core_nodes_config,
                    source_data=source_data,
                    prob_threshold=0.5,
                    agg_mode=agg_mode,
                    sum_on=sum_on,
                    sum_col=sum_col,
                    sum_match_col=sum_match_col,
                )
                all_T_true_results[query_basename] = float(T_true)

                # 每算完一个 query 就立即保存
                with open(cache_path, "w") as f:
                    json.dump(all_T_true_results, f, indent=4, ensure_ascii=False)

                print(f"[SAVE] {query_basename} -> 当前已保存 {len(all_T_true_results)} 条")

        except FileNotFoundError as e:
            print(f"[严重错误] 计算 T_true 时依赖文件未找到: {e}")
            return all_T_true_results
        except Exception as e:
            print(f"[严重错误] 计算 T_true 时发生未知异常: {e}")
            return all_T_true_results

        print(f"\n--- T_true 计算完成，结果汇总 ---")
        print(json.dumps(all_T_true_results, indent=4, ensure_ascii=False))
        print(f"✅ T_true 已增量缓存到: {cache_path}")

        return all_T_true_results

    def _load_and_prepare_sources(self,
                              agg_mode: str = "count",
                              sum_on: str = None,
                              sum_col: str = None) -> Dict[str, pl.LazyFrame]:
        """惰性加载和预处理源文件。"""
        try:
            idmap_df = pl.scan_csv(self.id_mapping_path)
            # post_df = pl.scan_csv(self.post_csv_path)
            # comment_df = pl.scan_csv(self.comment_csv_path)
            table1_df = pl.scan_csv(self.table1_csv_path)
            table2_df = pl.scan_csv(self.table2_csv_path)

            # post_raw = post_df
            # comment_raw = comment_df
            t1_raw = table1_df
            t2_raw = table2_df
        except Exception as e:
            raise FileNotFoundError(f"无法扫描输入文件: {e}")

        # 2. 预处理 id_mapping
        # idmap_df = idmap_df.select(
        #     pl.col("internal_id"),
        #     pl.col("orig_id"),
        #     pl.col("type").str.to_lowercase().alias("type")
        # )
        idmap_df = idmap_df.select(
            pl.col("internal_id").cast(pl.String),      # <--- 强制转为 String
            pl.col("orig_id").cast(pl.String),          # <--- 强制转为 String
            pl.col("type").str.to_lowercase().alias("type")
        )

        # 3. 动态处理 Post 数据
        # post_cols = post_df.collect_schema().names()
        # if self.post_oracle_col in post_cols:
        #     post_prob_expr = pl.col(self.post_oracle_col)
        # else:
        #     print(f"[警告] post.csv 缺少列 '{self.post_oracle_col}'，该表的 oracle_prob 将全部填 0。")
        #     post_prob_expr = pl.lit(0.0)

        # post_df = post_df.select(
        #     pl.col("id:ID").alias("orig_id"),
        #     post_prob_expr.cast(pl.Float64).fill_null(0.0).alias("oracle_prob")
        # ).with_columns(pl.lit("post").alias("type"))
        t1_cols = table1_df.collect_schema().names()
        if self.post_oracle_col in t1_cols:
            t1_prob_expr = pl.col(self.post_oracle_col)
        else:
            print(f"[警告] {self.table1}.csv 缺少列 '{self.post_oracle_col}'，全填 0。")
            t1_prob_expr = pl.lit(0.0)

        # table1_df = table1_df.select(
        #     pl.col("id:ID").alias("orig_id"),
        #     t1_prob_expr.cast(pl.Float64).fill_null(0.0).alias("oracle_prob")
        # ).with_columns(pl.lit(self.table1).alias("type"))  # <--- 使用表名作为类型
        table1_df = table1_df.select(
            pl.col("id:ID").cast(pl.String).alias("orig_id"),  # <--- 加入 .cast(pl.String)
            t1_prob_expr.cast(pl.Float64).fill_null(0.0).alias("oracle_prob")
        ).with_columns(pl.lit(self.table1).alias("type"))

        # 4. 动态处理 Comment 数据
        t2_cols = table2_df.collect_schema().names()
        if self.comment_oracle_col in t2_cols:
            t2_prob_expr = pl.col(self.comment_oracle_col)
        else:
            print(f"[警告] {self.table2}.csv 缺少列 '{self.comment_oracle_col}'，全填 0。")
            t2_prob_expr = pl.lit(0.0)

        # table2_df = table2_df.select(
        #     pl.col("id:ID").alias("orig_id"),
        #     t2_prob_expr.cast(pl.Float64).fill_null(0.0).alias("oracle_prob")
        # ).with_columns(pl.lit(self.table2).alias("type"))
        table2_df = table2_df.select(
            pl.col("id:ID").cast(pl.String).alias("orig_id"),  # <--- 加入 .cast(pl.String)
            t2_prob_expr.cast(pl.Float64).fill_null(0.0).alias("oracle_prob")
        ).with_columns(pl.lit(self.table2).alias("type"))

        oracle_source = pl.concat([table1_df, table2_df])

        value_source = None
        agg_mode = str(agg_mode).lower()
        if agg_mode == "sum":
            sum_on = str(sum_on).lower()
            if sum_on == self.table1:
                if sum_col not in t1_cols: raise ValueError(f"{self.table1}.csv 缺列 '{sum_col}'")
                value_source = (
                    t1_raw.select(
                        pl.col("id:ID").alias("orig_id"),
                        pl.col(sum_col).cast(pl.Float64).fill_null(0.0).alias("sum_value")
                    ).with_columns(pl.lit(self.table1).alias("type"))
                )
            elif sum_on == self.table2:
                if sum_col not in t2_cols: raise ValueError(f"{self.table2}.csv 缺列 '{sum_col}'")
                value_source = (
                    t2_raw.select(
                        pl.col("id:ID").alias("orig_id"),
                        pl.col(sum_col).cast(pl.Float64).fill_null(0.0).alias("sum_value")
                    ).with_columns(pl.lit(self.table2).alias("type"))
                )
            else:
                raise ValueError(f"sum_on 只能是 '{self.table1}' 或 '{self.table2}'")
        return {"idmap": idmap_df, "oracle_source": oracle_source, "value_source": value_source}
        # comment_cols = comment_df.collect_schema().names()
        # if self.comment_oracle_col in comment_cols:
        #     comment_prob_expr = pl.col(self.comment_oracle_col)
        # else:
        #     print(f"[警告] comment.csv 缺少列 '{self.comment_oracle_col}'，该表的 oracle_prob 将全部填 0。")
        #     comment_prob_expr = pl.lit(0.0)

        # comment_df = comment_df.select(
        #     pl.col("id:ID").alias("orig_id"),
        #     comment_prob_expr.cast(pl.Float64).fill_null(0.0).alias("oracle_prob")
        # ).with_columns(pl.lit("comment").alias("type"))

        # oracle_source = pl.concat([post_df, comment_df])

        # value_source = None
        # agg_mode = str(agg_mode).lower()
        # if agg_mode == "sum":
        #     if sum_on is None or sum_col is None:
        #         raise ValueError("agg_mode='sum' 需要 sum_on 和 sum_col")
        #     sum_on = str(sum_on).lower()
        #     if sum_on == "post":
        #         if sum_col not in post_cols:
        #             raise ValueError(f"post.csv 不存在列 '{sum_col}'")
        #         value_source = (
        #             post_raw.select(
        #                 pl.col("id:ID").alias("orig_id"),
        #                 pl.col(sum_col).cast(pl.Float64).fill_null(0.0).alias("sum_value")
        #             ).with_columns(pl.lit("post").alias("type"))
        #         )
        #     elif sum_on == "comment":
        #         if sum_col not in comment_cols:
        #             raise ValueError(f"comment.csv 不存在列 '{sum_col}'")
        #         value_source = (
        #             comment_raw.select(
        #                 pl.col("id:ID").alias("orig_id"),
        #                 pl.col(sum_col).cast(pl.Float64).fill_null(0.0).alias("sum_value")
        #             ).with_columns(pl.lit("comment").alias("type"))
        #         )
        #     else:
        #         raise ValueError("sum_on 只能是 'post' 或 'comment'")
        # print("已成功预加载 id_mapping, post, 和 comment 数据。")
        # return {"idmap": idmap_df, "oracle_source": oracle_source, "value_source": value_source}
    
    
    def _compute_multi_predicate_polars(self, gt_path: str, core_nodes_config: Dict, source_data: Dict,
                                    prob_threshold: float = 0.5,
                                    agg_mode: str = "count",
                                    sum_on: str = None,
                                    sum_col: str = None,
                                    sum_match_col: str = None) -> float:
        """使用 Polars 计算多谓词场景下的 T_true（count 或 sum）。"""
        agg_mode = str(agg_mode).lower()

        query_basename = os.path.basename(gt_path).replace("_matches.csv", "")
        print(f"\n--- 正在为查询 '{query_basename}' 计算 T_true (multi, agg_mode={agg_mode}) ---")
        if query_basename not in core_nodes_config:
            return 0.0

        query_config = core_nodes_config[query_basename]
        query_config_int_labels = {int(k): v for k, v in query_config.items()}
        core_node_cols = [f"u{vid}" for label, vids in query_config_int_labels.items() for vid in vids]
        if not core_node_cols:
            return 0.0

        select_cols = list(core_node_cols)
        sum_match_cols = []
        if agg_mode == "sum":
            if sum_match_col is None:
                if len(core_node_cols) == 1:
                    sum_match_col = core_node_cols[0]
                else:
                    raise ValueError("agg_mode='sum' 且 core_node_cols>1：请显式传入 sum_match_col（例如 'u3'）")
            elif isinstance(sum_match_col, str):
                sum_match_cols = [sum_match_col]
            else:
                # 认为是列表形式
                sum_match_cols = list(sum_match_col)

            # if sum_match_col not in select_cols:
            #     select_cols.append(sum_match_col)
            for col in sum_match_cols:
                if col not in select_cols:
                    select_cols.append(col)

        try:
            gt_df = pl.scan_csv(gt_path).select(select_cols)
        except Exception:
            return 0.0

        idmap = source_data["idmap"]
        oracle_source = source_data["oracle_source"]

        current_df = gt_df.with_row_index(name="row_nr")
        all_oracle_cols = []

        for i, col_name in enumerate(core_node_cols):
            # temp_df = current_df.select([pl.col(col_name).alias("internal_id")]).with_row_index(name="row_nr_temp")
            # temp_df = temp_df.join(idmap, on="internal_id", how="left")
            temp_df = current_df.select([
                pl.col(col_name).cast(pl.String).alias("internal_id")
            ]).with_row_index(name="row_nr_temp")
            temp_df = temp_df.join(idmap, on="internal_id", how="left")
            temp_df = temp_df.join(oracle_source, on=["orig_id", "type"], how="left")

            oracle_col_name = f"oracle_ok_{i}"
            all_oracle_cols.append(oracle_col_name)
            temp_df = temp_df.with_columns(
                (pl.col("oracle_prob").fill_null(0.0) > prob_threshold).alias(oracle_col_name)
            )
            current_df = current_df.join(
                temp_df.select([pl.col("row_nr_temp").alias("row_nr"), pl.col(oracle_col_name)]),
                on="row_nr",
                how="left"
            )

        valid_matches = current_df.filter(
            pl.all_horizontal([pl.col(c) for c in all_oracle_cols])
        ) if all_oracle_cols else current_df

        if agg_mode == "count":
            result = valid_matches.select(pl.len()).collect()
            T_true = result[0, 0] if result is not None and not result.is_empty() else 0.0
            print(f"✅ 计算完成: T_true(count) for '{query_basename}' = {T_true}")
            return float(T_true)

        if agg_mode == "sum":
            value_source = source_data.get("value_source")
            if value_source is None:
                raise ValueError("agg_mode='sum' 但 value_source 未准备好（检查 _load_and_prepare_sources 参数）")

            # joined = (
            #     valid_matches
            #     .join(idmap, left_on=sum_match_col, right_on="internal_id", how="left")
            #     .join(value_source, on=["orig_id", "type"], how="left")
            # )
            # result = joined.select( pl.col("sum_value").fill_null(0.0).sum()).collect()
            # T_true = result[0, 0] if result is not None and not result.is_empty() else 0.0
            # print(f"✅ 计算完成: T_true(sum) for '{query_basename}' = {T_true}")
            # return float(T_true)
            total_sum = 0.0
            for col in sum_match_cols:
                # joined = (
                #     valid_matches
                #     .join(idmap, left_on=col, right_on="internal_id", how="left")
                #     .join(value_source, on=["orig_id", "type"], how="left")
                # )
                joined = (
                    # 匹配结果表中的对应 u_col（如 u3）也是整数读入的，需要强转 string
                    valid_matches.with_columns(pl.col(col).cast(pl.String))
                    .join(idmap, left_on=col, right_on="internal_id", how="left")
                    .join(value_source, on=["orig_id", "type"], how="left")
                )
                result = joined.select(pl.col("sum_value").fill_null(0.0).sum()).collect()
                if result is not None and not result.is_empty():
                    total_sum += float(result[0, 0])

            T_true = total_sum
            print(f"✅ 计算完成: T_true(sum) for '{query_basename}' = {T_true}")
            return float(T_true)

        raise ValueError("agg_mode 只能是 'count' 或 'sum'")
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
