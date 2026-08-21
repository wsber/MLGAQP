import os
import re
import pandas as pd
from typing import List, Optional
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from tqdm import tqdm

DetectorFactory.seed = 0  # 保证 langdetect 结果稳定


class CSVProcessor:
    def __init__(self, file_name: str, encoding: str = "utf-8"):
        """
        初始化 CSV 处理器
        参数:
            file_path: CSV 文件路径
            encoding: 读取编码（默认utf-8）
        使用示例:
            processor = CSVProcessor("comment.csv")
        """
        self.file_dir = '/home/wangshuo/resource/datasets/parler_data/30W_valid_user/10-10-5/'
        self.file_name = file_name
        self.file_path = self.file_dir + self.file_name
        self.encoding = encoding
        self.df = pd.read_csv(self.file_path, encoding=encoding)
        # 集成进度条
        tqdm.pandas(desc="Processing rows")

    def rename_columns(self, rename_dict: dict):
        """
        重命名列名
        参数:
            rename_dict: 旧列名到新列名的映射字典
        使用示例:
            processor.rename_columns({"old_name": "new_name"})
        """
        self.df.rename(columns=rename_dict, inplace=True)

    def drop_column(self, col_name: str):
        """
        删除指定列
        参数:
            col_name: 要删除的列名
        使用示例:
            processor.drop_column("ML1_score")
        """
        if col_name in self.df.columns:
            self.df.drop(columns=[col_name], inplace=True)

    def keep_columns(self, col_list: List[str]):
        """
        只保留指定列
        参数:
            col_list: 需要保留的列名列表
        使用示例:
            processor.keep_columns(["id:ID", "body", "score"])
        """
        self.df = self.df[col_list]

    def lowercase_column(self, col_name: str):
        """
        将某一列的字符串转换为小写
        参数:
            col_name: 列名
        使用示例:
            processor.lowercase_column("ML2_label")
        """
        if col_name in self.df.columns:
            self.df[col_name] = self.df[col_name].astype(str).str.lower()

    def replace_column_values(self, col_name: str, replace_map: dict):
        """
        替换某列中的值
        参数:
            col_name: 列名
            replace_map: 替换映射
        使用示例:
            processor.replace_column_values("LLM_label", {"yes": "deepseek_yes", "no": "deepseek_no"})
        """
        if col_name in self.df.columns:
            self.df[col_name] = self.df[col_name].replace(replace_map)

    def filter_by_value(self, col_name: str, value):
        """
        保留某列等于指定值的行
        参数:
            col_name: 列名
            value: 要匹配的值
        使用示例:
            processor.filter_by_value("label", "positive")
        """
        self.df = self.df[self.df[col_name] == value]

    def sort_by_column(self, col_name: str, ascending=True):
        """
        根据某一列进行排序
        参数:
            col_name: 列名
            ascending: 是否升序（默认True）
        使用示例:
            processor.sort_by_column("ML1_probability", ascending=False)
        """
        self.df = self.df.sort_values(by=col_name, ascending=ascending)

    def sort_by_labels_and_probs(self, label_col: str, prob_col: str, label_order: List[str]):
        """
        先按标签顺序排序，再按概率降序
        参数:
            label_col: 标签列名
            prob_col: 概率列名
            label_order: 标签的排序顺序列表
        使用示例:
            processor.sort_by_labels_and_probs("label", "prob", ["positive", "negative", "neutral"])
        """
        self.df[label_col] = pd.Categorical(self.df[label_col], categories=label_order, ordered=True)
        self.df = self.df.sort_values(by=[label_col, prob_col], ascending=[True, False])

    def update_column_from_another_df(self, other_df: pd.DataFrame, index_col: str, columns: List[str]):
        """
        根据另一张表的索引值更新当前表中的列
        参数:
            other_df: 另一个 DataFrame
            index_col: 索引列名
            columns: 要更新的列名列表
        使用示例:
            processor.update_column_from_another_df(df2, index_col="id:ID", columns=["ML1_prob", "ML1_label"])
        """
        self.df.set_index(index_col, inplace=True)
        other_df = other_df.set_index(index_col)
        for col in columns:
            self.df[col] = other_df[col]
        self.df.reset_index(inplace=True)

    def keep_rows_with_ids(self, id_list: List[str], id_col: str):
        """
        只保留指定 ID 值的行
        参数:
            id_list: 合法 ID 列表
            id_col: id 列名
        使用示例:
            processor.keep_rows_with_ids(["id1", "id2"], id_col="post")
        """
        self.df = self.df[self.df[id_col].isin(id_list)]

    def drop_short_or_non_english_rows(self, text_col: str):
        """
        删除英文字符数 ≤1 或 非英文的行
        参数:
            text_col: 文本列名
        使用示例:
            processor.drop_short_or_non_english_rows("body")
        """
        def is_letters_le_1(text: str) -> bool:
            return len(re.findall(r"[A-Za-z]", str(text))) <= 1

        def is_non_english(text: str) -> bool:
            try:
                return detect(str(text)) != 'en'
            except LangDetectException:
                return True

        mask_le1 = self.df[text_col].apply(is_letters_le_1)
        mask_non_en = self.df[text_col].apply(is_non_english)
        self.df = self.df.loc[~(mask_le1 | mask_non_en)].reset_index(drop=True)

    def reverse_probability_by_label(self, prob_col: str, label_col: str, label_val: str):
        """
        若 label_col 中值为 label_val，则将 prob_col 的值变为 1 - 原值
        参数:
            prob_col: 概率列名
            label_col: 标签列名
            label_val: 触发转换的标签值
        使用示例:
            processor.reverse_probability_by_label("prob", "label", "LABEL_0")
        """
        mask = self.df[label_col] == label_val
        self.df.loc[mask, prob_col] = 1 - self.df.loc[mask, prob_col]

    def modify_column_value(self, col_name: str, func):
        """
        对某列应用自定义函数（如加减乘除）
        参数:
            col_name: 列名
            func: 要应用的函数
        使用示例:
            processor.modify_column_value("score", lambda x: x + 10)
        """
        if col_name in self.df.columns:
            self.df[col_name] = self.df[col_name].apply(func)

    def save(self, output_path: Optional[str] = None):
        """
        保存 DataFrame 到 CSV
        参数:
            output_path: 输出路径（默认覆盖原文件）
        使用示例:
            processor.save("output.csv")
        """
        output_path = output_path or self.file_path
        self.df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"✅ 已保存到: {output_path}")
