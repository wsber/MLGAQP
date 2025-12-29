#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastestGraphConverter

功能：
1. 读取 user.csv、post.csv、comment.csv 三个文件；
2. 构建 Fastest 输入图；
3. 输出：
   - parler.graph
   - id_mapping.csv
4. 输出节点与边的统计信息。
"""

import os
import csv
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple


class FastestGraphConverter:
    # ============================================================
    # 🧩 类型定义
    # ============================================================
    NODE_LABELS = {
        "User": 0,
        "Post": 1,
        "Comment": 2
    }

    EDGE_LABELS = {
        "author_user_post": 0,      # user -> post (creator)
        "creator_user_comment": 1,  # user -> comment (creator)
        "post_has_comment": 2,      # post -> comment (post)
        "comment_reply_comment": 3, # comment -> comment (parent)
        "user_view_post": 4         # user -> post (view)
    }

    def __init__(self, base_dir: str,Graph_Lib_Dir:str):
        """
        初始化转换器
        :param base_dir: 存放 user.csv, post.csv, comment.csv 的目录
        """
        self.base_dir = base_dir
        self.user_file = os.path.join(base_dir, "user.csv")
        self.post_file = os.path.join(base_dir, "post.csv")
        self.comment_file = os.path.join(base_dir, "comment.csv")
        self.output_graph = os.path.join(Graph_Lib_Dir, "parler.graph")
        self.output_map = os.path.join(Graph_Lib_Dir, "id_mapping.csv")

        self.nodes = []
        self.edges = set()
        self.id_to_internal = {}
        self.internal_counter = 0

    # ============================================================
    # 🧩 辅助函数
    # ============================================================

    @staticmethod
    def _clean_column_name(col: str) -> str:
        """清理CSV中的复杂列名（去掉冒号、类型标记）"""
        return col.strip().lstrip(':').split(':')[0].strip()

    @classmethod
    def _load_csv_robust(cls, path):
        """用pandas读取CSV并清理列名"""
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        new_cols = {c: cls._clean_column_name(c) for c in df.columns}
        df.rename(columns=new_cols, inplace=True)
        return df

    # ============================================================
    # 🧩 图构建逻辑
    # ============================================================

    def _add_node(self, orig_id, type_str):
        key = (type_str, orig_id)
        if key in self.id_to_internal:
            return self.id_to_internal[key]
        iid = self.internal_counter
        self.id_to_internal[key] = iid
        self.nodes.append((iid, orig_id, type_str, self.NODE_LABELS[type_str]))
        self.internal_counter += 1
        return iid

    def _add_edge(self, type_u, orig_u, type_v, orig_v, edge_label):
        """添加一条无向边"""
        ku = (type_u, str(orig_u))
        kv = (type_v, str(orig_v))
        if ku not in self.id_to_internal or kv not in self.id_to_internal:
            return
        iu, iv = self.id_to_internal[ku], self.id_to_internal[kv]
        u, v = sorted([iu, iv])
        self.edges.add((u, v, edge_label))

    def build_graph(self, users_df, posts_df, comments_df):
        """根据三类节点构建完整图"""
        # ---- 添加 User ----
        if users_df is not None:
            id_col = "id" if "id" in users_df.columns else list(users_df.columns)[0]
            for _, r in users_df.iterrows():
                self._add_node(str(r[id_col]), "User")

        # ---- 添加 Post ----
        if posts_df is not None:
            id_col = "id" if "id" in posts_df.columns else list(posts_df.columns)[0]
            for _, r in posts_df.iterrows():
                self._add_node(str(r[id_col]), "Post")

        # ---- 添加 Comment ----
        if comments_df is not None:
            id_col = "id" if "id" in comments_df.columns else list(comments_df.columns)[0]
            for _, r in comments_df.iterrows():
                self._add_node(str(r[id_col]), "Comment")

        # ---- 生成边 ----
        if posts_df is not None and "creator" in posts_df.columns:
            for _, r in posts_df.iterrows():
                if r["creator"]:
                    self._add_edge("User", r["creator"], "Post", r["id"],
                                   self.EDGE_LABELS["author_user_post"])

        if comments_df is not None:
            for _, r in comments_df.iterrows():
                # user -> comment
                if "creator" in r and r["creator"]:
                    self._add_edge("User", r["creator"], "Comment", r["id"],
                                   self.EDGE_LABELS["creator_user_comment"])
                # post -> comment
                if "post" in r and r["post"]:
                    self._add_edge("Post", r["post"], "Comment", r["id"],
                                   self.EDGE_LABELS["post_has_comment"])
                # comment -> comment
                if "parent" in r and r["parent"]:
                    self._add_edge("Comment", r["parent"], "Comment", r["id"],
                                   self.EDGE_LABELS["comment_reply_comment"])
                # user -> post (view)
                if "creator" in r and "post" in r and r["creator"] and r["post"]:
                    self._add_edge("User", r["creator"], "Post", r["post"],
                                   self.EDGE_LABELS["user_view_post"])

        print(f"[INFO] 图构建完成: 节点数={len(self.nodes)}, 边数={len(self.edges)}")

    def build_graph_without_author_user_post(self, users_df, posts_df, comments_df):
        """
        根据三类节点构建图，但不创建 user -> post (creator) 边。
        其他边（user->comment、post->comment、comment->comment、user->post(view)）保持不变。
        """
        # ---- 添加 User ----
        if users_df is not None:
            id_col = "id" if "id" in users_df.columns else list(users_df.columns)[0]
            for _, r in users_df.iterrows():
                self._add_node(str(r[id_col]), "User")

        # ---- 添加 Post ----
        if posts_df is not None:
            id_col = "id" if "id" in posts_df.columns else list(posts_df.columns)[0]
            for _, r in posts_df.iterrows():
                self._add_node(str(r[id_col]), "Post")

        # ---- 添加 Comment ----
        if comments_df is not None:
            id_col = "id" if "id" in comments_df.columns else list(comments_df.columns)[0]
            for _, r in comments_df.iterrows():
                self._add_node(str(r[id_col]), "Comment")

        # ---- 生成边 ----
        # ⚠️ 此处省略 user->post(creator) 边的生成

        if comments_df is not None:
            for _, r in comments_df.iterrows():
                # user -> comment
                if "creator" in r and r["creator"]:
                    self._add_edge("User", r["creator"], "Comment", r["id"],
                                   self.EDGE_LABELS["creator_user_comment"])
                # post -> comment
                if "post" in r and r["post"]:
                    self._add_edge("Post", r["post"], "Comment", r["id"],
                                   self.EDGE_LABELS["post_has_comment"])
                # comment -> comment
                if "parent" in r and r["parent"]:
                    self._add_edge("Comment", r["parent"], "Comment", r["id"],
                                   self.EDGE_LABELS["comment_reply_comment"])
                # user -> post (view)
                if "creator" in r and "post" in r and r["creator"] and r["post"]:
                    self._add_edge("User", r["creator"], "Post", r["post"],
                                   self.EDGE_LABELS["user_view_post"])

        print(f"[INFO] 图构建完成 (无 user->post creator 边): 节点数={len(self.nodes)}, 边数={len(self.edges)}")

    # ============================================================
    # 🧩 输出函数
    # ============================================================

    def _write_fastest_graph(self):
        """写入Fastest格式"""
        deg = defaultdict(int)
        for u, v, _ in self.edges:
            deg[u] += 1
            deg[v] += 1

        with open(self.output_graph, "w") as fout:
            fout.write(f"t {len(self.nodes)} {len(self.edges)}\n")
            for iid, orig_id, type_str, label_int in sorted(self.nodes, key=lambda x: x[0]):
                fout.write(f"v {iid} {label_int} {deg.get(iid, 0)}\n")
            for u, v, e in sorted(self.edges):
                fout.write(f"e {u} {v} {e}\n")

    def _write_mapping(self):
        """保存id映射"""
        with open(self.output_map, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["internal_id", "orig_id", "type", "label_int"])
            for iid, orig_id, type_str, label_int in sorted(self.nodes, key=lambda x: x[0]):
                writer.writerow([iid, orig_id, type_str, label_int])

    def _count_statistics(self):
        """输出统计信息"""
        node_count = defaultdict(int)
        for _, _, t, _ in self.nodes:
            node_count[t] += 1

        edge_count = defaultdict(int)
        for _, _, e in self.edges:
            edge_count[e] += 1

        print("\n===== 📊 图统计信息 =====")
        print(f"总节点数: {len(self.nodes)}")
        for t, cnt in node_count.items():
            print(f"  {t:<10}: {cnt}")
        print(f"总边数: {len(self.edges)}")
        for e, cnt in edge_count.items():
            relation = [k for k, v in self.EDGE_LABELS.items() if v == e]
            print(f"  EdgeLabel {e} ({relation[0] if relation else '未知'}): {cnt}")

    # ============================================================
    # 🧩 主执行流程
    # ============================================================

    def run(self):
        """执行整个转换流程"""
        users_df = self._load_csv_robust(self.user_file) if os.path.exists(self.user_file) else None
        posts_df = self._load_csv_robust(self.post_file) if os.path.exists(self.post_file) else None
        comments_df = self._load_csv_robust(self.comment_file) if os.path.exists(self.comment_file) else None

        self.build_graph(users_df, posts_df, comments_df)
        self._write_fastest_graph()
        self._write_mapping()
        self._count_statistics()

        print(f"[INFO] 图文件已保存到: {self.output_graph}")
        print(f"[INFO] 映射文件已保存到: {self.output_map}")

    def run_without_author_user_post(self):
        """执行整个转换流程（不创建 user->post creator 边）"""
        users_df = self._load_csv_robust(self.user_file) if os.path.exists(self.user_file) else None
        posts_df = self._load_csv_robust(self.post_file) if os.path.exists(self.post_file) else None
        comments_df = self._load_csv_robust(self.comment_file) if os.path.exists(self.comment_file) else None

        self.build_graph_without_author_user_post(users_df, posts_df, comments_df)
        self._write_fastest_graph()
        self._write_mapping()
        self._count_statistics()

        print(f"[INFO] 图文件已保存到: {self.output_graph}")
        print(f"[INFO] 映射文件已保存到: {self.output_map}")

    def simplify_graph_merge_edges_update_degree(self, input_path=None, output_path=None):
        """
        读取 Fastest 格式的 .graph 文件，
        去掉边标签、合并重复边（例如 user-post 间 view/public），
        并重新计算每个节点的度数，输出新图文件。

        :param input_path: 输入图路径（默认使用 self.output_graph）
        :param output_path: 输出文件路径（默认生成 *_merged.graph）
        """
        import os
        from collections import defaultdict

        input_path = input_path or self.output_graph
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"找不到输入图文件: {input_path}")

        output_path = output_path or input_path.replace(".graph", "_merged.graph")

        with open(input_path, "r") as fin:
            lines = fin.readlines()

        vertex_lines_raw = []
        edge_set = set()

        # === 读取阶段 ===
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("v "):
                vertex_lines_raw.append(line)
            elif line.startswith("e "):
                parts = line.split()
                if len(parts) < 3:
                    continue
                u, v = int(parts[1]), int(parts[2])
                # 无向边去重
                edge = tuple(sorted((u, v)))
                edge_set.add(edge)

        # === 重新计算度数 ===
        deg = defaultdict(int)
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1

        # === 生成输出内容 ===
        with open(output_path, "w") as fout:
            fout.write(f"t {len(vertex_lines_raw)} {len(edge_set)}\n")

            for line in vertex_lines_raw:
                parts = line.split()
                if len(parts) >= 3:
                    vid, label = int(parts[1]), parts[2]
                    fout.write(f"v {vid} {label} {deg.get(vid, 0)}\n")
                elif len(parts) == 2:
                    # 防止异常格式
                    vid = int(parts[1])
                    fout.write(f"v {vid} 0 {deg.get(vid, 0)}\n")

            for (u, v) in sorted(edge_set):
                fout.write(f"e {u} {v}\n")

        print(f"[✅] 图简化完成，边标签已去除、重复边已合并并更新度数：{output_path}")
        print(f"[INFO] 节点数: {len(vertex_lines_raw)}，新边数: {len(edge_set)}")

    def remove_edge_labels(self, input_path=None, output_path=None):
        """
        从 Fastest 格式的 .graph 文件中读取图，去掉所有边标签（仅保留 e u v），并去重。
        会生成新的 _nolabel.graph 文件。

        :param input_path: 输入文件路径（默认 self.output_graph）
        :param output_path: 输出文件路径（默认在同目录生成 *_nolabel.graph）
        """
        import os
        from collections import defaultdict

        input_path = input_path or self.output_graph
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"找不到输入图文件: {input_path}")

        output_path = output_path or input_path.replace(".graph", "_nolabel.graph")

        with open(input_path, "r") as fin:
            lines = fin.readlines()

        vertex_lines = []
        edge_set = set()
        node_count = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("t "):
                # 头部（忽略原有节点/边统计）
                continue
            elif line.startswith("v "):
                vertex_lines.append(line)
                node_count += 1
            elif line.startswith("e "):
                parts = line.split()
                if len(parts) < 3:
                    continue
                # 无向边：只保留 u, v
                u, v = int(parts[1]), int(parts[2])
                edge_set.add(tuple(sorted((u, v))))

        # === 重新计算度数 ===
        deg = defaultdict(int)
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1

        # === 写出新文件 ===
        with open(output_path, "w") as fout:
            fout.write(f"t {len(vertex_lines)} {len(edge_set)}\n")
            for line in vertex_lines:
                parts = line.split()
                if len(parts) >= 3:
                    vid = int(parts[1])
                    label = parts[2]
                    fout.write(f"v {vid} {label} {deg.get(vid, 0)}\n")
                else:
                    # 若节点行缺少标签，默认0
                    vid = int(parts[1])
                    fout.write(f"v {vid} 0 {deg.get(vid, 0)}\n")

            for (u, v) in sorted(edge_set):
                fout.write(f"e {u} {v}\n")

        print(f"[✅] 边标签已去除，新文件已保存: {output_path}")
        print(f"[INFO] 节点数: {len(vertex_lines)}, 新边数: {len(edge_set)}")


# ============================================================
# 🧩 独立执行入口
# ============================================================


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastestEstimateMerger

将 Fastest 的无偏节点估计结果 (sv_estimate_result.txt)
与 id_mapping.csv、post.csv 按 internal_id -> orig_id 映射合并，
仅输出估计值非零且不为空的 Post，并重命名输出列。
"""

import os
import re
from typing import Optional, List

import pandas as pd


class FastestEstimateMerger:
    DEFAULT_RENAME_MAP = {
        "ML1_oracle1_probability": "post_oracle1",
        "ML1_proxy4b1_probability": "post_proxy4b1",
        "ML1_proxy2b1_probability": "post_proxy2b1",
        "orig_id": "postId"
    }

    def __init__(
        self,
        sv_file: str,
        map_file: str,
        post_file: str,
        output_file: Optional[str] = None,
        rename_map: Optional[dict] = None,
    ):
        """
        :param sv_file: sv_estimate_result.txt 的完整路径
        :param map_file: id_mapping.csv 的完整路径（包含 internal_id, orig_id, type 列）
        :param post_file: 原始 post.csv 路径
        :param output_file: 输出 CSV 路径（默认放在 sv_file 同目录下，名为 post_with_estimate.csv）
        :param rename_map: 可选的列重命名映射（默认会替换 ML 列为 post_*，并把 orig_id->postId）
        """
        self.sv_file = sv_file
        self.map_file = map_file
        self.post_file = post_file
        if output_file is None:
            base = os.path.dirname(sv_file) or "."
            output_file = os.path.join(base, "post_with_estimate.csv")
        self.output_file = output_file

        self.rename_map = rename_map or self.DEFAULT_RENAME_MAP

        # parsed data
        self.idmap_df: Optional[pd.DataFrame] = None
        self.sv_df: Optional[pd.DataFrame] = None
        self.posts_map_df: Optional[pd.DataFrame] = None
        self.posts_df: Optional[pd.DataFrame] = None
        self.result_df: Optional[pd.DataFrame] = None
        self.nonzero_df: Optional[pd.DataFrame] = None

    # -------------------------
    # 辅助：更稳健的浮点解析
    # -------------------------
    @staticmethod
    def _safe_parse_float(val_str: str):
        """
        尝试把 val_str 解析成 float。支持：
          - 标准小数与科学计数法（3.45e-05, 1.10788e+06）
          - 含 + 符号的指数
          - 千位分隔符逗号 (1,234.56)
          - 替换常见 Unicode 几种负号 / 破损的符号
        返回 float 或 np.nan（解析失败）
        同时不抛异常，便于批处理。
        """
        if val_str is None:
            return float("nan")
        s = str(val_str).strip()

        if s == "":
            return float("nan")

        # 1) 用常见替换清理字符串（Unicode负号、长破折号等）
        s = s.replace("\u2212", "-")   # unicode minus
        s = s.replace("−", "-")        # other minus char
        s = s.replace("—", "-")
        s = s.replace("–", "-")
        s = s.replace("\u00A0", "")    # NBSP
        s = s.replace(" ", "")         # remove embedded spaces (千分符或格式问题)

        # 2) 去掉千分分隔符（逗号），但保留 e/E 和 ± 符号
        #    注意：某些 locale 会把小数点替换为 ',' —— 这里假定文件使用 '.' 作为小数点
        s = s.replace(",", "")

        # 3) 有时文件包含附加标识（如 "*", "%", "NA"），尽量处理
        s = s.rstrip("*").rstrip("%")

        # 4) 最后尝试直接 float()
        try:
            return float(s)
        except Exception:
            # 5) 尝试用 pandas 解析（更宽松）
            try:
                val = pd.to_numeric(s, errors="coerce")
                if pd.isna(val):
                    return float("nan")
                else:
                    return float(val)
            except Exception:
                # 6) fallback: log once (但不要阻塞)
                # print(f"[WARN] _safe_parse_float: cannot parse value '{val_str}' -> set NaN")
                return float("nan")

    # -------------------------
    # Step 1: 读取 id 映射
    # -------------------------
    def load_idmap(self):
        if not os.path.exists(self.map_file):
            raise FileNotFoundError(f"id map not found: {self.map_file}")
        self.idmap_df = pd.read_csv(self.map_file, dtype=str)
        # 确保 internal_id 为 int（原脚本这样做）
        if "internal_id" in self.idmap_df.columns:
            try:
                self.idmap_df["internal_id"] = self.idmap_df["internal_id"].astype(int)
            except Exception:
                # 如果不能转 int，保留为字符串 — 但后面合并时要注意类型一致
                pass

        # 选择 Post 类型映射
        if "type" in self.idmap_df.columns:
            self.posts_map_df = self.idmap_df[self.idmap_df["type"] == "Post"][["internal_id", "orig_id"]].copy()
        else:
            # 如果没有 type 列，则尝试由 label_int 或其它推断（退化处理）
            # 这里保守策略：把所有映射都当 Post
            self.posts_map_df = self.idmap_df[["internal_id", "orig_id"]].copy()

        print(f"[INFO] 载入映射表，共 {len(self.idmap_df)} 条，Post 类型 {len(self.posts_map_df)} 条")

    # -------------------------
    # Step 2: 解析单 Query 的 sv 文件（兼容简单格式）
    # -------------------------
    def parse_sv_file(self):
        """改进单文件解析，支持科学计数法等复杂格式"""
        if not os.path.exists(self.sv_file):
            raise FileNotFoundError(f"sv estimate file not found: {self.sv_file}")

        records = []
        line_re = re.compile(r"^\s*(\d+)\s+(.+?)\s*$")
        with open(self.sv_file, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("Query:") or line.startswith("All Est:"):
                    continue
                m = line_re.match(line)
                if m:
                    nid = int(m.group(1))
                    valstr = m.group(2)
                    est = self._safe_parse_float(valstr)
                    records.append({"internal_id": nid, "estimate": est})

        self.sv_df = pd.DataFrame(records)
        print(f"[INFO] parse_sv_file: 已读取 {len(self.sv_df)} 条节点估计结果")
    def merge_idmap_with_sv(self):
        if self.sv_df is None:
            raise RuntimeError("sv_df 未解析，请先调用 parse_sv_file()")
        if self.posts_map_df is None:
            raise RuntimeError("idmap 未加载，请先调用 load_idmap()")

        # 确保两边 internal_id 类型一致
        left = self.posts_map_df.copy()
        right = self.sv_df.copy()

        # 如果 left.internal_id 是 str (csv读取为 str)，把右侧转为 str 再合并；如果是 int 则确保右侧为 int
        if left["internal_id"].dtype == object:
            right["internal_id"] = right["internal_id"].astype(str)
        else:
            # left likely int, ensure right int
            try:
                left["internal_id"] = left["internal_id"].astype(int)
            except Exception:
                # fallback to string merge
                left["internal_id"] = left["internal_id"].astype(str)
                right["internal_id"] = right["internal_id"].astype(str)

        merged = pd.merge(left, right, on="internal_id", how="inner")
        self.merged_map_sv = merged
        print(f"[INFO] 合并映射后：{len(merged)} 条有效 Post 节点")

    # -------------------------
    # Step 4: 合并原始 post.csv
    # -------------------------
    def merge_with_posts(self):
        if not os.path.exists(self.post_file):
            raise FileNotFoundError(f"post file not found: {self.post_file}")
        # 读取 posts
        self.posts_df = pd.read_csv(self.post_file, dtype=str, keep_default_na=False)
        # 判断 id 列名
        id_col = None
        for candidate in ["id:ID", "id", "postId", "post_id"]:
            if candidate in self.posts_df.columns:
                id_col = candidate
                break
        if id_col is None:
            # fallback 使用第一列
            id_col = self.posts_df.columns[0]
            print(f"[WARN] 未找到常见的 id 列名，使用第一列: {id_col}")

        # merged_map_sv 的 orig_id 可能为字符串或数字，统一类型以便合并
        merged = pd.merge(self.posts_df, self.merged_map_sv, left_on=id_col, right_on="orig_id", how="left")
        self.result_df = merged

        # 将 estimate 转为数字，无法匹配的设为 0（原脚本逻辑）
        self.result_df["estimate"] = pd.to_numeric(self.result_df.get("estimate", pd.Series([])), errors="coerce").fillna(0.0)

        print(f"[INFO] 合并 post 与估计结果后: 总 Post 行数={len(self.result_df)}")

    # -------------------------
    # Step 5: 仅保留 estimate>0 的项并重命名列
    # -------------------------
    def filter_and_rename(self):
        if self.result_df is None:
            raise RuntimeError("result_df 未准备好，请先调用 merge_with_posts()")

        self.nonzero_df = self.result_df[self.result_df["estimate"] > 0].copy()
        print(f"[INFO] 估计值非零的 Post 数: {len(self.nonzero_df)}")

        # 重命名列（如果列存在才重命名）
        available_rename = {k: v for k, v in self.rename_map.items() if k in self.nonzero_df.columns}
        if available_rename:
            self.nonzero_df.rename(columns=available_rename, inplace=True)

        # 如果没有 orig_id 被重命名为 postId，确保输出中存在 postId（有时 orig_id 已存在）
        if "postId" not in self.nonzero_df.columns:
            if "orig_id" in self.nonzero_df.columns:
                self.nonzero_df.rename(columns={"orig_id": "postId"}, inplace=True)
            else:
                # 尝试构造 postId：优先使用 posts_df 的 id 列
                for candidate in ["id:ID", "id", "postId", "post_id"]:
                    if candidate in self.posts_df.columns:
                        self.nonzero_df["postId"] = self.nonzero_df[candidate]
                        break

    # -------------------------
    # Step 6: 保存结果 & 输出统计
    # -------------------------
    def save_and_report(self):
        if self.nonzero_df is None:
            raise RuntimeError("nonzero_df 未准备好，请先调用 filter_and_rename()")
        # 保存
        self.nonzero_df.to_csv(self.output_file, index=False)
        print(f"[✅] 已生成仅包含非零估计值的 Post 文件: {self.output_file}")

        # 统计
        total_posts = len(self.result_df) if self.result_df is not None else 0
        nonzero = len(self.nonzero_df)
        ratio = nonzero / total_posts if total_posts else 0
        print("\n===== 📊 统计结果 =====")
        print(f"总 Post 数量: {total_posts}")
        print(f"估计值非零的 Post 数: {nonzero}")
        print(f"非零比率: {ratio:.4%}")
        if "estimate" in self.result_df.columns and not self.result_df["estimate"].isna().all():
            print("估计值范围（总体）：")
            print(self.result_df["estimate"].describe())

    # -------------------------
    # 高层 run()：按步骤执行全部流程
    # -------------------------
    def run(self):
        self.load_idmap()
        self.parse_sv_file()
        self.merge_idmap_with_sv()
        self.merge_with_posts()
        self.filter_and_rename()
        self.save_and_report()
        return self.nonzero_df

    # ---------------------------
    # Step A: 解析 SV 多 Query 文件
    # ---------------------------
    def parse_sv_multi(self, sv_path: str) -> pd.DataFrame:
        """
        解析 in_estimateW_result.txt（multi-query），返回 DataFrame:
          columns: ['query_index', 'query_path', 'query_basename', 'internal_id', 'estimate', 'all_est']
        保持 query_index 为出现顺序（0..）
        更稳健地处理科学计数法与奇怪字符。
        """
        if not os.path.exists(sv_path):
            raise FileNotFoundError(f"sv file not found: {sv_path}")

        recs = []
        current_query_path = None
        current_qbasename = None
        current_index = -1
        current_all_est = None

        # 更松的正则：第一列是整数 id，其余一并捕获为 value 字符串（包含科学计数法等）
        node_line_re = re.compile(r"^\s*(\d+)\s+(.+?)\s*$")

        with open(sv_path, "r") as f:
            for raw in f:
                line = raw.rstrip("\n")
                if not line or line.strip() == "":
                    continue
                if line.startswith("Query:"):
                    current_query_path = line.split("Query:", 1)[1].strip()
                    current_qbasename = os.path.basename(current_query_path)
                    current_index += 1
                    current_all_est = None
                elif line.startswith("All Est:"):
                    try:
                        # 尝试解析 All Est（也用 _safe_parse_float）
                        s = line.split("All Est:", 1)[1].strip()
                        current_all_est = float(self._safe_parse_float(s))
                    except Exception:
                        current_all_est = None
                else:
                    m = node_line_re.match(line)
                    if m and current_index >= 0:
                        nid = int(m.group(1))
                        valstr = m.group(2)
                        est = self._safe_parse_float(valstr)
                        recs.append({
                            "query_index": current_index,
                            "query_path": current_query_path,
                            "query_basename": current_qbasename,
                            "internal_id": nid,
                            "estimate": est,
                            "all_est": current_all_est
                        })
                    else:
                        # 无法解析的行忽略（但可选打印调试）
                        # print("[WARN] Unparsed sv line:", line)
                        pass

        df = pd.DataFrame.from_records(recs)
        if df.empty:
            print("[WARN] parse_sv_multi: no records parsed from sv file.")
        else:
            print(f"[INFO] parse_sv_multi: parsed {len(df)} node records across {df['query_index'].nunique()} queries.")
        return df

    # ---------------------------
    # Step B: 读取 infer node 列表（u1 / u2 / ...）
    # ---------------------------
    def read_infer_node_list(self,infer_node_path: str) -> List[str]:
        if not os.path.exists(infer_node_path):
            raise FileNotFoundError(f"infer node file not found: {infer_node_path}")
        with open(infer_node_path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        print(f"[INFO] read_infer_node_list: loaded {len(lines)} infer nodes (first 10): {lines[:10]}")
        return lines

    # ---------------------------
    # Step C: 合并到 post 表，生成 post_with_estimate.csv
    # ---------------------------
    def build_post_with_estimates(self,sv_df: pd.DataFrame, idmap_file: str, post_csv: str, out_csv: str) -> pd.DataFrame:
        """
        输入 sv_df (parse_sv_multi 输出)，将每个 query 的估计合并为 columns estimate__<basename>
        返回合并后的 DataFrame (posts merged with mapping & estimate columns)，并写出 out_csv。
        """
        # 读取 id map
        idmap = pd.read_csv(idmap_file, dtype=str)
        # 标准化列
        if "internal_id" not in idmap.columns:
            raise ValueError("id_mapping.csv must contain 'internal_id' column")
        if "orig_id" not in idmap.columns and "origId" not in idmap.columns:
            # 尝试其他名字或抛错
            if "orig_id" not in idmap.columns:
                # try lowercase
                if "origId" in idmap.columns:
                    idmap.rename(columns={"origId": "orig_id"}, inplace=True)
                else:
                    raise ValueError("id_mapping.csv must contain 'orig_id' column (or origId)")

        # only posts
        if "type" in idmap.columns:
            posts_map = idmap[idmap["type"].str.lower() == "post"][["internal_id", "orig_id"]].copy()
        else:
            # fallback: assume all are posts
            posts_map = idmap[["internal_id", "orig_id"]].copy()

        # read posts
        posts = pd.read_csv(post_csv, dtype=str, keep_default_na=False)
        # find id column
        id_col = None
        for candidate in ["id:ID", "id", "postId", "post_id"]:
            if candidate in posts.columns:
                id_col = candidate
                break
        if id_col is None:
            id_col = posts.columns[0]
            print(f"[WARN] build_post_with_estimates: post id column not found, fallback to {id_col}")

        # normalize to str for merges
        posts[id_col] = posts[id_col].astype(str)
        posts_map["orig_id"] = posts_map["orig_id"].astype(str)
        posts_map["internal_id"] = posts_map["internal_id"].astype(str)

        merged = pd.merge(posts, posts_map, left_on=id_col, right_on="orig_id", how="left")
        # ensure internal_id column exists in merged (as string)
        merged["internal_id"] = merged["internal_id"].astype(str)

        # For each query (by query_index order), create column
        # We want predictable column names: estimate__<query_index>__<basename> to avoid collisions
        q_order = sv_df[["query_index", "query_basename"]].drop_duplicates().sort_values("query_index")
        for _, row in q_order.iterrows():
            qi = int(row["query_index"])
            qbase = row["query_basename"]
            colname = f"estimate__{qi}__{qbase}"
            # get group
            grp = sv_df[sv_df["query_index"] == qi][["internal_id", "estimate"]].copy()
            grp["internal_id"] = grp["internal_id"].astype(str)
            # there might be duplicate internal_id entries (unlikely) — aggregate by sum
            grp = grp.groupby("internal_id", as_index=False)["estimate"].sum()
            # merge: left on merged.internal_id
            merged = pd.merge(merged, grp, on="internal_id", how="left", suffixes=("", f"_{qi}"))
            # after merge, 'estimate' column appended; but if multiple merges, column name may vary — find latest
            # safe approach: if 'estimate' exists and colname not present, rename last 'estimate' to colname
            if "estimate" in merged.columns and colname not in merged.columns:
                merged.rename(columns={"estimate": colname}, inplace=True)
            elif colname not in merged.columns:
                # fallback: try 'estimate_<qi>'
                tmpname = f"estimate_{qi}"
                if tmpname in merged.columns:
                    merged.rename(columns={tmpname: colname}, inplace=True)
            # ensure numeric and NaN->0
            merged[colname] = pd.to_numeric(merged[colname], errors="coerce").fillna(0.0)

        merged.to_csv(out_csv, index=False)
        print(f"[INFO] build_post_with_estimates: wrote {out_csv}, total rows={len(merged)}")
        return merged


if __name__ == "__main__":
    # 下面是将各种节点的csv数据所在文件夹，转换为Graphlib格式
    datasets_name = "parler_data"
    dataset_name = "dataset_one"
    CSV_BASE_DIR = f"/home/wangshuo/resource/datasets/{datasets_name}/{dataset_name}/csv_data"
    Graph_Lib_Dir = f"/home/wangshuo/resource/datasets/{datasets_name}/{dataset_name}/data_graph"
    # # 1. 将 CSV 数据转换为 GraphLib 格式，这个格式中边是带标签的
    converter = FastestGraphConverter(CSV_BASE_DIR,Graph_Lib_Dir)
    # converter.run_without_author_user_post()
    converter.remove_edge_labels()
    # converter.simplify_graph_merge_edges_update_degree()
    # BASE_DIR = "/home/wangshuo/projects/FaSTest-main/dataset/sv"
    # SV_FILE = os.path.join(BASE_DIR, "sv_estimate_result.txt")
    # MAP_DIR = "/home/wangshuo/resource/datasets/parler_data/30W_valid_user/10-10-5/structure_first"
    # MAP_FILE = os.path.join(MAP_DIR, "id_mapping.csv")
    # POST_FILE = os.path.join(MAP_DIR, "post.csv")
    # OUTPUT_FILE = os.path.join(BASE_DIR, "post_with_estimate1.csv")
    #
    # merger = FastestEstimateMerger(
    #     sv_file=SV_FILE,
    #     map_file=MAP_FILE,
    #     post_file=POST_FILE,
    #     output_file=OUTPUT_FILE
    # )
    # merged_df = merger.run()
    #
    # # 如果需要在后续步骤直接使用 DataFrame，可从 merged_df 获取
    # print("\n[INFO] 运行完成，前 5 行预览：")
    # print(merged_df.head())