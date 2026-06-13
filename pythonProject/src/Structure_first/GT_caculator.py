import os
import json
import glob
import sys
from tqdm import tqdm

project_root = "/home/wangshuo/projects/Neo4j_Exp"
if project_root not in sys.path:
    sys.path.append(project_root)
from pythonProject.src.Structure_first.compute_truth import GroundTruthManager

# ==========================================
# 1. 用户配置区 (按需修改)
# ==========================================
agg_mode = "count"  # [新增切换开关]: "count" 或 "sum"

parent_dataset = "amazon_data"
dataset_name = "amazon_extend"
table1 = "product"   # 映射第一张表 (原post)
table2 = "review"    # 映射第二张表 (原comment)

post_oracle_col = "ML3_oracle1_probability"
comment_oracle_col = "ML2_oracle1_probability"

# [SUM 模式参数]: 仅当 agg_mode == "sum" 时生效
sum_on = table1      # 对 product 求和
sum_col = "upvotes"  # 你要 SUM 的列

# 针对当前数据集设置目标表对应的节点标签列表
# (例如 user=2, product=10~14)
target_labels = [10, 11, 12, 13, 14]

# ==========================================
# 2. 初始化与数据加载
# ==========================================
gt = GroundTruthManager(
    dataset_name=dataset_name,
    post_oracle_col=post_oracle_col,
    comment_oracle_col=comment_oracle_col,
    parent_dataset=parent_dataset,
    table1=table1,   # 传入新修改支持的表名
    table2=table2    # 传入新修改支持的表名
)

# 读取 core_nodes_config.json
core_path = os.path.join(gt.base_path, "data_graph", "core_nodes_config.json")
with open(core_path, "r") as f:
    core = json.load(f)

# 根据 agg_mode 预加载源数据
if agg_mode == "sum":
    source_data = gt._load_and_prepare_sources(agg_mode="sum", sum_on=sum_on, sum_col=sum_col)
else:
    source_data = gt._load_and_prepare_sources(agg_mode="count")

# 获取所有匹配文件
gt_files = sorted(glob.glob(os.path.join(gt.gt_dir, "*_matches.csv")))
all_T_true = {}

# ==========================================
# 3. 循环计算 GT
# ==========================================
for gt_path in tqdm(gt_files, desc=f"GT {agg_mode.upper()}"):
    qbase_graph = os.path.basename(gt_path).replace("_matches.csv", "")
    qconf = core.get(qbase_graph)
    
    if not qconf:
        all_T_true[qbase_graph] = 0.0
        continue

    # 在 core_nodes_config 中搜集当前查询图对应的目标节点映射
    target_uids = []
    for lbl in target_labels:
        uids = qconf.get(str(lbl)) or qconf.get(int(lbl))
        if uids:
            target_uids.extend(uids)

    # SUM 模式下计算一定要有目标节点来取值；COUNT 模式下只要匹配通过即可
    if not target_uids and agg_mode == "sum":
        raise ValueError(f"{qbase_graph}: 没有在核心配置中找到属于 {target_labels} 的计算目标！")

    sum_match_cols = [f"u{int(uid)}" for uid in target_uids] if target_uids else None

    # 执行底层 T_true 计算
    t_val = gt._compute_multi_predicate_polars(
        gt_path=gt_path,
        core_nodes_config=core,      
        source_data=source_data,
        prob_threshold=0.5,
        agg_mode=agg_mode,
        sum_on=sum_on if agg_mode == "sum" else None,
        sum_col=sum_col if agg_mode == "sum" else None,
        sum_match_col=sum_match_cols,
    )
    all_T_true[qbase_graph] = float(t_val)

# ==========================================
# 4. 导出 JSON
# ==========================================
base, ext = os.path.splitext(gt.cache_path)

if agg_mode == "sum":
    safe_sum_col = str(sum_col).replace("/", "_").replace(":", "_")
    out_path = f"{base}_sum_{sum_on}_{safe_sum_col}{ext}"
else:
    # count模式的缓存名直接加 _count 或保留默认
    out_path = f"{base}_count{ext}"

os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(all_T_true, f, indent=4)

print(f"[DONE] {agg_mode.upper()} GT 已保存到: {out_path}")
print(f"[INFO] 共处理 queries: {len(all_T_true)}")