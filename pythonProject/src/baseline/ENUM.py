import os
import re
import json
import zlib
import numpy as np
import pandas as pd
import polars as pl
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple
import sys
import argparse
from tqdm.auto import tqdm
import traceback
import logging  

PROB_THRESHOLD = 0.5

WORKER_ID_TO_META = None
WORKER_ORACLE_PROB_MAPS = None
WORKER_CORE_CFG = None
WORKER_SUM_CFG = None      
WORKER_ROLE_RULES = None
WORKER_PROB_THRESHOLD = None

WORKER_VALUE_MAP = None
WORKER_AGG_MODE = None
WORKER_SUM_ON = None
WORKER_SUM_COL = None

def stable_seed(text: str) -> int:
    return zlib.adler32(text.encode("utf-8")) & 0xFFFFFFFF


def load_role_resources(base_dir: str, role_rules: Dict[str, dict], agg_mode: str, sum_on: str, sum_col: str):
    idmap_path = os.path.join(base_dir, "data_graph", "id_mapping.csv")
    idmap_df = pl.read_csv(idmap_path, infer_schema_length=0)

    if not {"internal_id", "orig_id", "type"}.issubset(set(idmap_df.columns)):
        raise ValueError("id_mapping.csv 缺少 internal_id / orig_id / type 列")

    id_to_meta = {}
    for row in idmap_df.iter_rows(named=True):
        iid = str(row["internal_id"])
        id_to_meta[iid] = {
            "orig_id": str(row["orig_id"]),
            "type": str(row["type"]).lower(),
        }

    dfs = {}
    for role in role_rules.keys():
        path = os.path.join(base_dir, "csv_data", f"{role}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到所需的属性文件: {path}")
        df = pl.read_csv(path, infer_schema_length=0)
        if "id:ID" in df.columns:
            df = df.rename({"id:ID": "orig_id"})
        dfs[role] = df

    oracle_prob_maps = {}
    for role, cfg in role_rules.items():
        oracle_col = cfg["oracle_col"]
        df = dfs[role]

        if oracle_col not in df.columns:
            raise ValueError(f"{role}.csv 缺少列 {oracle_col}")

        subset = df.select(["orig_id", oracle_col])
        probs = {}
        for row in subset.iter_rows(named=True):
            oid = str(row["orig_id"])
            val = row[oracle_col]
            try:
                probs[oid] = float(val) if val is not None else 0.0
            except Exception:
                probs[oid] = 0.0
        oracle_prob_maps[role] = probs

    value_map = {}
    if agg_mode == "sum":
        target_df = dfs.get(sum_on)
        if target_df is None:
            raise ValueError(f"sum_on 设置为 {sum_on}，但在 role_rules 中未定义。")
        if sum_col not in target_df.columns:
            raise ValueError(f"{sum_on}.csv 缺少作为 sum_col 的列: {sum_col}")
        
        subset = target_df.select(["orig_id", sum_col])
        for row in subset.iter_rows(named=True):
            oid = str(row["orig_id"])
            val = row[sum_col]
            try:
                value_map[oid] = float(val) if val is not None else 0.0
            except Exception:
                value_map[oid] = 0.0

    return id_to_meta, oracle_prob_maps, value_map

def load_json_config(file_path: str):
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r") as f:
        return json.load(f)

def locate_gt_path(gt_dir: str, qbase: str):
    candidates = [
        os.path.join(gt_dir, f"{qbase}_matches.csv"),
        os.path.join(gt_dir, f"{qbase}.graph_matches.csv"),
        os.path.join(gt_dir, f"{qbase}.matches.csv"),
        os.path.join(gt_dir, qbase),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    for fname in os.listdir(gt_dir):
        p = os.path.join(gt_dir, fname)
        if qbase in fname and os.path.isfile(p):
            return p
    return None

def resolve_match_cols(qbase: str, gt_columns: List[str], config: Dict):
    if config is None:
        return [c for c in gt_columns if re.fullmatch(r"u\d+", c)]
        
    qconf = config.get(qbase)
    if qconf is None and qbase.endswith(".graph"):
        qconf = config.get(qbase[:-6])

    if qconf is None:
        return [c for c in gt_columns if re.fullmatch(r"u\d+", c)]

    vids = []
    qconf_int = {int(k): v for k, v in qconf.items()}
    for _, vlist in sorted(qconf_int.items(), key=lambda x: x[0]):
        if isinstance(vlist, list):
            vids.extend(int(v) for v in vlist)
        else:
            vids.append(int(vlist))

    cols = [f"u{vid}" for vid in vids]
    return [c for c in cols if c in gt_columns]

def init_worker(base_dir: str, prob_threshold: float, agg_mode: str, sum_on: str, sum_col: str, sum_cfg_path: str, role_rules: dict):
    
    os.environ["POLARS_MAX_THREADS"] = "1"
    
    global WORKER_ID_TO_META, WORKER_ORACLE_PROB_MAPS, WORKER_CORE_CFG, WORKER_SUM_CFG
    global WORKER_ROLE_RULES, WORKER_PROB_THRESHOLD
    global WORKER_VALUE_MAP, WORKER_AGG_MODE, WORKER_SUM_ON, WORKER_SUM_COL

    WORKER_AGG_MODE = agg_mode
    WORKER_SUM_ON = sum_on
    WORKER_SUM_COL = sum_col

    WORKER_ID_TO_META, WORKER_ORACLE_PROB_MAPS, WORKER_VALUE_MAP = load_role_resources(
        base_dir, role_rules, agg_mode, sum_on, sum_col
    )
    
    WORKER_CORE_CFG = load_json_config(os.path.join(base_dir, "data_graph", "core_nodes_config.json"))
    
    if agg_mode == "sum" and sum_cfg_path:
        WORKER_SUM_CFG = load_json_config(sum_cfg_path)
    else:
        WORKER_SUM_CFG = None

    WORKER_ROLE_RULES = role_rules
    WORKER_PROB_THRESHOLD = prob_threshold

def eval_row_short_circuit(
    core_values: np.ndarray,
    id_to_meta: Dict[str, Dict[str, str]],
    oracle_prob_maps: Dict[str, Dict[str, float]],
    oracle_cache: Dict[Tuple[str, str], bool],
    budget_left: float,
    role_rules: Dict[str, dict],
    prob_threshold: float,
):
    items = []
    for raw_id in core_values:
        iid = str(raw_id) 
        meta = id_to_meta.get(iid)
        if meta is None: return None, {}, budget_left

        role = meta["type"]
        if role not in role_rules: continue

        items.append((role, meta["orig_id"], int(role_rules[role]["cost"])))


    role_names = list(role_rules.keys())
    role_priority = {r: i for i, r in enumerate(role_names)}
    items.sort(key=lambda x: (role_priority.get(x[0], 99), x[1]))

    calls = {r: 0 for r in role_names}

    for role, orig_id, cost in items:
        key = (role, orig_id)

        if key in oracle_cache:
            ok = oracle_cache[key]
        else:
            if budget_left < cost:
                return None, calls, budget_left

            prob = oracle_prob_maps[role].get(orig_id, 0.0)
            ok = prob > prob_threshold
            oracle_cache[key] = ok
            budget_left -= cost

            calls[role] += 1

        if not ok: return 0, calls, budget_left

    return 1, calls, budget_left

def process_one_query(qbase: str, tasks: List[dict], gt_dir: str, seed: int, max_trials: int, r1: str, r2: str):
    try:
        gt_path = locate_gt_path(gt_dir, qbase)
        if gt_path is None: return []

        try:
            preview_cols = pl.read_csv(gt_path, n_rows=1, infer_schema_length=0).columns
        except Exception: return []

        core_cols = resolve_match_cols(qbase, preview_cols, WORKER_CORE_CFG)
        if not core_cols: return []

        sum_cols = []
        if WORKER_AGG_MODE == "sum":
            if WORKER_SUM_CFG and (qbase in WORKER_SUM_CFG or qbase+".graph" in WORKER_SUM_CFG):
                cfg_entry = WORKER_SUM_CFG.get(qbase) or WORKER_SUM_CFG.get(qbase+".graph")
                vids = []
                for lbl, v_list in cfg_entry.items():
                    vids.extend(v_list)
                sum_cols = [f"u{vid}" for vid in vids]
            else:
                if len(core_cols) == 1:
                    sum_cols = [core_cols[0]]
                else:
                    return [] 

            sum_cols = [c for c in sum_cols if c in preview_cols]

        all_cols_to_load = list(set(core_cols + sum_cols))
        if not all_cols_to_load: return []

        try:
            exact_df = pl.read_csv(gt_path, columns=all_cols_to_load, infer_schema_length=0)
        except Exception: return []

        if exact_df.height == 0: return []
        
        
        exact_np = exact_df.to_numpy()
        N_exact = exact_np.shape[0]
        if N_exact == 0: return []

        col_name_to_idx = {name: i for i, name in enumerate(exact_df.columns)}
        core_indices = [col_name_to_idx[c] for c in core_cols]
        sum_indices = [col_name_to_idx[c] for c in sum_cols]

        T_true = float(tasks[0]["T_true"])
        q_seed = seed + stable_seed(qbase)
        active_max_trials = max_trials if max_trials is not None else max(100000, N_exact * 20)

        results = []
        for task in tasks:
            run_id = int(task["run_id"])
            budget_frac = float(task["budget_frac"])
            budget_n_out = int(task["budget_n"])
            B = float(task["B"])

            rng = np.random.default_rng(q_seed + run_id)
            oracle_cache = {}
            budget_left = B
            n_completed = 0
            sum_oracle = 0.0
            
            calls_total = {r1: 0, r2: 0}
            BATCH_SIZE = 10000 
            out_of_budget = False
            
            while not out_of_budget:
                if n_completed >= active_max_trials: break
                    
                indices = rng.integers(0, N_exact, size=BATCH_SIZE)
                budget_start_of_batch = budget_left
                
                for idx in indices:
                    row_values = exact_np[idx]
                    core_values = row_values[core_indices]
                    
                    oracle_val, calls, budget_left = eval_row_short_circuit(
                        core_values=core_values,
                        id_to_meta=WORKER_ID_TO_META,
                        oracle_prob_maps=WORKER_ORACLE_PROB_MAPS,
                        oracle_cache=oracle_cache,
                        budget_left=budget_left,
                        role_rules=WORKER_ROLE_RULES,
                        prob_threshold=WORKER_PROB_THRESHOLD,
                    )

                    if oracle_val is None:
                        out_of_budget = True
                        break

                    n_completed += 1
                    calls_total[r1] += calls.get(r1, 0)
                    calls_total[r2] += calls.get(r2, 0)
                    
                    if oracle_val == 1:
                        if WORKER_AGG_MODE == "sum":
                            val_to_add = 0.0
                            for s_idx in sum_indices:
                                iid = str(row_values[s_idx]) 
                                meta = WORKER_ID_TO_META.get(iid)
                                if meta and meta["type"] == WORKER_SUM_ON:
                                    val_to_add += WORKER_VALUE_MAP.get(meta["orig_id"], 0.0)
                            sum_oracle += val_to_add
                        else:
                            sum_oracle += 1.0
                    
                    if n_completed >= active_max_trials:
                        out_of_budget = True
                        break

                if not out_of_budget and (budget_left == budget_start_of_batch):
                    break

            T_hat = 0.0 if n_completed == 0 else float(N_exact) * float(sum_oracle) / float(n_completed)
            Qerror = abs(T_hat - T_true) / T_true if T_true != 0 else np.nan
            oracle_cost = calls_total[r1] + calls_total[r2]

            results.append({
                "query_basename": qbase,
                "run_id": run_id,
                "budget_frac": budget_frac,
                "budget_n": budget_n_out,
                "T_true": T_true,
                "T_hat": T_hat,
                "Qerror": Qerror,
                f"n_{r1}": calls_total[r1],
                f"n_{r2}": calls_total[r2],
                "oracle_cost": oracle_cost,
                "method": f"Exact_structureO",
            })

        return results
    except Exception as e:
        
        logging.error(f"[Worker 致命错误] {qbase} 处理失败:\n{traceback.format_exc()}")
        return []

def run_exact_structure_uniform_baseline(
    dataset_name: str,
    base_dir: str,
    role_rules: dict,
    r1: str,
    r2: str,
    source_alloc_csv: str = None,
    out_csv: str = None,
    target_method: str = "8_POSSA",
    target_budget_fracs: List[float] = None,
    c1: int = 50,
    c2: int = 20,
    prob_threshold: float = 0.5,
    seed: int = 42,
    max_trials: int = None,
    max_workers: int = None,
    agg_mode: str = "count",
    sum_on: str = None,
    sum_col: str = None,
    sum_nodes_config: str = None
):
    gt_dir = os.path.join(base_dir, "ground_truth", "structure_result")
    results_dir = os.path.join(base_dir, "results", "efficiency")

    if source_alloc_csv is None:
        if agg_mode == "sum":
            source_alloc_csv = os.path.join(results_dir, f"allocation_strategy_comparison_sum.csv")
        else:
            source_alloc_csv = os.path.join(results_dir, "allocation_strategy_comparison_count.csv")
            
    if out_csv is None:
        if agg_mode == "sum":
            out_csv = os.path.join(results_dir, f"Exact_structureO_budget_curve_sum.csv")
        else:
            out_csv = os.path.join(results_dir, "Exact_structureO_budget_curve_count.csv")

    if agg_mode == "sum" and not sum_nodes_config:
        default_sum_cfg = os.path.join(base_dir, "data_graph", "sum_nodes_config.json")
        if os.path.exists(default_sum_cfg):
            sum_nodes_config = default_sum_cfg
            logging.info(f"[*] 自动找到 sum_nodes_config: {sum_nodes_config}")
        else:
            logging.warning("[警告] 未提供 --sum_nodes_config，如果你的查询包含多个 sum 节点，结果将不准确。")

    t_true_filename = f"T_true_ML3_oracle2_probability_ML2_oracle1_probability_{agg_mode}.json"
    # t_true_filename = f"T_true_ML1_oracle2_probability_ML2_oracle2_probability_{agg_mode}.json"
    t_true_json_path = os.path.join(base_dir, "results", t_true_filename)
    
    if not os.path.exists(t_true_json_path):
        raise FileNotFoundError(f"找不到指定的 T_true 缓存文件: {t_true_json_path}")
        
    with open(t_true_json_path, "r") as f:
        t_true_map = json.load(f)
    logging.info(f"[*] 成功加载真实值配置文件: {t_true_filename} (包含 {len(t_true_map)} 个查询)")

    source_df = pd.read_csv(source_alloc_csv)
    source_df = source_df[source_df["method"] == target_method].copy()
    if source_df.empty:
        raise ValueError(f"在 {source_alloc_csv} 中找不到 method={target_method}")

    if target_budget_fracs is not None:
        source_df["budget_frac_float"] = source_df["budget_frac"].astype(float)
        mask = source_df["budget_frac_float"].apply(
            lambda x: any(np.isclose(x, bf, atol=1e-5) for bf in target_budget_fracs)
        )
        source_df = source_df[mask].drop(columns=["budget_frac_float"])

    source_df["budget_frac"] = source_df["budget_frac"].astype(float)
    
    col_n1 = f"n_{r1}" if f"n_{r1}" in source_df.columns else "n_post"
    col_n2 = f"n_{r2}" if f"n_{r2}" in source_df.columns else "n_comment"
    
    source_df["B"] = c1 * source_df[col_n1].astype(int) + c2 * source_df[col_n2].astype(int)

    tasks_by_query = {}
    for row in source_df.itertuples(index=False):
        qbase = str(row.query_basename)
        t_true_val = t_true_map.get(qbase, t_true_map.get(f"{qbase}.graph", 0.0))

        task = {
            "run_id": int(row.run_id),
            "budget_frac": float(row.budget_frac),
            "budget_n": int(row.budget_n),
            "B": float(row.B),
            "T_true": float(t_true_val),
        }
        tasks_by_query.setdefault(qbase, []).append(task)

    if max_workers is None:
        max_workers = max(1, min(len(tasks_by_query), (os.cpu_count() or 2) - 1))

    logging.info(f"[*] dataset={dataset_name}, queries={len(tasks_by_query)}, workers={max_workers}")
    logging.info(f"[*] agg_mode={agg_mode}, sum_on={sum_on}, sum_col={sum_col}")

    headers = [
        "query_basename", "run_id", "budget_frac", "budget_n",
        "T_true", "T_hat", "Qerror", f"n_{r1}", f"n_{r2}",
        "oracle_cost", "method"
    ]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(columns=headers).to_csv(out_csv, index=False)

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_worker,
        initargs=(base_dir, prob_threshold, agg_mode, sum_on, sum_col, sum_nodes_config, role_rules),
    ) as executor:
        future_to_qbase = {}
        for qbase, tasks in tasks_by_query.items():
            fut = executor.submit(
                process_one_query,
                qbase=qbase,
                tasks=tasks,
                gt_dir=gt_dir,
                seed=seed,
                max_trials=max_trials,
                r1=r1,
                r2=r2
            )
            future_to_qbase[fut] = qbase

        for fut in tqdm(as_completed(future_to_qbase), total=len(future_to_qbase), desc="Queries", dynamic_ncols=True):
            qbase = future_to_qbase[fut]
            try:
                recs = fut.result()
                if recs:
                    pd.DataFrame(recs, columns=headers).to_csv(out_csv, mode='a', header=False, index=False)
            except Exception as e:
                
                logging.error(f"[WARN] worker抛出异常: {qbase} | {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exact_structureO baseline (count/sum)")
    parser.add_argument("--dataset_name", type=str, default="amazon_extend") 
    parser.add_argument("--source_alloc_csv", type=str, default=None)
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--target_method", type=str, default="8_POSSA")
    parser.add_argument("--target_budget_fracs", type=str, default="0.01,0.05,0.075,0.1,0.125,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--c1", type=int, default=50)
    parser.add_argument("--c2", type=int, default=20)
    parser.add_argument("--prob_threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_trials", type=int, default=None)
    parser.add_argument("--max_workers", type=int, default=None)
    
    parser.add_argument("--agg_mode", type=str, default="sum", choices=["count", "sum"])
    parser.add_argument("--sum_on", type=str, default="product", help="Column name mapping (e.g., 'post' or 'product')")
    parser.add_argument("--sum_col", type=str, default="price", help="Column name to sum, e.g., 'upvotes' or 'price'")
    parser.add_argument("--sum_nodes_config", type=str, default=None)
    
    args = parser.parse_args()

    
    if "amazon" in args.dataset_name.lower():
        data_domain = "amazon_data"
        r1, r2 = "product", "review"
        if args.agg_mode == "sum" and not args.sum_on:
            args.sum_on = "product"
            args.sum_col = "average_rating"
    else:
        data_domain = "parler_data"
        r1, r2 = "post", "comment"
        if args.agg_mode == "sum" and not args.sum_on:
            args.sum_on = "post"
            args.sum_col = "upvotes"

    BASE_DIR = f"/home/wangshuo/resource/datasets/{data_domain}/{args.dataset_name}"
    
    
    log_dir = os.path.join(BASE_DIR, "results", "efficiency")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"exact_structureO_run_{args.agg_mode}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, encoding='utf-8'),  
            logging.StreamHandler(sys.stdout)                      
        ]
    )
    
    logging.info("=" * 60)
    logging.info(f"Task Started: dataset={args.dataset_name}, agg_mode={args.agg_mode}")
    logging.info(f"Log file will be saved to: {log_file_path}")
    logging.info("=" * 60)
    
    DYNAMIC_ROLE_RULES = {
        r1: {"cost": args.c1, "oracle_col": "ML3_oracle2_probability"},
        r2: {"cost": args.c2, "oracle_col": "ML2_oracle1_probability"},
    }

    # DYNAMIC_ROLE_RULES = {
    #     r1: {"cost": args.c1, "oracle_col": "ML1_oracle2_probability"},
    #     r2: {"cost": args.c2, "oracle_col": "ML2_oracle2_probability"},
    # }

    if args.target_budget_fracs.strip() == "":
        fracs = None
    else:
        fracs = [float(x.strip()) for x in args.target_budget_fracs.split(",") if x.strip()]
        
    if args.agg_mode == "sum" and (args.sum_on is None or args.sum_col is None):
        parser.error("--agg_mode='sum' 要求必须同时指定 --sum_on 和 --sum_col")

    run_exact_structure_uniform_baseline(
        dataset_name=args.dataset_name,
        base_dir=BASE_DIR,
        role_rules=DYNAMIC_ROLE_RULES,
        r1=r1,
        r2=r2,
        source_alloc_csv=args.source_alloc_csv,
        out_csv=args.out_csv,
        target_method=args.target_method,
        target_budget_fracs=fracs,
        c1=args.c1,
        c2=args.c2,
        prob_threshold=args.prob_threshold,
        seed=args.seed,
        max_trials=args.max_trials,
        max_workers=args.max_workers,
        agg_mode=args.agg_mode,
        sum_on=args.sum_on,
        sum_col=args.sum_col,
        sum_nodes_config=args.sum_nodes_config
    )