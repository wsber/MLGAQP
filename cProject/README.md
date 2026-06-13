## FaSTest Extended: Projection Sampling & Oracle Verification
**Based on the paper: *Cardinality Estimation of Subgraph Matching: A Filtering-Sampling Approach* (VLDB 2024)**

This repository extends the original FaSTest framework to support complex query aggregations (`COUNT` and `SUM`) over heterogeneous graphs with machine learning (ML) predicates. It implements both **our proposed method** and an **extended baseline** for comprehensive evaluation:

1. **Projection Sampling & Weight Estimation (Our Proposed Method):** Computes fine-grained frequencies and aggregations for core projection instances (`EstimateCoreInstances`, `EstimateCoreInstancesAgg`) guided by `core_nodes_config.json`.
2. **FaSTest-Oracle (Extended Baseline):** Integrates Oracle-guided tree sampling (`EstimateWithMultiPredicate`, `EstimateWithMultiPredicateAgg`) to verify multi-predicates under strict sampling budgets.

### Dependencies
- Boost Library
- g++ Compiler with C++20 support

### Build Instructions
Build the project using CMake:
```sh
mkdir build
cd build
cmake .. && make  
```

### Arguments Overview

| Argument | Description |
| :--- | :--- |
| **Basic Parameters** | |
| `-d` | Dataset name (e.g., `dataset_test`, `amazon_extend`). |
| `--STRUCTURE X / 3 / 4`| Substructure filter condition: No filter (X), Triangle Safety (3), Triangle & Four-cycle Safety (4). |
| `--ROOT_LABEL` | Root vertex label index used for tree sampling. |
| `--SAMPLE_BUDGET` | The maximum structural sampling budget for tree generation. |
| **Aggregation Parameters** | *(Used by both Proposed Method and Baseline)* |
| `--AGG_FUNC` | Aggregation type: `count` or `sum`. |
| `--SUM_TABLE` | The target CSV file containing the numerical attribute to sum. |
| `--SUM_COL` | The target column name for the `SUM` aggregation (e.g., `price`, `upvotes`). |
| `--SUM_LABEL` | The specific vertex label in the query graph that the `SUM` operates on. |
| **Oracle Baseline Mode** | *(Triggers FaSTest-Oracle)* |
| `--ESTIMATE_WITH_PREDICATE`| **Enable FaSTest-Oracle baseline mode**. If omitted, runs the proposed projection sampling. |
| `--ORACLE_TABLE1 / 2` | The CSV file names storing the ML proxy/oracle probabilities. |
| `--POST_ORACLE_COL` | The column name in Table 1 containing predicate probabilities. |
| `--COMMENT_ORACLE_COL` | The column name in Table 2 containing predicate probabilities. |
| `--MULTI_PROXY_PROB` | The column storing the multi-proxy probabilities. |
| **Budget Curve Evaluation** | *(For Baseline Testing)* |
| `--FASTESTO_BUDGET_CURVE` | Enable budget curve evaluation mode (running across different budget fractions). |
| `--BUDGET_CURVE_IN` | Path to the input allocation strategy CSV to align Oracle costs. |
| `--FASTESTO_BUDGET_CURVE_OUT`| Output path for the generated evaluation curve CSV. |
| `--FASTESTO_RUNS` | Number of independent runs for reliable averaging (e.g., `5`). |

---

### Execution Examples

#### Part 1: Projection Sampling & Weight Estimation (Our Proposed Method)
By omitting the `--ESTIMATE_WITH_PREDICATE` flag, the framework automatically reads `core_nodes_config.json` and executes our proposed projection sampling (`EstimateCoreInstances` or `EstimateCoreInstancesAgg`).

**COUNT Estimation:**
```sh
./Fastest -d dataset_three --SAMPLE_BUDGET 30000 --AGG_FUNC count
```

**SUM Estimation:**
```sh
./Fastest -d dataset_three --SAMPLE_BUDGET 30000 \
  --AGG_FUNC sum --SUM_TABLE post --SUM_COL upvotes --SUM_LABEL 1
```

---

#### Part 2: FaSTest-Oracle (Baseline Evaluation)
By adding `--ESTIMATE_WITH_PREDICATE`, the framework switches to the FaSTest-Oracle baseline, applying Oracle verifications iteratively during the sampling phase. 

**1. Parler Dataset (Single Predicate)**
*COUNT Estimation:*
```sh
./Fastest -d dataset_three --ROOT_LABEL 1 --SAMPLE_BUDGET 30000 \
  --ESTIMATE_WITH_PREDICATE \
  --POST_ORACLE_COL ML1_oracle2_probability \
  --COMMENT_ORACLE_COL ML2_oracle2_probability \
  --AGG_FUNC count \
  --MULTI_PROXY_PROB ML1_proxy4b_probability \
  --BUDGET_CURVE_IN /home/wangshuo/resource/datasets/parler_data/dataset_three/results/efficiency/allocation_strategy_comparison_count.csv \
  --FASTESTO_BUDGET_CURVE --FASTESTO_RUNS 5 \
  --FASTESTO_BUDGET_CURVE_OUT /home/wangshuo/resource/datasets/parler_data/dataset_three/results/efficiency/FastestO_budget_curve_count.csv
```
*SUM Estimation:*
```sh
./Fastest -d dataset_three --ROOT_LABEL 1 --SAMPLE_BUDGET 30000 \
  --ESTIMATE_WITH_PREDICATE \
  --POST_ORACLE_COL ML1_oracle2_probability \
  --COMMENT_ORACLE_COL ML2_oracle2_probability \
  --AGG_FUNC sum --SUM_TABLE post --SUM_COL upvotes --SUM_LABEL 1 \
  --MULTI_PROXY_PROB ML1_proxy4b_probability \
  --BUDGET_CURVE_IN /home/wangshuo/resource/datasets/parler_data/dataset_three/results/efficiency/allocation_strategy_comparison_sum.csv \
  --FASTESTO_BUDGET_CURVE --FASTESTO_RUNS 5 \
  --FASTESTO_BUDGET_CURVE_OUT /home/wangshuo/resource/datasets/parler_data/dataset_three/results/efficiency/FastestO_budget_curve_sum.csv
```

**2. Parler-E Dataset (Multi-Predicate)**
*COUNT Estimation:*
```sh
./Fastest -d dataset_test --ROOT_LABEL 1 --SAMPLE_BUDGET 30000 \
  --ESTIMATE_WITH_PREDICATE \
  --POST_ORACLE_COL ML1_oracle2_probability \
  --COMMENT_ORACLE_COL ML2_oracle2_probability \
  --AGG_FUNC count \
  --MULTI_PROXY_PROB ML1_proxy4b_probability \
  --BUDGET_CURVE_IN /home/wangshuo/resource/datasets/parler_data/dataset_test/results/efficiency/allocation_strategy_comparison_count.csv \
  --FASTESTO_BUDGET_CURVE --FASTESTO_RUNS 5 \
  --FASTESTO_BUDGET_CURVE_OUT /home/wangshuo/resource/datasets/parler_data/dataset_test/results/efficiency/FastestO_budget_curve_count.csv
```
*SUM Estimation:*
```sh
./Fastest -d dataset_test --ROOT_LABEL 2 --SAMPLE_BUDGET 30000 \
  --ESTIMATE_WITH_PREDICATE \
  --POST_ORACLE_COL ML1_oracle2_probability \
  --COMMENT_ORACLE_COL ML2_oracle2_probability \
  --AGG_FUNC sum --SUM_TABLE post --SUM_COL upvotes --SUM_LABEL 2 \
  --MULTI_PROXY_PROB ML1_proxy4b_probability \
  --BUDGET_CURVE_IN /home/wangshuo/resource/datasets/parler_data/dataset_test/results/efficiency/allocation_strategy_comparison_sum.csv \
  --FASTESTO_BUDGET_CURVE --FASTESTO_RUNS 2 \
  --FASTESTO_BUDGET_CURVE_OUT /home/wangshuo/resource/datasets/parler_data/dataset_test/results/efficiency/FastestO_budget_curve_sum.csv
```

**3. Amazon-Extend Dataset (Multi-Predicate Heterogeneous Graphs)**
*For heterogeneous graphs like Amazon, explicitly declare the Oracle tables using `--ORACLE_TABLE1` and `--ORACLE_TABLE2`.*

*COUNT Estimation:*
```sh
./Fastest -d amazon_extend --ROOT_LABEL 1 --SAMPLE_BUDGET 30000 \
  --ESTIMATE_WITH_PREDICATE \
  --ORACLE_TABLE1 product --POST_ORACLE_COL ML3_oracle2_probability \
  --ORACLE_TABLE2 review --COMMENT_ORACLE_COL ML2_oracle1_probability \
  --AGG_FUNC count \
  --MULTI_PROXY_PROB ML3_proxy2_probability \
  --BUDGET_CURVE_IN /home/wangshuo/resource/datasets/amazon_data/amazon_extend/results/efficiency/allocation_strategy_comparison_count.csv \
  --FASTESTO_BUDGET_CURVE --FASTESTO_RUNS 5 \
  --FASTESTO_BUDGET_CURVE_OUT /home/wangshuo/resource/datasets/amazon_data/amazon_extend/results/efficiency/FastestO_budget_curve_count.csv
```
*SUM Estimation:*
```sh
./Fastest -d amazon_extend --ROOT_LABEL 1 --SAMPLE_BUDGET 60000 \
  --ESTIMATE_WITH_PREDICATE \
  --ORACLE_TABLE1 product --POST_ORACLE_COL ML3_oracle2_probability \
  --ORACLE_TABLE2 review --COMMENT_ORACLE_COL ML2_oracle1_probability \
  --AGG_FUNC sum --SUM_TABLE product --SUM_COL price --SUM_LABEL 12 \
  --MULTI_PROXY_PROB ML3_proxy2_probability \
  --BUDGET_CURVE_IN /home/wangshuo/resource/datasets/amazon_data/amazon_extend/results/efficiency/allocation_strategy_comparison_sum.csv \
  --FASTESTO_BUDGET_CURVE --FASTESTO_RUNS 5 \
  --FASTESTO_BUDGET_CURVE_OUT /home/wangshuo/resource/datasets/amazon_data/amazon_extend/results/efficiency/FastestO_budget_curve_sum.csv
```

---

### Datasets Input Format
The datasets and query graphs use the format derived from [RapidMatch](https://github.com/RapidsAtHKUST/RapidMatch/).
```
t [#Vertex] [#Edge]
v [ID] [Label] [Degree]
v [ID] [Label] [Degree]
...
e [SourceID] [TargetID] [EdgeLabel]
```
*(Note: Edge labels are optional; missing edge labels are treated as zero).*

#### Ground Truth Cardinalities
To accurately assess the estimation quality, the framework compares predictions against exact true cardinalities (typically generated using [DAF](https://github.com/SNUCSE-CTA/DAF) or exact matchers). The framework extracts `T_true` automatically from JSON/TXT mappings placed in your dataset's `results`  directories.
