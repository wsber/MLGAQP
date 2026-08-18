以下是为您翻译的英文文档，已按照学术开源代码库的专业规范与习惯表达进行微调：

***

# Proxy-Guided Sampling for Approximate Graph Aggregation with Machine Learning Predicates

# Evaluation & Reproduction Guide (Continuously Updated...)

This repository provides the complete code and configuration required to reproduce all experimental results reported in the paper. Users can choose to **fully automate the reproduction via a master script** or **execute the pipeline step-by-step by module**.

> **Overhead Notice:** Directly invoking Large Language Models or Deep Neural Networks (Oracle models) for real-time inference incurs substantial computational and time costs. To facilitate rapid reproduction and verification, **we have pre-cached all Oracle and Proxy verification results for all queries into the `csv_data/` directory of each dataset, enabling ready-to-use execution.**

---

> ### (1) Instant Verification: 1-Minute Figure Reproduction
> **No need to wait for time-consuming stratified importance sampling and graph matching!** We have fully persisted the ground-truth evaluation results of all paper experiments under the `datasets/{workload}/results/efficiency/` directory for each dataset.
> If you wish to **immediately verify and reproduce all experimental figures (RQ1–RQ4) and statistical significance metrics from the paper**:
> 1. Launch Jupyter Notebook:
>    ```bash
>    jupyter notebook pythonProject/src/RQS/RQX.ipynb
>    ```
> 2. Click **`Run`** to directly load the pre-cached data and instantly render all high-resolution vector figures presented in the paper!

---

> ### (2) Technical Report & Supplementary Theoretical Proofs
> For details regarding the **Case Study** and the **complete theoretical proofs for uniform tree sampling (Tree Sampling Proofs)** mentioned in the paper, please refer to the technical report in the repository root directory: [**`TR.pdf`**](./TR0.1.pdf).

> ### (3) Codebase Design & Workflow Recommendations
> 1. **Reproducibility & Debugging Design**: The codebase is currently in its academic open-source stage. To facilitate state verification at each stage, algorithm logic debugging, and support a complete data reproduction pipeline, the code retains **detailed intermediate result disk-persistence and I/O validation logic**.
> 2. **Recommended Execution Strategy**:
>    * **Step 1 (Structural Projection & Weight Materialization)**: Run `Projection_Sampling_and_Weight_Estimation_Runner.py`. This **only needs to be executed once** for each workload and aggregation type (`COUNT` / `SUM`). This step fixes the topological projection space $\hat{\Psi}$ and offline-materializes the structural extension weights $\hat{w}(\psi)$ for all projections.
>    * **Step 2 (Sampling Algorithm & Baseline Evaluation)**: Upon completion of the materialized $\hat{\Psi}$, you can efficiently run `Proxy_Guided_Stratified_Importance_Sampling_Runner.py` and various baseline scripts multiple times to rapidly evaluate different sampling strategies, ablation variants, and multi-run stochastic variance.
> 3. **Ongoing Refactoring Plan**: Our team will continuously modularize and decouple the codebase, streamline I/O workflows, and optimize execution efficiency. Stay tuned for future updates.

---

## (4) Environment Setup

Before running the code, please follow the steps below to set up the Python virtual environment and compile the underlying C++ sampling engine.

### 1. Python Environment (Conda)
Python 3.8+ is recommended:
```bash
# 1. Create and activate conda virtual environment
conda create -n iogs python=3.8 -y
conda activate iogs

# 2. Install all dependencies in one click
pip install -r requirements.txt
```

### 2. C++ Sampling Engine Compilation
The underlying graph matching and candidate space tree sampling engine is developed in C++20, depending on CMake, Boost, and GSL:
```bash
# 1. Install system dependencies (Ubuntu/Debian)
conda install -c conda-forge cmake gxx_linux-64 boost gsl -y

# 2. Compile to generate the Fastest binary executable (Pre-compiled binaries are included; the steps below demonstrate recompilation)
cd cProject
mkdir -p build && cd build
cmake ..
make -j$(nproc)
cd ../..
```
*Upon successful compilation, the binary executable will be located at `cProject/build/Fastest`.*

---

## 0. Experimental Foundation: Datasets & ML Predicate Architecture

Before running experiments, please familiarize yourself with the underlying graph datasets, synthetic query workloads, and machine learning predicate architectures.

---

### 0.1. Datasets & Query Workloads

Experiments are conducted on three real-world attributed/multimodal graph datasets:

1. **`Parler`:** A text-attributed social network dataset comprising **3 distinct vertex labels/types**: users (`user`), posts (`post`), and comments (`comment`).
2. **`Parler-E`:** Adapted from `Parler`. It escalates label sparsity by subdividing vertex types into disjoint sub-types, expanding the label set to **6 distinct vertex labels**. This drastically reduces query selectivity to evaluate performance in extreme low-selectivity scenarios.
3. **`Amazon`:** A multimodal heterogeneous graph containing users (`user`), textual reviews (`review`), and product images (`product`). By randomly partitioning vertex types into disjoint sub-types, it features **11 distinct vertex labels** to evaluate structural complexity and multimodal feature integration.

#### Query Generation & Aggregation Constraints
Query graphs $Q$ are generated via **Random Walks** on the data graph, with aggregate attributes and ML predicates attached:
* **Aggregation Constraints:** To ensure valid `SUM` and `AVG` aggregations over $Q$, each query must contain at least one numeric attribute:
  * **`Parler` / `Parler-E`:** The `upvotes` attribute of a `post` vertex.
  * **`Amazon`:** The `price` or `rating` attribute of a `product` vertex.
* **Query Scale & Predicate Configurations:**
  * **`Parler`:** Contains **245** single-predicate queries ($|V(Q)| \in [4, 8]$, $k=1$), with the predicate randomly assigned to 1 `post` or `comment` vertex.
  * **`Parler-E`:** Contains **115** multi-predicate composite queries ($|V(Q)| \in [4, 8]$, $k \ge 2$), with predicates simultaneously assigned to at least 1 `post` vertex and at least 1 `comment` vertex.
  * **`Amazon`:** Contains **750** multimodal composite multi-predicate queries ($|V(Q)| \in [3, 8]$, $k \ge 2$), with predicates assigned to at least 1 `product` image vertex and at least 1 `review` text vertex.

---

### 0.2. ML Predicates: Oracle vs. Proxy Models

Each atomic ML predicate $\mathcal{P}_i$ is assigned an **accurate Oracle model** (for exact, unbiased validation) and a **lightweight Proxy model** (for efficient approximate scoring and guiding stratified importance sampling):

| Dataset | Vertex Type | Predicate Semantics ($\mathcal{P}$) | Oracle Model (# Params) | Proxy Model (# Params) | Proxy $F_1$ | Proxy Speedup |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| **`Amazon`** | `product` | **Image Texture Classification**<br>*(wooden/plastic/metal/fabric/glass?)* | `siglip-so400m-patch14-384`<br>*(878M)* | `siglip-base`<br>*(84M)* | 0.7546 | **$26.8\times$** |
| | `review` | **Sentiment Analysis**<br>*(positive/negative?)* | `roberta-large-sst2`<br>*(355M)* | `bert-mini-finetuned-sst2`<br>*(11M)* | 0.8890 | **$22.1\times$** |
| **`Parler`** /<br>**`Parler-E`** | `post` | **Opinion Inference**<br>*(Support/Oppose Donald Trump?)* | `deberta-v2-xxlarge-mnli`<br>*(1.5B)* | `deberta-v3-base-mnli`<br>*(184M)* | 0.7720 | **$42.5\times$** |
| | `comment` | **Sentiment Analysis**<br>*(positive/negative?)* | `roberta-large-sst2`<br>*(355M)* | `bert-mini-finetuned-sst2`<br>*(11M)* | 0.7876 | **$22.1\times$** |

* **Proxy Quality Tiers ($M_{P1} \sim M_{P4}$):** To evaluate $\text{PROXY}$'s sensitivity to proxy accuracy (RQ3), we construct 4 proxy quality tiers per task by fine-tuning or adopting simpler architectures. Their relative $F_1$ scores decrease monotonically within $[0.65, 0.89]$, while inference speeds increase monotonically.

---

### 0.3. Hardware Setup

All experiments were conducted on a high-performance server with the following specifications:
* **Operating System:** Ubuntu 22.04 LTS
* **Processor (CPU):** Dual Intel(R) Xeon(R) Gold 6130 CPUs @ 2.10GHz
* **Memory (RAM):** 503 GB
* **Graphics Cards (GPU):** $4 \times$ NVIDIA GeForce RTX 3090 GPUs (24GB VRAM each)

> **⚠️ Notice:** Each workload contains hundreds of complex subgraph isomorphism and ML predicate evaluations. Running full exact matching takes hours and requires at least 100 GB of free disk space per workload. **We provide precomputed Ground Truth (GT) files for all workloads**, allowing you to bypass expensive full-graph matching and exhaustive Oracle evaluations.

---

### 0.4. Repository Structure

The project is structured with a high-performance **C++ sampling engine (`cProject`)** at the lower level and a **Python proxy-guided sampling framework (`pythonProject`)** at the upper level:

```text
PROXY/
├── cProject/                                   # [C++ Core Engine] CS construction, tree sampling, and semantic projection weight estimation
│   ├── build/                                  # Precompiled binary directory (contains the compiled 'Fastest' executable)
│   ├── driver/                                 # C++ entry point (subgraph-cardinality-estimation.cc)
│   ├── lib/                                    # Graph data structures, CS builder, uniform tree sampler, etc.
│   └── CMakeLists.txt                          # CMake configuration file
│
├── datasets/                                   # [Data & Results Storage] Data graphs, query graphs, and results for the three workloads
│   ├── parler/                                 # Parler single-predicate workload (data_graph / query_graph / ground_truth / results)
│   ├── parler-e/                               # Parler-E multi-predicate expanded workload
│   └── amazon/                                 # Amazon multimodal heterogeneous graph workload
│
├── Model/                                      # [ML Model Repository] Oracle and Proxy model weights and configs
│
├── pythonProject/                              # [Python Experimental Framework] PROXY sampling algorithms, baseline evaluations, and plotting
│   └── src/
│       ├── algorithms/                         # Core PROXY algorithm implementations
│       │   ├── exact_subgraph_match.py         # Exact subgraph isomorphism matching script
│       │   ├── compute_truth.py                # Ground Truth (GT) computation and predicate verification class
│       │   └── proxy_sample.py                 # Proxy-guided Stratified Sampling (POSSA) and ablation variants
│       │
│       ├── baseline/                           # Baseline algorithm library
│       │   ├── ...                             # Core baselines (ENUM, FASTEST-ORACLE, WEE, etc.)
│       │   └── ...                             # Comparative methods (e.g., PRO-ABAE, PSF cascade filter, etc.)
│       │
│       ├── runner/                             # Execution Runners
│       │   └── ...                             # Execution scripts interfacing C++ and Python modules
│       │
│       └── RQS/                                # Visualization and plotting scripts for paper figures (RQ1 ~ RQ4)
│
├── scripts/                                    # [Automation Scripts] One-click reproduction shell scripts (run_all_experiments.sh, etc.)
└── ...                                         # Auxiliary utility scripts and configuration files
```

---

## 1. One-Click Reproduction

You can reproduce the experimental results either by running the master all-in-one script across all datasets or by executing dedicated workload-specific one-click scripts.

---

### 1.1. Master Script (All Workloads & Aggregations)
To automatically run the entire pipeline (including C++ weight materialization and stratified sampling) across all three workloads (`Parler`, `Parler-E`, `Amazon`) and generate all data required for plotting:
<!-- 
```bash
# Ensure your conda environment is activated
conda activate iogs

# Grant execute permissions and run the master script
chmod +x scripts/run_all_experiments.sh
bash scripts/run_all_experiments.sh
``` -->

---

### 1.2. Workload-Specific One-Click Scripts
If you wish to evaluate or debug a specific dataset without running the entire multi-hour benchmark suite, dedicated one-click automation scripts are provided.

#### Example: Parler (`COUNT` Mode)
The script `run_parler_count.sh` automates the end-to-end workflow on the **Parler** workload in `COUNT` mode:
1. **Step 1 (Offline Projection & Materialization):** Invokes the C++ engine to perform uniform tree sampling, estimate projection extension weights $\hat{w}(\psi)$, and materialize the compact core instance space.
2. **Step 2 (Online POSSA Sampling):** Executes proxy-guided stratified importance sampling across budget gradients $\alpha \in [1\%, 90\%]$ with 5 independent runs per tick.

```bash
# 1. Activate conda environment
conda activate iogs
cd pythonProject/scripts
# 2. Grant execution permission
chmod +x *.sh


# 3. Execute the one-click script

./run_all_proxy_experiments.sh

OR

./run_parler_count.sh
./run_parler_sum.sh

./run_parler_e_count.sh
./run_parler_e_sum.sh

./run_amazon_count.sh
./run_amazon_sum.sh

./run_all_avg.sh
```

**Generated Output File:**
  ```text
  datasets/parler/results/efficiency/allocation_strategy_comparison_count.csv
  ```

---


---

## 2. Step-by-Step Pipeline

If you prefer to inspect individual pipeline stages, reproduce specific Research Questions (RQs), or execute standalone baselines, follow **Steps 2.1 through 2.6** sequentially.

### 2.1. Compute Ground Truth (EXACT)

Obtain exact query answers free of sampling noise by performing exhaustive subgraph matching followed by Oracle verification *(it is recommended to skip this step and directly use the provided GT files)*.

1. **Exact Subgraph Matching:** Run `exact_subgraph_match.py` to invoke the C++ engine to perform exact subgraph matching and save intermediate embeddings.
2. **Predicate Verification & Aggregation:** Run `EXACT.py` to evaluate Oracle predicates over matching embeddings and perform final aggregation (supports `agg_mode={count, sum}`).

```bash
python pythonProject/src/algorithms/EXACT.py --dataset dataset_test --agg_mode count
python pythonProject/src/algorithms/EXACT.py --dataset dataset_test --agg_mode sum
```
* **Output Files:** `results/T_true_*_count.json` and `results/T_true_*_sum.json`

---

### 2.2. $\text{PROXY}$ `count` / `sum` Experiments

Execute experimental evaluations for `count` and `sum` aggregations.

#### 2.2.1. Projection Weight Estimation & Aggregation
Decompose each query graph into the semantic projection space $\hat{\Psi}$, estimate structural extension weights $\hat{w}(\psi)$ via the C++ engine, and associate corresponding ML proxy/oracle probabilities:

* **Parler Dataset (`COUNT` Aggregation Example):**
```bash
python pythonProject/src/runner/Projection_Sampling_and_Weight_Estimation_Runner.py \
  --base_dir $(pwd) \
  --dataset parler \
  --sample_budget 60000 \
  --agg_func count \
  --table1 post \
  --table2 comment
```

* **Amazon Dataset (`SUM` Aggregation Example):**
```bash
python pythonProject/src/runner/Projection_Sampling_and_Weight_Estimation_Runner.py \
  --base_dir $(pwd) \
  --dataset amazon_extend \
  --sample_budget 60000 \
  --agg_func sum \
  --sum_table product \
  --sum_col price \
  --sum_label 12 \
  --table1 product \
  --table2 review
```

* **Intermediate Output:** `results/structure_estimate/*.csv` (raw instance files partitioned per query).
* **Final Output:** `results/aggregated_results/aggregated_list_*.csv` (materialized compact projection space containing weights $a$ and node ML probabilities).

#### 2.2.2. Core Performance & Ablation Studies (RQ1, RQ2 & RQ4)
Run Proxy-guided Stratified Importance Sampling (POSSA) and ablation variants over the materialized projection space (`aggregated_results/`):

* **For RQ1 & RQ2 (Core Performance Across Sampling Rates):**
  Evaluate $\text{PROXY}$ (POSSA) across a progressive budget gradient $\alpha \in [1\%, 90\%]$:

  * *Parler Dataset (`parler` / `dataset_three`):*
  ```bash
  python pythonProject/src/runner/Proxy_Guided_Stratified_Importance_Sampling_Runner.py \
    --parent_dataset parler \
    --dataset_name dataset_three \
    --target_ticks "0.01,0.05,0.075,0.1,0.125,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9" \
    --run_times 5 \
    --max_workers 16
  ```

  * *Parler-E Dataset (`parler-e` / `dataset_test`):*
  ```bash
  python pythonProject/src/runner/Proxy_Guided_Stratified_Importance_Sampling_Runner.py \
    --parent_dataset parler-e \
    --dataset_name dataset_test \
    --target_ticks "0.01,0.05,0.075,0.1,0.125,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9" \
    --run_times 5 \
    --max_workers 16
  ```

  * *Amazon Dataset (`amazon` / `amazon_extend`):*
  ```bash
  python pythonProject/src/runner/Proxy_Guided_Stratified_Importance_Sampling_Runner.py \
    --parent_dataset amazon \
    --dataset_name amazon_extend \
    --target_ticks "0.01,0.05,0.075,0.1,0.125,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9" \
    --run_times 5 \
    --max_workers 16
  ```
  * **Output File:** `datasets/<parent_dataset>/results/efficiency/allocation_strategy_comparison_{agg_mode}.csv`

* **For RQ4 (Component Ablation under Fixed Budget $\alpha=10\%$):**
  Evaluate component variants under a fixed sampling budget (`UN`: Uniform Sampling, `PO`: Proxy-Only, `WO`: Weight-Only, `MAB`: Multi-Armed Bandit, `POSSA`: Full Proposed Method):
  ```bash
  python pythonProject/src/runner/Proxy_Guided_Stratified_Importance_Sampling_Runner.py \
    --parent_dataset parler-e \
    --dataset_name dataset_test \
    --target_ticks "0.1" \
    --run_times 5 \
    --max_workers 16
  ```
  * **Output File:** `datasets/<parent_dataset>/results/efficiency/allocation_strategy_comparison_ablation_{agg_mode}.csv`

#### 2.2.3. Sensitivity & Proxy Quality Degradation Analysis (RQ3)
Evaluate algorithm robustness against proxy quality degradation and complex multi-predicate noise:
```bash
python pythonProject/src/runner/Sensitivity_single_predicate_Runner.py --dataset dataset_test
python pythonProject/src/runner/Sensitivity_multi_predicate_comparation.py --dataset dataset_test
```
* **Output File:** `results/efficiency/proxy_quality_ablation_{agg_mode}.csv`

---

### 2.3. Offline Ratio Synthesis for `avg` Queries (Theorem 6)
Based on the ratio estimator proposed in **Theorem 6** ($\hat{\tau}_{\text{avg}} = \hat{\tau}_{\text{sum}} / \hat{\tau}_{\text{count}}$), `avg` results do not require re-running the graph sampling engine; they are synthesized offline from completed `count` and `sum` evaluation data.

1. **Synthesizing Ground Truth JSON:**
   Calculate exact average values via $\tau_{\text{avg}} = \tau_{\text{sum}} / \tau_{\text{count}}$:
   * **Inputs:** `results/T_true_*_sum.json` and `results/T_true_*_count.json`
   * **Output:** Generates `results/T_true_*_avg.json`

2. **Synthesizing Result Curves & Error Alignment:**
   Merge `count` and `sum` CSV files via an inner join on `(query_basename, budget_frac, run_id)`:
   ```bash
   python pythonProject/src/baseline/synthesize_avg_results.py --dataset dataset_test
   ```
   * Merge `allocation_strategy_comparison_{count,sum}.csv` $\rightarrow$ `allocation_strategy_comparison_avg.csv`
   * Merge `FastestO_budget_curve_{count,sum}.csv` $\rightarrow$ `FastestO_budget_curve_avg.csv`
   * Merge `Exact_structureO_budget_curve_{count,sum}.csv` $\rightarrow$ `Exact_structureO_budget_curve_avg.csv`

3. **Adaptive Column Extraction:**
   The synthesis script automatically identifies the dataset schema and extracts corresponding node sampling statistics:
   * **Parler / Parler-E:** Automatically extracts `n_post` and `n_comment`.
   * **Amazon / Amazon-E:** Automatically extracts `n_product` and `n_review`.

---

### 2.4. Baseline Evaluation (Under Strict Oracle Budget Alignment)
To ensure a fair comparison under identical physical Oracle cost constraints ($B = \text{oracle\_cost}_{\text{POSS}}$), evaluate all baseline methods:

1. **FaSTest-Oracle (`FaSTestO`):**
   Invoke the C++ engine to perform online tree sampling with short-circuit Oracle verification:
   ```bash
   # Example: Run FaSTestO on Parler-E (SUM Aggregation)
   ./cProject/build/Fastest \
     -d dataset_test --ROOT_LABEL 2 --SAMPLE_BUDGET 30000 \
     --ESTIMATE_WITH_PREDICATE \
     --POST_ORACLE_COL ML1_oracle2_probability \
     --COMMENT_ORACLE_COL ML2_oracle2_probability \
     --AGG_FUNC sum --SUM_TABLE post --SUM_COL upvotes --SUM_LABEL 2 \
     --MULTI_PROXY_PROB ML1_proxy4b_probability \
     --BUDGET_CURVE_IN datasets/parler-e/results/efficiency/allocation_strategy_comparison_sum.csv \
     --FASTESTO_BUDGET_CURVE --FASTESTO_RUNS 5 \
     --FASTESTO_BUDGET_CURVE_OUT datasets/parler-e/results/efficiency/FastestO_budget_curve_sum.csv
   ```

2. **Projection-ABae (`PRO-ABAE.py`):**
   Adapts the two-stage pilot-sampling algorithm (VLDB 2021) to the core instance projection space:
   ```bash
   python pythonProject/src/baseline/PRO-ABAE.py \
     --parent_dataset parler-e \
     --dataset_name dataset_test \
     --ablation_csv datasets/parler-e/results/efficiency/allocation_strategy_comparison_ablation_sum.csv \
     --t1_proxy ML1_proxy4b_probability --t1_oracle ML1_oracle2_probability \
     --t2_proxy ML2_proxy1_probability --t2_oracle ML2_oracle2_probability \
     --workers 16 --runs 10 \
     --out_csv Projection_ABae_results_sum.csv
   ```

3. **Proxy-Cascade-Filter (`PSF.py`):**
   Simulates traditional relational AQP hard filtering ($<0.2$ discard, $>0.3$ accept), invoking the Oracle only in the $[0.2, 0.3]$ gray zone until the budget is exhausted:
   ```bash
   python pythonProject/src/baseline/PSF.py \
     --parent_dataset parler-e \
     --dataset dataset_test \
     --ablation_csv datasets/parler-e/results/efficiency/allocation_strategy_comparison_ablation_sum.csv \
     --table1 post --t1_proxy ML1_proxy4b_probability --t1_oracle ML1_oracle2_probability \
     --table2 comment --t2_proxy ML2_proxy1_probability --t2_oracle ML2_oracle2_probability \
     --t1_low 0.2 --t1_high 0.3 --t2_low 0.2 --t2_high 0.3 \
     --num_workers 16 \
     --out_csv PSF_results_sum.csv
   ```

4. **Exact-structureO / ENUM Baseline:**
   Run `ENUM.py` to evaluate the exact structural matching baseline subject to the same Oracle budget limits.

---

### 2.5. Statistical Significance Testing & Figure Plotting
Generate publication-quality vector PDF figures and perform statistical hypothesis testing:

1. **Error Convergence Curves (RQ1):**
   Plot error convergence curves across sampling budget gradients:
   ```bash
   python pythonProject/src/RQS/plot_convergence_curves.py --dataset dataset_test --agg_type sum
   ```
2. **Bias Analysis & Boxplots (RQ2):**
   Plot Symmetric Relative Error (SymRE) boxplots to demonstrate unbiased distribution:
   ```bash
   python pythonProject/src/RQS/plot_bias_boxplots.py --dataset dataset_test --budget 0.1
   ```
3. **Statistical Significance Testing:**
   Execute one-tailed paired $t$-tests ($p < 10^{-15}$) and Wilcoxon signed-rank tests ($p < 10^{-18}$) against baseline methods, and verify per-query stability across repeated sampling runs ($\sigma < 2.0\%$):
   ```bash
   python pythonProject/src/baseline/compare_poss_vs_fastesto.py
   ```

---

### 2.6. Theoretical Performance Upper Bound (`WEE`)
Compute the asymptotic bounds and Worst-case Execution Efficiency (WEE) metrics:
```bash
python pythonProject/src/algorithms/WEE.py --dataset dataset_test
```