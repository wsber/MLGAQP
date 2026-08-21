***

# Proxy-Guided Sampling for Approximate Graph Aggregation with Machine Learning Predicates

# Evaluation & Reproduction Guide ( V-0.99 Continuously Updated...)

This repository provides the complete code and configuration required to reproduce all experimental results reported in the paper. Users can choose to **fully automate the reproduction via a master script** or **execute the pipeline step-by-step by module**.

> **Overhead Notice:** Directly invoking Large Language Models or Deep Neural Networks (Oracle models) for real-time inference incurs substantial computational and time costs. To facilitate rapid reproduction and verification, **we have pre-cached all Oracle and Proxy verification results for all queries into the `csv_data/` directory of each dataset, enabling ready-to-use execution.**

---

> ### (1) Instant Verification: 1-Minute Figure Reproduction
> **No need to wait for time-consuming stratified importance sampling and graph matching!** We have fully persisted the ground-truth evaluation results of all paper experiments under the `datasets/{workload}/results/efficiency/` directory for each dataset.
> If you wish to **immediately verify and reproduce all experimental figures (RQ1–RQ4) and statistical significance metrics from the paper**:
> 1. Launch Jupyter Notebook:
>    ```bash
>    jupyter notebook pythonProject/src/RQS_plots/RQX.ipynb
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
Python 3.10 is recommended:
```bash
# 1. Create and activate conda virtual environment
conda create -n iogs python=3.10 -y
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
  * **`Parler`:** Contains **~245** single-predicate queries ($|V(Q)| \in [4, 8]$, $k=1$), with the predicate randomly assigned to 1 `post` or `comment` vertex.
  * **`Parler-E`:** Contains **~115** multi-predicate composite queries ($|V(Q)| \in [4, 8]$, $k \ge 2$), with predicates simultaneously assigned to at least 1 `post` vertex and at least 1 `comment` vertex.
  * **`Amazon`:** Contains **~750** multimodal composite multi-predicate queries ($|V(Q)| \in [3, 8]$, $k \ge 2$), with predicates assigned to at least 1 `product` image vertex and at least 1 `review` text vertex.

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

> 💡 **Implementation Note (Theory vs. System Optimization):** 
> To achieve lower possible variance in practice, our execution engine uses **without-replacement (WOR) sampling** and merges tiny strata ($|S_i| < 2$) when the allocated budget exceeds stratum sizes. This is a strict system-level safety fallback to prevent out-of-bounds errors, while our theoretical proofs model *with-replacement (WR)* for closed-form tractability. All reported tables directly correspond to this codebase.

```text
PROXY/
├── cProject/                                   # [updated] [C++ Core Engine] CS construction, tree sampling, and semantic projection weight estimation 
│   ├── build/                                  # Precompiled binary directory (contains the compiled 'Fastest' executable)
│   ├── driver/                                 # C++ entry point (subgraph-cardinality-estimation.cc)
│   ├── lib/                                    # Graph data structures, CS builder, uniform tree sampler, etc.
│   └── CMakeLists.txt                          # CMake configuration file
│
├── datasets/                                   # [updated] [Data & Results Storage] Data graphs, query graphs, and results for the three workloads
│   ├── parler/                                 # Parler single-predicate workload (data_graph / query_graph / ground_truth / results)
│   ├── parler-e/                               # Parler-E multi-predicate expanded workload
│   └── amazon/                                 # Amazon multimodal Multi-Modal graph workload
│
├── Model/                                      # [Updating] [ML Model Repository] Oracle and Proxy model weights and configs
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
│       └── RQS-plot/                           # Visualization and plotting scripts for paper figures (RQ1 ~ RQ4)
│
├── scripts/                                    # [Automation Scripts] One-click benchmark & reproduction shell scripts
│   ├── run_get_all_structural_matching.sh      # [updated] Exact subgraph match: get all structural_matching         
│   ├── run_RQ1.sh                              # [updated] RQ1: PROXY End-to-end efficiency and runtime breakdown benchmark      
│   ├── run_RQ2.sh                              # [updated] RQ2: PROXY Accuracy convergence across budget ticks (COUNT, SUM, AVG) 
│   ├── run_RQ3.sh                              # [updated] RQ3: Proxy quality sensitivity and robustness ablation          
│   ├── run_RQ4.sh                              # [updated] RQ4: Allocation strategy ablation (UN, PO, WO, MAB, POSS)
│   ├── run_all_gt.sh                           # [updated] Ground Truth caculate
│   ├── run_Enum.sh                             # [updated] RQ1&RQ2: Baseline: Alog 1 in paper        
│   ├── run_Fastest-Oracle.sh                   # [updated] RQ1&RQ2: Baseline: Parallel FaSTest-Oracle  runner
│   ├── run_WEE.sh                           # [updated] RQ2:     Baseline : Weight Estimation Error (WEE) Floor   
│   ├── run_Pro_Abae.sh                      # [updated] Rebuttal: Baseline: Projection space + ABAE.   
│   ├── run_Pro_Cacade_Filter.sh                # [updated] Rebuttal: Baseline: Projection space + cascade filter (SUPG.ScaleDoc style)              
│   ├── run_parler_{count,sum}.sh               # [updated] Modular one-click scripts for Parler (Single Predicate)         
│   ├── run_parler_e_{count,sum}.sh             # [updated] Modular one-click scripts for Parler-E (Multi-Predicate)        
│   └── run_amazon_{count,sum}.sh               # [updated] Modular one-click scripts for Amazon (multi-modality Graph)     
│   ├── run_all_avg.sh                          # [updated] Offline ratio-based AVG synthesis from COUNT and SUM results    
└── ...                                         
```

---

## 1. One-Click Reproduction

You can reproduce the experimental results either by running the master all-in-one script across all datasets or by executing dedicated workload-specific one-click scripts.

> ⚠️ **IMPORTANT NOTE FOR REPRODUCIBILITY:**
>
> All pre-computed exact cardinalities and ground truth mappings (`results/T_true_*. json`) are **already packaged in this repository by default**, allowing you to immediately run all downstream sampling and approximation benchmarks (RQ1–RQ4).
>
> **However, if you wish to generate the ground truth from scratch or reproduce the exact enumeration baseline (ENUM / EXACT):**
> Getting all exact subgraph matching is #-hard and involves massive memory and CPU overhead (can take several hours per workload).  We provide an **interactive, single-workload generator** `run_get_all_structural_matching.sh` to safely execute exact matching one dataset at a time:

```bash
cd pythonProject/scripts
chmod +x *.sh
# Get structural matching for all queries in the dataset (without Oracle verification, i.e., ML predicate validation). Only one workload can be selected and run at a time.
./run_get_all_structural_matching.sh

# EXACT: Obtain GroundTruth
./run_all_gt.sh

# Baseline ENUM: For details, please refer to Algorithm One in Paper
./run_Enum.sh
```
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
2. **Step 2 (Online Proxy Sampling):** Executes proxy-guided stratified importance sampling across budget gradients $\alpha \in [1\%, 90\%]$ with 5 independent runs per tick.

```bash
# 1. Activate conda environment
conda activate iogs
cd pythonProject/scripts
# 2. Grant execution permission
chmod +x *.sh

# 3. Execute the one-click script

# RQ-Specific Reproduction Scripts (RQ1 – RQ4)

# Generates  `PROXY` results for RQ1. Visualize in `pythonProject/src/RQs_plots/RQ1&RQ2.ipynb` or `pythonProject/src/RQs_plots/Rebuttal_exp_result.ipynb`.
./run_RQ1.sh 

# Generates  `PROXY` results for RQ2. Visualize in `pythonProject/src/RQs_plots/RQ1&RQ2.ipynb`.

./run_RQ2.sh 

# Generates all results for RQ3. Visualize in `pythonProject/src/RQs_plots/RQ3:Sensitivity.ipynb`.
./run_RQ3.sh

# Generates all results for RQ4. Visualize in `pythonProject/src/RQs_plots/RQ4:Ablation.ipynb`.
./run_RQ4.sh


#  baseline : Fastest-Oracle Generates `Fastest-Oracle` results for RQ1&RQ2. Visualize in `pythonProject/src/RQs_plots/RQ1&RQ2.ipynb` 
./run_Fastest-Oracle.sh

#  baseline : Weight Estimation Error (WEE) Floor
./run_WEE.sh

#  baseline - in rebuttal:  Projection space + ABAE. Visualize in `pythonProject/src/RQs_plots/Rebuttal_exp_result.ipynb`
#  After rebttual, the method was systematically improved. The method ARE was improved to some extent, but it still lagged behind PROXY
./run_Pro_Abae.sh

#  baseline - in rebuttal:  Projection space + cascade filter (SUPG.ScaleDoc style). Visualize in `pythonProject/src/RQs_plots/Rebuttal_exp_result.ipynb` 
./run_Pro_Cacade_Filter.sh 


./run_all_avg.sh

# 4. Execuate Single workload test

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
  eg. datasets/parler/results:

  1. /efficiency/allocation_strategy_comparison_count.csv: The Proxy result file stores the estimated values and the number of oracle calls for each query at each sampling rate, and is used for RQ1&RQ2 

  2. FastestO_budget_curve_count.csv: Important baseline FASTEST-ORACLE results for RQ1&RQ2

  3.allocation_strategy_comparison_ablation_count.csv: The ablation experiment result file stores the estimated values at each sampling rate queried under each ablation experiment, which is used for RQ4
  
  ```

---

---
## The following are the experimental details that are being sorted out and supplemented. All the above content is now ready and can fully reproduce the experiments in the paper and the supplementary experiments in rebuttal..

