# Proxy-Guided Sampling for Approximate Graph Aggregation with Machine Learning Predicates

# 实验复现指南 (Evaluation & Reproduction Guide) -正在更新

本仓库提供了复现论文全部实验结果的完整代码与配置。用户可以选择通过总脚本**一键全自动复现**，或按照流水线**分模块逐步执行**。

> 开销说明：直接调用大模型或深度神经网络（Oracle 模型）进行实时推理会产生高昂的计算开销与时间成本。为了便于快速复现与验证，**我们已将所有查询涉及的 Oracle 和 Proxy 验证结果预先缓存至各数据集的 `csv_data/` 目录中，支持开箱即用。**

---


> ###  (1) 快速查看与即时画图 (Instant Verification: 1-Minute Figure Reproduction)
> **无需等待耗时的分层重要性采样与图匹配！** 我们已将论文全部实验评估的真实结果完整持久化在各数据集的 `datasets/{workload}results/efficiency/` 目录下。
> 如果您希望**立即验证并复现论文中的全部实验图表（RQ1 ~ RQ4）与显著性检验指标**：
> 1. 启动 Jupyter Notebook：
>    ```bash
>    jupyter notebook pythonProject/src/RQS/RQX.ipynb
>    ```
> 2. 点击 **`Run`**，即可直接读取预存数据，秒级渲染出论文中的全部高清矢量图表！

---

> ###  (2) 理论证明补充与技术报告 (Technical Report)
> 关于论文中涉及的**案例分析 (Case Study)** 详情以及**均匀树采样的完整理论证明 (Tree Sampling Proofs)**，请查阅仓库根目录下的技术报告：[**`TR.pdf`**](./TR0.1.pdf)。

> ###  (3) 代码设计说明与运行建议 (Codebase Notice & Workflow)
> 1. **可复现性与排查设计**：当前代码处于学术开源阶段。为了便于校验各阶段的中间状态、排查算法逻辑并支撑完整的数据复现链条，代码中保留了**较为细致的中间结果落盘与 I/O 校验逻辑**。
> 2. **推荐实验运行策略**：
>    * **步骤一（结构投影与权重物化）**：运行 `Projection_Sampling_and_Weight_Estimation_Runner.py`，针对每个工作负载（Workload）及聚合类型（`COUNT` / `SUM`）**仅需执行一次**。该步骤会固定拓扑投影空间 $\hat{\Psi}$ 并离线物化所有投影的结构延伸权重 $\hat{w}(\psi)$。
>    * **步骤二（采样算法与基准评估）**：在物化完成的 $\hat{\Psi}$ 基础上，可多次高效运行 `Proxy_Guided_Stratified_Importance_Sampling_Runner.py` 及各类基线脚本，快速评测不同采样策略、消融变体和多轮随机方差。
> 3. **持续重构计划**：我们团队将持续对代码库进行模块化解耦、精简 I/O 流程并优化执行效率，敬请关注后续更新。


## (4) 环境配置指南 (Environment Setup)

在运行代码前，请按照以下步骤配置 Python 虚拟环境及编译底层 C++ 采样引擎。

### 1. Python 依赖环境 (Conda)
推荐使用 Python 3.8+ 环境：
```bash
# 1. 创建并激活 conda 虚拟环境
conda create -n iogs python=3.8 -y
conda activate iogs

# 2. 一键安装全部依赖包
pip install -r requirements.txt

### 2. C++ 底层采样引擎编译
底层图匹配与候选空间树采样引擎基于 C++20 开发，依赖 CMake、Boost 和 GSL：
```bash
# 1. 安装系统依赖 (Ubuntu/Debian)
conda install -c conda-forge cmake gxx_linux-64 boost gsl -y

# 2. 编译生成 Fastest 二进制可执行文件 (已有编译好的程序,下面是重新编译一遍的过程)
cd cProject
mkdir -p build && cd build
cmake ..
make -j$(nproc)
cd ../..
```
*编译完成后，二进制程序将位于 `cProject/build/Fastest`。*
---

## 0. 实验基础：数据集与 ML 谓词架构

在进行实验前，请先了解本研究所基于的图数据集、合成查询负载及机器学习谓词架构。

---

### 0.1. 数据集与查询负载 (Datasets & Workloads)

实验基于 3 个真实的属性图/多模态图数据集开展：

1. **`Parler`**：包含文本属性的真实社交网络图，包含 **3 种不同的顶点标签/类型**：用户 (`user`)、帖子 (`post`)、评论 (`comment`)。
2. **`Parler-E`**：基于 `Parler` 派生拓展的数据集。通过将原始实体细分为互不相交的子类型，将标签数扩展至 **6 个不同顶点标签**，显著增大了标签稀疏度（Label Sparsity），从而大幅降低查询的选择率（Selectivity），用于评测低选择率极值场景。
3. **`Amazon`**：包含用户 (`user`)、文本评论 (`review`) 与商品图像 (`product`) 的多模态异构图。通过随机细分实体类型拓展为 **11 个不同顶点标签**，用于评估拓扑结构复杂度与多模态属性融合。

#### 查询生成与聚合约束 (Query Workloads)
查询图 $Q$ 通过在数据图上执行**随机游走 (Random Walk)** 生成，并挂载聚合属性与 ML 谓词：
* **聚合属性约束**：为了确保在 $Q$ 上能够执行合法的 `SUM` 和 `AVG` 聚合计算，$Q$ 必须包含至少一个数值型属性：
  * **`Parler` / `Parler-E`**：选取 `post` 节点的 `upvotes`（点赞数）属性；
  * **`Amazon`**：选取 `product` 节点的 `price`（价格）或 `rating`（评分）属性。
* **查询规模与谓词配置**：
  * **`Parler`**：包含 **245** 个单谓词查询（$|V(Q)| \in [4, 8]$，$k=1$），谓词随机挂载在 1 个 `post` 或 `comment` 节点上。
  * **`Parler-E`**：包含 **115** 个多谓词复合查询（$|V(Q)| \in [4, 8]$，$k \ge 2$），谓词同时挂载在至少 1 个 `post` 节点和至少 1 个 `comment` 节点上。
  * **`Amazon`**：包含 **750** 个多模态复合多谓词查询（$|V(Q)| \in [3, 8]$，$k \ge 2$），谓词同时挂载在至少 1 个 `product` 图像节点和至少 1 个 `review` 文本节点上。

---

### 0.2. ML 谓词架构与模型选型 (ML Predicates: Oracle vs. Proxy)

每个原子 ML 谓词 $\mathcal{P}_i$ 均配置一个**高精度 Oracle 模型**（用于精确无偏验证）和一个**轻量级 Proxy 模型**（用于高效近似打分与引导分层重要性采样）：

| 数据集 (Dataset) | 目标实体 (Vertex Type) | 谓词语义任务 (Predicate Semantics $\mathcal{P}$) | Oracle 模型 (参数量) | Proxy 代理模型 (参数量) | Proxy $F_1$ | 代理推理加速比 (Speedup) |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| **`Amazon`** | `product` | **图像材质分类 (Image Texture)**<br>*(wooden/plastic/metal/fabric/glass?)* | `siglip-so400m-patch14-384`<br>*(878M)* | `siglip-base`<br>*(84M)* | 0.7546 | **$26.8\times$** |
| | `review` | **文本情感分析 (Sentiment Analysis)**<br>*(positive/negative?)* | `roberta-large-sst2`<br>*(355M)* | `bert-mini-finetuned-sst2`<br>*(11M)* | 0.8890 | **$22.1\times$** |
| **`Parler`** /<br>**`Parler-E`** | `post` | **观点/立场推断 (Opinion Inference)**<br>*(Support/Oppose Donald Trump?)* | `deberta-v2-xxlarge-mnli`<br>*(1.5B)* | `deberta-v3-base-mnli`<br>*(184M)* | 0.7720 | **$42.5\times$** |
| | `comment` | **文本情感分析 (Sentiment Analysis)**<br>*(positive/negative?)* | `roberta-large-sst2`<br>*(355M)* | `bert-mini-finetuned-sst2`<br>*(11M)* | 0.7876 | **$22.1\times$** |

* **代理质量分级 (Proxy Quality Tiers $M_{P1} \sim M_{P4}$)**：为了评估 $\text{PROXY}$ 对代理精度的敏感性与鲁棒性（RQ3），我们为每项任务通过微调或更简化的模型架构构建了 4 个代理质量梯度，其相对 $F_1$ 分数在 $[0.65, 0.89]$ 范围内单调递减，推理速度单调递增。

---

### 0.3. 硬件实验环境 (Hardware Setup)

论文所有实验均在以下配置的高性能服务器上完成测试与评测：
* **操作系统**：Ubuntu 22.04 LTS
* **处理器 (CPU)**：Dual Intel(R) Xeon(R) Gold 6130 CPUs @ 2.10GHz
* **内存 (RAM)**：503 GB
* **图形计算卡 (GPU)**：$4 \times$ NVIDIA GeForce RTX 3090 GPUs (24GB VRAM)

> **⚠️ 注意**：每个 workload 包含数百个复杂的子图同构与大模型谓词评估，全量执行耗时数小时并需要每个 workload 预留至少 100GB 磁盘空间。**本仓库已为所有 workload 预先提供了计算完成的精确真值 (Ground Truth, GT)**，复现时无需重新运行昂贵的全量匹配与全量 Oracle 验证。

---

### 0.4. 项目目录结构说明 (Repository Structure)

本项目由底层的 **C++ 采样加速引擎 (`cProject`)** 与上层的 **Python 代理引导采样算法框架 (`pythonProject`)** 协同构成。整体目录结构如下：

```text
PROXY/
├── cProject/                                   # [C++ 核心引擎] 负责候选空间构建、树采样与语义投影权重估计
│   ├── build/                                  # 预编译二进制目录（包含编译就绪的 Fastest 可执行程序）
│   ├── driver/                                 # C++ 入口文件 (subgraph-cardinality-estimation.cc)
│   ├── lib/                                    # 图数据结构、CS 构建、均匀树采样等底层算法库
│   └── CMakeLists.txt                          # C++ 项目 CMake 编译配置文件
│
├── datasets/                                   # [数据与结果存储] 存放三大工作负载的数据图、查询图与实验结果
│   ├── parler/                                 # Parler 单谓词工作负载 (data_graph / query_graph / ground_truth / results)
│   ├── parler-e/                               # Parler-E 多谓词扩展工作负载
│   └── amazon/                                 # Amazon 多模态复杂异构图工作负载
│
├── Model/                                      # [ML 模型库] 存放 Oracle 真值模型与 Proxy 代理模型权重及配置
│
├── pythonProject/                              # [Python 实验框架] PROXY 算法实现、基准对比、流水线控制与绘图
│   └── src/
│       ├── algorithms/                         # PROXY 核心算法实现
│       │   ├── exact_subgraph_match.py         # 精确子图同构匹配实现脚本
│       │   ├── compute_truth.py                # 精确真值 (Ground Truth, GT) 计算与谓词聚合验证类
│       │   └── proxy_sample.py                 # 代理引导的分层重要性采样 (POSSA) 核心算法与消融变体实现
│       │
│       ├── baseline/                           # 基准方法实现库
│       │   ├── ...                             # 论文基础基线 (ENUM, FASTEST-ORACLE, WEE 等)
│       │   └── ...                             # 对比方法 (如 PRO-ABAE, PSF 级联过滤等)
│       │
│       ├── runner/                             # 执行调度器 (Runners)
│       │   └── ...                             # 封装调用 C++ 引擎与 Python 采样模块的端到端运行脚本
│       │
│       └── RQS/                                # 实验可视化与图表绘制脚本 (绘制论文中 RQ1 ~ RQ4 的全部图表)
│
├── scripts/                                    # [自动化脚本] 一键复现 Shell 脚本 (run_all_experiments.sh 等)
└── ...                                         # 辅助代码与数据配置文件
```

---

## 1. 一键复现

您可以通过运行跨所有数据集的主一体化脚本，或执行专用的工作负载特定一键脚本来复现实验结果。

---

### 1.1. 主脚本（所有工作负载与聚合）
要自动运行完整流水线（包括 C++ 权重物化和分层采样），覆盖全部三个工作负载（`Parler`、`Parler-E`、`Amazon`）并生成绘图所需的全部数据：
<!-- 
```bash
# 确保您的 conda 环境已激活
conda activate iogs

# 授予执行权限并运行主脚本
chmod +x scripts/run_all_experiments.sh
bash scripts/run_all_experiments.sh
``` -->

---

### 1.2. 工作负载特定的一键脚本
如果您希望评估或调试特定数据集，而无需运行整个数小时的基准测试套件，我们提供了专用的一键自动化脚本。

#### 示例：Parler（`COUNT` 模式）
脚本 `run_parler_count.sh` 在 **Parler** 工作负载的 `COUNT` 模式下自动化端到端工作流：
1. **步骤 1（离线投影与物化）：** 调用 C++ 引擎执行均匀树采样、估计投影扩展权重 $\hat{w}(\psi)$，并物化紧凑核心实例空间。
2. **步骤 2（在线 POSSA 采样）：** 在预算梯度 $\alpha \in [1\%, 90\%]$ 上执行代理引导的分层重要性采样，每个刻度进行 5 次独立运行。

```bash
# 1. 激活 conda 环境
conda activate iogs

# 2. 授予执行权限
chmod +x scripts/run_parler_e_count.sh

# 3. 执行一键脚本
./scripts/run_parler_e_count.sh
```

* **生成的输出文件：**
  ```text
  datasets/parler/results/efficiency/allocation_strategy_comparison_count.csv
  ```

> **💡 环境说明：** 
> 所有一键脚本都会自动配置 `PYTHONPATH`，并导出指向 `${CONDA_PREFIX}/lib` 的 `LD_LIBRARY_PATH`，以确保与 `GLIBCXX_3.4.31/32` 的运行时兼容性。


---

## 2. 分步执行流水线 (Step-by-Step Pipeline)

如果您希望分阶段检查流水线细节、复现论文中特定的研究问题（Research Questions, RQs），或单独运行某个对比基线，请按顺序执行步骤 **2.1 至 2.6**。

### 2.1. 计算精确真值 (Ground Truth / EXACT)

通过穷举 + Oracle 验证所有子图匹配嵌入（Embeddings），获取不含任何采样噪声的精确真值 *(建议跳过此步，直接使用已预先计算好的 GT 文件)*。

1. **精确子图匹配**：运行 `exact_subgraph_match.py`，调用底层的 C++ 引擎执行精确子图匹配，并保存中间结果。
2. **谓词验证与聚合**：运行 `EXACT.py`，使用查询对应的 Oracle 谓词验证上述匹配结果，并进行最终聚合计算（支持 `agg_mode={count, sum}`）。

```bash
python pythonProject/src/algorithms/EXACT.py --dataset dataset_test --agg_mode count
python pythonProject/src/algorithms/EXACT.py --dataset dataset_test --agg_mode sum
```
* **输出文件：** `results/T_true_*_count.json` 与 `results/T_true_*_sum.json`

---

### 2.2. $\text{PROXY}$ `count` / `sum` 实验

针对聚合模式为 `count` 和 `sum` 的情况进行实验验证。

#### 2.2.1. 投影权重估计与实例聚合 (Projection Weight Estimation & Aggregation)
将查询图分解为语义投影 $\hat{\Psi}$，通过 C++ 引擎估计各语义投影的权重 $\hat{w}(\psi)$，并关联对应的 ML 代理/Oracle 概率值：

* **Parler 数据集 (`COUNT` 聚合示例)**：
```bash
python pythonProject/src/runner/Projection_Sampling_and_Weight_Estimation_Runner.py \
  --base_dir $(pwd) \
  --dataset parler \
  --sample_budget 60000 \
  --agg_func count \
  --table1 post \
  --table2 comment
```

* **Amazon 数据集 (`SUM` 聚合示例)**：
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

* **中间输出：** `results/structure_estimate/*.csv`（按单个查询拆分后的原始实例文件）。
* **最终输出：** `results/aggregated_results/aggregated_list_*.csv`（包含权重 $a$ 与各节点 ML 概率的核心实例紧凑投影空间）。

#### 2.2.2. 核心性能与组件消融实验 (RQ1, RQ2 & RQ4)
在物化好的核心实例投影空间（`aggregated_results/`）上，运行代理引导的分层重要性采样（POSSA）及各消融变体：

* **针对 RQ1 与 RQ2（跨采样率的核心性能对比）：**
  在渐进采样预算梯度 $\alpha \in [1\%, 90\%]$ 下评估 $\text{PROXY}$ (POSSA) 方法：

  * *Parler 数据集 (`parler` / `dataset_three`)*:
  ```bash
  python pythonProject/src/runner/Proxy_Guided_Stratified_Importance_Sampling_Runner.py \
    --parent_dataset parler \
    --dataset_name dataset_three \
    --target_ticks "0.01,0.05,0.075,0.1,0.125,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9" \
    --run_times 5 \
    --max_workers 16
  ```

  * *Parler-E 数据集 (`parler-e` / `dataset_test`)*:
  ```bash
  python pythonProject/src/runner/Proxy_Guided_Stratified_Importance_Sampling_Runner.py \
    --parent_dataset parler-e \
    --dataset_name dataset_test \
    --target_ticks "0.01,0.05,0.075,0.1,0.125,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9" \
    --run_times 5 \
    --max_workers 16
  ```

  * *Amazon 数据集 (`amazon` / `amazon_extend`)*:
  ```bash
  python pythonProject/src/runner/Proxy_Guided_Stratified_Importance_Sampling_Runner.py \
    --parent_dataset amazon \
    --dataset_name amazon_extend \
    --target_ticks "0.01,0.05,0.075,0.1,0.125,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9" \
    --run_times 5 \
    --max_workers 16
  ```
  * **输出文件：** `datasets/<parent_dataset>/results/efficiency/allocation_strategy_comparison_{agg_mode}.csv`

* **针对 RQ4（固定预算 $\alpha=10\%$ 下的组件消融研究）：**
  在固定采样率下评估各组件变体（`UN`: 均匀采样, `PO`: 仅代理, `WO`: 仅权重, `MAB`: 多臂老虎机, `POSSA`: 完整方法）：
  ```bash
  python pythonProject/src/runner/Proxy_Guided_Stratified_Importance_Sampling_Runner.py \
    --parent_dataset parler-e \
    --dataset_name dataset_test \
    --target_ticks "0.1" \
    --run_times 5 \
    --max_workers 16
  ```
  * **输出文件：** `datasets/<parent_dataset>/results/efficiency/allocation_strategy_comparison_ablation_{agg_mode}.csv`

#### 2.2.3. 敏感性与代理质量衰减分析 (RQ3)
评估在单谓词代理质量下降以及复杂多谓词噪声场景下算法的鲁棒性：
```bash
python pythonProject/src/runner/Sensitivity_single_predicate_Runner.py --dataset dataset_test
python pythonProject/src/runner/Sensitivity_multi_predicate_comparation.py --dataset dataset_test
```
* **输出文件：** `results/efficiency/proxy_quality_ablation_{agg_mode}.csv`

---

### 2.3. `avg` 查询的离线比率合成 (Theorem 6)
基于论文 **Theorem 6** 提出的比率估计量（$\hat{\tau}_{\text{avg}} = \hat{\tau}_{\text{sum}} / \hat{\tau}_{\text{count}}$），`avg` 结果无需重新运行图采样引擎，而是通过离线合成已完成的 `count` 和 `sum` 实验数据获得。

1. **合成真值 JSON (Synthesizing Ground Truth)：**
   根据公式 $\tau_{\text{avg}} = \tau_{\text{sum}} / \tau_{\text{count}}$ 计算精确平均值：
   * **输入：** `results/T_true_*_sum.json` 与 `results/T_true_*_count.json`
   * **输出：** 生成 `results/T_true_*_avg.json`

2. **合成实验结果曲线与误差对齐：**
   通过对公共键 `(query_basename, budget_frac, run_id)` 进行内联结（Inner Join），合并 `count` 与 `sum` 的评估 CSV 文件：
   ```bash
   python pythonProject/src/baseline/synthesize_avg_results.py --dataset dataset_test
   ```
   * 合并 `allocation_strategy_comparison_{count,sum}.csv` $\rightarrow$ 生成 `allocation_strategy_comparison_avg.csv`
   * 合并 `FastestO_budget_curve_{count,sum}.csv` $\rightarrow$ 生成 `FastestO_budget_curve_avg.csv`
   * 合并 `Exact_structureO_budget_curve_{count,sum}.csv` $\rightarrow$ 生成 `Exact_structureO_budget_curve_avg.csv`

3. **自适应数据列提取机制：**
   合成脚本会自动识别数据集类型，并提取相应的节点采样统计列：
   * **Parler / Parler-E 数据集：** 自动提取 `n_post` 和 `n_comment`。
   * **Amazon / Amazon-E 数据集：** 自动提取 `n_product` 和 `n_review`。

---

### 2.4. 基线方法评估 (在严格的 Oracle 预算对齐下)
为确保在完全相同的物理 Oracle 成本限制（$B = \text{oracle\_cost}_{\text{POSS}}$）下进行公平比较，评估所有基线方法：

1. **FaSTest-Oracle (`FaSTestO`)：**
   调用 C++ 引擎执行带有短路 Oracle 校验的在线树采样：
   ```bash
   # 示例：在 Parler-E 上运行 FaSTestO (SUM 聚合)
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

2. **Projection-ABae (`PRO-ABAE.py`)：**
   将原版两阶段飞行采样（Pilot-Sampling）算法（VLDB 2021）迁移应用到核心实例投影空间：
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

3. **Proxy-Cascade-Filter (`PSF.py`)：**
   模拟传统关系表 AQP 的硬剪枝机制（$<0.2$ 舍弃, $>0.3$ 接受），并在 $[0.2, 0.3]$ 灰色地带消耗预算调用 Oracle 验证：
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

4. **Exact-structureO / ENUM 基线：**
   运行 `ENUM.py` 评估基于精确结构匹配并受 Oracle 预算限制的枚举基线。

---

### 2.5. 统计显著性检验与论文图表绘制
生成所有符合 VLDB/IEEE 出版级标准的 PDF 矢量图，并计算统计假设检验：

1. **误差收敛曲线 (RQ1)：**
   绘制不同采样预算梯度下的误差收敛 PDF 折线图：
   ```bash
   python pythonProject/src/RQS/plot_convergence_curves.py --dataset dataset_test --agg_type sum
   ```
2. **偏差分析与箱线图 (RQ2)：**
   绘制对称相对误差（SymRE）箱线图以展示无偏性分布：
   ```bash
   python pythonProject/src/RQS/plot_bias_boxplots.py --dataset dataset_test --budget 0.1
   ```
3. **统计显著性检验 (Statistical Significance Testing)：**
   针对基线方法执行单尾配对 $t$ 检验（$p < 10^{-15}$）与 Wilcoxon 符号秩检验（$p < 10^{-18}$），并验证多轮重复采样的单查询稳定性（$\sigma < 2.0\%$）：
   ```bash
   python pythonProject/src/baseline/compare_poss_vs_fastesto.py
   ```

---

### 2.6. 理论性能上界计算 (`WEE`)
计算最坏情况执行效率（Worst-case Execution Efficiency, WEE）的理论渐近指标与上界：
```bash
python pythonProject/src/algorithms/WEE.py --dataset dataset_test
```