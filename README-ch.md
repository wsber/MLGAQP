
# 实验复现指南 (Evaluation Reproduction) - 加速更新中

本仓库提供了复现论文所有实验结果的完整代码。您可以选择通过总脚本一键运行，或者按照完整的流水线分步执行。运行 oracle 也就是ML 谓词检验会消耗大量的时间或金钱, 为方便大家运行, 已经将所有用到的 Oracle结果缓存到各数据集 csv_data中. 

---

> ###  理论证明与技术报告 (Technical Report)
> 关于论文中涉及的**案例分析 (Case Study)** 详情以及**均匀树采样完整理论证明 (Tree Sampling Proofs)**，请查阅仓库根目录下的技术报告文档：[**`TR.pdf`**](./TR0.1.pdf)。

> ###  代码说明与重构声明 (Codebase Notice & Roadmap)
> 当前代码处于研究阶段的开源状态。为了便于深入排查算法各个模块的中间状态、校验实验结果以及支撑完整的数据复现链条，代码中保留了**较多的中间结果文件读写与落盘逻辑**，导致整体架构略显臃肿。
> 我们团队后续将**持续对代码库进行重构、解耦与性能优化**，进一步精简 I/O 流程并提升代码可读性，敬请期待。

--- 

## 0. 实验基础：数据集与 ML 谓词架构

在进行实验前，请先了解本研究所基于的图数据集、合成查询负载及机器学习谓词架构。

---

###  A. 数据集与查询负载 (Datasets & Workloads)

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

###  B. ML 谓词架构与模型选型 (ML Predicates: Oracle vs. Proxy)

每个原子 ML 谓词 $\mathcal{P}_i$ 均配置一个**高精度 Oracle 模型**（用于精确无偏验证）和一个**轻量级 Proxy 模型**（用于高效近似打分与引导分层重要性采样）：

| 数据集 (Dataset) | 目标实体 (Vertex Type) | 谓词语义任务 (Predicate Semantics $\mathcal{P}$) | Oracle 模型 (参数量) | Proxy 代理模型 (参数量) | Proxy $F_1$ | 代理推理加速比 (Speedup) |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| **`Amazon`** | `product` | **图像材质分类 (Image Texture)**<br>*(wooden/plastic/metal/fabric/glass?)* | `siglip-so400m-patch14-384`<br>*(878M)* | `siglip-base`<br>*(84M)* | 0.7546 | **$26.8\times$** |
| | `review` | **文本情感分析 (Sentiment Analysis)**<br>*(positive/negative?)* | `roberta-large-sst2`<br>*(355M)* | `bert-mini-finetuned-sst2`<br>*(11M)* | 0.8890 | **$22.1\times$** |
| **`Parler`** /<br>**`Parler-E`** | `post` | **观点/立场推断 (Opinion Inference)**<br>*(Support/Oppose Donald Trump?)* | `deberta-v2-xxlarge-mnli`<br>*(1.5B)* | `deberta-v3-base-mnli`<br>*(184M)* | 0.7720 | **$42.5\times$** |
| | `comment` | **文本情感分析 (Sentiment Analysis)**<br>*(positive/negative?)* | `roberta-large-sst2`<br>*(355M)* | `bert-mini-finetuned-sst2`<br>*(11M)* | 0.7876 | **$22.1\times$** |

* **代理质量分级 (Proxy Quality Tiers $M_{P1} \sim M_{P4}$)**：为了评估 $\text{PROXY}$ 对代理精度的敏感性与鲁棒性（RQ3），我们为每项任务通过微调或更简化的模型架构构建了 4 个代理质量梯度，其相对 $F_1$ 分数在 $[0.65, 0.89]$ 范围内单调递减，推理速度单调递增。

---

###  C. 硬件实验环境 (Hardware Setup)

论文所有实验均在以下配置的高性能服务器上完成测试与评测：
* **操作系统**：Ubuntu 22.04 LTS
* **处理器 (CPU)**：Dual Intel(R) Xeon(R) Gold 6130 CPUs @ 2.10GHz
* **内存 (RAM)**：503 GB
* **图形计算卡 (GPU)**：$4 \times$ NVIDIA GeForce RTX 3090 GPUs (24GB VRAM)

> **⚠️ 注意**：每个 workload 包含数百个复杂的子图同构与大模型谓词评估，全量执行耗时数小时并需要每个 workload 预留至少 100GB 磁盘空间。**本仓库已为所有 workload 预先提供了计算完成的精确真值 (Ground Truth, GT)**，复现时无需重新运行昂贵的全量匹配与全量 Oracle 验证。

## 1. 一键复现 (One-Click Reproduction)

最简单的复现方式是直接运行总控脚本。执行以下 Shell 脚本，即可自动跑完全部流程并生成实验绘图所需的所有数据：

```bash
bash scripts/run_all_experiments.sh  
```

---

## 2. 分步执行流水线 (Step-by-Step Pipeline)

如果您希望深入了解每个步骤的细节或仅复现特定模块，请按照以下 A~F 的步骤依次运行，以获取各项 Baseline 和本论文所提方法（$\text{PROXY}$）的输出结果。

### A. 计算精确真值 (Ground Truth / EXACT)

本步骤用于获取没有任何采样误差的精准查询结果。(建议跳过直接使用GT文件)


1. **精确子图匹配**：运行 `exact_subgraph_match.py`，该脚本将调用底层的 C++ 引擎执行精确子图匹配，并保存中间结果。
2. **谓词验证与聚合**：运行 `EXACT.py`，使用查询对应的 Oracle 谓词验证上述匹配结果，并进行最终聚合计算（支持 `agg_mode={count, sum}`）。

### B. $\text{PROXY}$ `count` / `sum` 实验

针对聚合模式为 `count` 和 `sum` 的情况进行实验验证。

1. **预处理与权重估计**：运行 `Projection_Sampling_and_Weight_Estimation_Runner.py`，得到论文中所定义的投影采样空间 $\hat{\Psi}$ 以及权重估计器 $\hat{w}(\psi)$。
2. **核心性能与消融实验 (RQ1, RQ2 & RQ4)**：运行 `Proxy_Guided_Stratified_Importance_Sampling_Runner.py` 在 $\hat{\Psi}$ 上执行分层重要性采样：
   
   * **对于 RQ1 & RQ2 (核心性能对比)**：
     仅启用 PROXY 方法，配置目标采样率梯度：
     ```python
     methods_map = {
         "PROXY": sampler.run_possa,
     }
     # --target_ticks 0.01,0.05,0.075,0.1,0.125,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
     ```
     输出结果保存于：`allocation_strategy_comparison_{agg_mode}.csv`
     
   * **对于 RQ4 (组件消融研究)**：
     在固定的采样率下运行各个消融变体：
     ```python
     methods_map = {
         "PROJ": sampler.run_baseline_uniform,
         "PO": sampler.run_baseline_proxy,
         "WO": sampler.run_baseline_weight_only,
         "MAB": sampler.run_mab_sampling,
         "PROXY": sampler.run_possa,
     }
     # --target_ticks 0.1
     ```
     输出结果保存于：`allocation_strategy_comparison_ablation_{agg_mode}.csv`

3. **鲁棒性与退化分析 (RQ3)**：运行 `Sensitivity_single_predicate_Runner.py` 与 `Sensitivity_multi_predicate_comparation.py`，分别检验在单谓词退化情况和多谓词复杂情况下的算法鲁棒性。
   输出结果保存于：`proxy_quality_ablation_{agg_mode}.csv`


### C. $\text{PROXY}$ `avg` 实验

基于论文中**定理 6** 提出的比率估计器，`avg` 的结果无需重新运行 C++ 引擎，而是通过离线合成已完成的 `count` 和 `sum` 实验数据来获得。

1. **合并真值 (Ground Truth 合成)**：
   根据公式 $\tau_{\text{avg}} = \tau_{\text{sum}} / \tau_{\text{count}}$ 计算 `avg` 真值。
   * **输入**：`T_true_*_sum.json` 与 `T_true_*_count.json`
   * **输出**：生成 `T_true_*_avg.json`

2. **合成实验结果与误差计算**：
   运行比率对齐脚本（遵循 $\hat{\tau}_{\text{avg}} = \hat{\tau}_{\text{sum}} / \hat{\tau}_{\text{count}}$ 逻辑）：
   * **核心与消融策略 (RQ1, RQ2, RQ4)**：合并 `allocation_strategy_comparison_{count,sum}.csv` $\rightarrow$ 得到 `allocation_strategy_comparison_avg.csv`
   * **基线 Fastest-Oracle**：合并 `FastestO_budget_curve_{count,sum}.csv` $\rightarrow$ 得到 `FastestO_budget_curve_avg.csv`
   * **基线 Exact-structureO**：合并 `Exact_structureO_budget_curve_{count,sum}.csv` $\rightarrow$ 得到 `Exact_structureO_budget_curve_avg.csv`

3. **自适应数据列提取**：
   合成脚本内置自动适配机制，可根据数据集风格自动抓取所需采样统计列：
   * **Parler 风格数据集**：自动提取 `n_post` 与 `n_comment`
   * **Amazon 风格数据集**：自动提取 `n_product` 与 `n_review`


### D. 时间对等协议：计算等效虚拟预算 $B_{\text{virtual}}$

为了公平对比各类算法的效率，我们在统一的时间维度上折算预算。

1. **提取调用次数**：读取 `allocation_strategy_comparison_{count,sum}.csv`，提取在指定采样率 $\alpha$ 下，Oracle 模型和 Proxy 模型的实际运行次数 $N_{oi}$ 和 $N_{pi}$。
2. **折算虚拟预算**：结合当前数据集上 Oracle 模型与 Proxy 模型的平均推理延迟（$c_i$ 与 $c_p^i$），计算总的时间等效虚拟预算 $B_{\text{virtual}}$。


### E. 基于 $B_{\text{virtual}}$ 评估基准方法 (ENUM & FASTEST-ORACLE)

在统一的虚拟预算下，评估其他基准方法估计 $\hat{\tau}$ 的绝对平均误差 (AAE) 等核心指标。

1. **评估 ENUM 基线**：
   运行 `ENUM.py`，根据等效预算计算 ENUM 方法在当前查询负载下的表现（支持 `agg_mode={count, sum}`）。
   
2. **评估 FASTEST-ORACLE 基线**：
   运行 `FASTEST-ORACLE.py`，调用底层 C++ 引擎的相应算法，在相同的预算下估算目标值 $\hat{\tau}$（支持 `agg_mode={count, sum}`）。
   * 输出结果将保存于 `results/efficiency` 文件夹中的 `FastestO_budget_curve_{agg_mode}.csv`。
   * 该文件记录了每个查询 $Q$ 在指定数据集与采样率下，独立运行 $k$ 次（如 $k=5$ 或 $10$）的估计值 $\hat{\tau}$。


### F. 计算理论性能上界 (WEE 渐近线)

1. 运行 `WEE.py`，为每个目标数据集计算其 WEE 理论渐近线指标。
```