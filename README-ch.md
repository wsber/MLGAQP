
# 实验复现指南 (Evaluation Reproduction)

本仓库提供了复现论文所有实验结果的完整代码。您可以选择通过总脚本一键运行，或者按照完整的流水线分步执行。

## 1. 一键复现 (One-Click Reproduction)

最简单的复现方式是直接运行总控脚本。执行以下 Shell 脚本，即可自动跑完全部流程并生成实验绘图所需的所有数据：

```bash
bash scripts/run_all_experiments.sh  
```

---

## 2. 分步执行流水线 (Step-by-Step Pipeline)

如果您希望深入了解每个步骤的细节或仅复现特定模块，请按照以下 A~F 的步骤依次运行，以获取各项 Baseline 和本论文所提方法（$\text{PROXY}$）的输出结果。

### A. 计算精确真值 (Ground Truth / EXACT)

本步骤用于获取没有任何采样误差的精准查询结果。

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