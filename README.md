
# Evaluation Reproduction Guide

This repository provides the complete source code required to reproduce all the experimental results presented in the paper. You can choose to execute everything automatically via a single master script, or run the pipeline step-by-step.

## 1. One-Click Reproduction

The simplest way to reproduce the results is to run the master script. Execute the following shell script to automatically run the entire pipeline and generate all data required for the experimental figures:

```bash
bash scripts/run_all_experiments.sh 
```

---

## 2. Step-by-Step Pipeline

If you wish to examine the details of each step or reproduce specific modules, please execute steps A through F in sequence to obtain the outputs for all baselines and our proposed method ($\text{PROXY}$).

### A. Computing the Ground Truth (EXACT)

This step is used to obtain the exact query results without any sampling error.

1. **Exact Subgraph Matching:** Run `exact_subgraph_match.py`. This script invokes the underlying C++ engine to perform exact subgraph matching and saves the intermediate results.
2. **Predicate Verification and Aggregation:** Run `EXACT.py`. It uses the corresponding Oracle predicates of the queries to verify the matching results and performs the final aggregation calculation (supports `agg_mode={count, sum}`).

### B. $\text{PROXY}$ Experiments for `count` and `sum`

Conduct experimental validation for the `count` and `sum` aggregation modes.

1. **Preprocessing and Weight Estimation:** Run `Projection_Sampling_and_Weight_Estimation_Runner.py` to construct the projected sampling space $\hat{\Psi}$ and the weight estimator $\hat{w}(\psi)$ defined in the paper.
2. **Core Performance and Ablation Studies (RQ1, RQ2 & RQ4):** Run `Proxy_Guided_Stratified_Importance_Sampling_Runner.py` to perform stratified importance sampling on $\hat{\Psi}$:
   
   * **For RQ1 & RQ2 (Core Performance Comparison):**
     Enable only the PROXY method and configure the target sampling rate gradient:
     ```python
     methods_map = {
         "PROXY": sampler.run_possa,
     }
     --target_ticks 0.01,0.05,0.075,0.1,0.125,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
     ```
     Output saved to: `allocation_strategy_comparison_{agg_mode}.csv`
     
   * **For RQ4 (Component Ablation Study):**
     Run various ablation variants at a fixed sampling rate:
     ```python
     methods_map = {
         "PROJ": sampler.run_baseline_uniform,
         "PO": sampler.run_baseline_proxy,
         "WO": sampler.run_baseline_weight_only,
         "MAB": sampler.run_mab_sampling,
         "PROXY": sampler.run_possa,
     }
     --target_ticks 0.1
     ```
     Output saved to: `allocation_strategy_comparison_ablation_{agg_mode}.csv`

3. **Robustness and Degradation Analysis (RQ3):** Run `Sensitivity_single_predicate_Runner.py` and `Sensitivity_multi_predicate_comparation.py` to evaluate the robustness of the algorithm under single-predicate degradation and complex multi-predicate scenarios.
   Output saved to: `proxy_quality_ablation_{agg_mode}.csv`


### C. $\text{PROXY}$ Experiments for `avg`

Based on the ratio estimator proposed in **Theorem 6**, the `avg` results do not require re-running the C++ engine. Instead, they are obtained by offline synthesis of the completed `count` and `sum` experimental data.

1. **Synthesizing Ground Truth:**
   Calculate the `avg` ground truth according to the formula $\tau_{\text{avg}} = \tau_{\text{sum}} / \tau_{\text{count}}$.
   * **Input:** `T_true_*_sum.json` and `T_true_*_count.json`
   * **Output:** Generates `T_true_*_avg.json`

2. **Synthesizing Experimental Results and Error Calculation:**
   Run the ratio alignment script (following the logic $\hat{\tau}_{\text{avg}} = \hat{\tau}_{\text{sum}} / \hat{\tau}_{\text{count}}$):
   * **Core & Ablation Strategies (RQ1, RQ2, RQ4):** Merge `allocation_strategy_comparison_{count,sum}.csv` $\rightarrow$ obtain `allocation_strategy_comparison_avg.csv`
   * **Baseline Fastest-Oracle:** Merge `FastestO_budget_curve_{count,sum}.csv` $\rightarrow$ obtain `FastestO_budget_curve_avg.csv`
   * **Baseline Exact-structureO:** Merge `Exact_structureO_budget_curve_{count,sum}.csv` $\rightarrow$ obtain `Exact_structureO_budget_curve_avg.csv`

3. **Adaptive Data Column Extraction:**
   The synthesis script features an automatic adaptation mechanism to extract the necessary sampling statistical columns based on dataset conventions:
   * **Parler-style Datasets:** Automatically extracts `n_post` and `n_comment`
   * **Amazon-style Datasets:** Automatically extracts `n_product` and `n_review`


### D. Time-Equivalence Protocol: Calculating the Equivalent Virtual Budget $B_{\text{virtual}}$

To ensure a fair comparison of algorithm efficiency, we project the budget into a unified time dimension.

1. **Extracting Execution Counts:** Read `allocation_strategy_comparison_{count,sum}.csv` to extract the actual execution counts $N_{oi}$ and $N_{pi}$ of the Oracle and Proxy models at a specified sampling rate $\alpha$.
2. **Converting to Virtual Budget:** Combined with the average inference latencies ($c_i$ and $c_p^i$) of the Oracle and Proxy models on the current dataset, calculate the total time-equivalent virtual budget $B_{\text{virtual}}$.


### E. Evaluating Baselines (ENUM & FASTEST-ORACLE) Based on $B_{\text{virtual}}$

Under the unified virtual budget, evaluate core metrics such as the Absolute Average Error (AAE) of the estimated $\hat{\tau}$ for other baseline methods.

1. **Evaluating ENUM Baseline:**
   Run `ENUM.py` to calculate the performance of the ENUM method under the current query workload based on the equivalent budget (supports `agg_mode={count, sum}`).
   
2. **Evaluating FASTEST-ORACLE Baseline:**
   Run `FASTEST-ORACLE.py` to call the corresponding algorithm in the underlying C++ engine to estimate the target value $\hat{\tau}$ under the same budget (supports `agg_mode={count, sum}`).
   * The outputs will be saved to the `FastestO_budget_curve_{agg_mode}.csv` file in the `results/efficiency` folder.
   * This file records the estimated value $\hat{\tau}$ for each query $Q$ after running independently for $k$ times (e.g., $k=5$ or $10$) under the specified dataset and sampling rate.


### F. Calculating Theoretical Performance Upper Bounds (WEE Asymptote)

1. Run `WEE.py` to calculate the Worst-case Execution Efficiency (WEE) theoretical asymptotic metrics for each target dataset.
```
