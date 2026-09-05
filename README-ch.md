***

# 基于代理引导采样的机器学习谓词近似图聚合

# 评测与复现指南 (V-1.01 持续更新中...)

本仓库提供了复现论文中报告的所有实验结果所需的完整代码与配置。用户可以选择**通过主脚本实现全自动化复现**，也可以**按模块分步执行整个流程**。

> **开销提示：** 直接调用大语言模型（LLM）或深度神经网络（Oracle 模型）进行实时推理会产生巨大的计算与时间开销。为了便于快速复现与验证，**我们已将所有查询的全部 Oracle 和 Proxy 验证结果预缓存至各个数据集的 `csv_data/` 目录中，支持即开即用。**

---

> ### (1) 即时验证：1 分钟图表复现
> **无需等待耗时的分层重要性采样与图匹配！** 我们已将论文所有实验的真实评测结果完整持久化在每个数据集的 `datasets/{workload}/results/efficiency/` 目录下。
> 如果您希望**立即验证并复现论文中的所有实验图表（RQ1–RQ4）以及统计显著性指标**：
> 1. 启动 Jupyter Notebook：
>    ```bash
>    jupyter notebook pythonProject/src/RQS_plots/RQX.ipynb
>    ```
> 2. 点击 **`Run`**，即可直接加载预缓存数据，瞬间渲染论文中展示的所有高分辨率矢量图表！

---

> ### (2) 技术报告与补充理论证明
> 本仓库提供了一份扩展技术报告（[**`Technical_Report.pdf`**](./Technical_Report_0_9.pdf)），以回应审稿人意见，并提供完整的数学推导、系统模型和扩展基准测试。
> 
> **技术报告中的核心内容与已完成的证明：**
> * **定理 9 的严格证明（\textsf{AVG} 一致性与渐近偏差）：** 给出了完整的数学证明，利用两步正交误差分解和精确的比率代数恒等式，在联合渐近体制（$K, m \to \infty, m = o(K)$）下证明了真实的一致性以及 $\mathcal{O}(m^{-1})$ 的渐近偏差。
> * **均匀树采样证明：** 给出了完整的归纳法证明，证明在候选空间（Candidate Space, CS）概要上的自顶向下采样是严格均匀（$P(h) = 1/|\Omega|$）且无损的。
> * **精细化的端到端代价模型：** 详细刻画了延迟特性，将串行内存 CPU 图操作（当 $K=60,000$ 时，$C_{\text{CS}} + K \cdot c_{\text{tree}} \approx 800\text{--}1000\text{ ms}$）与繁重的 GPU 机器学习推理（$m \cdot \bar{c}_{\text{eff}}$）进行解耦。
> * **案例研究：** 多模态电商图。
> 
> *📌 正在扩充的内容（积极更新中）：* 形式化理论收支平衡分析（$S > 1/(1-\alpha)$）、详尽的工作负载特征表（>90% 环状查询拓扑、选择率及投影缩减比 $|\hat{\Psi}|/|\Phi|$）以及细粒度的缓存命中率性能分析。

---

> ### (3) 完整模型库与基准测试指南（Oracle 对比 Proxy）
> 关于单张 RTX 3090 GPU 上所有**自然语言推理（NLI）**、**文本情绪/情感分析（TE）**和**计算机视觉（CV）**模型的详细架构、参数量、$F_1$ 分数、微调方案及吞吐量基准测试，请参阅专门的指南：
> 👉 [**`模型库文档 (./Models/README.md)`**](./Models/README.md)

---

> ### (4) 代码库设计与工作流建议
> 1. **复现性与调试设计**：当前代码库处于学术开源阶段。为了便于各阶段的状态验证、算法逻辑调试，并支持完整的数据复现流程，代码保留了**详细的中间结果落盘（持久化）与 I/O 校验逻辑**。
> 2. **推荐执行策略**：
>    * **步骤 1（结构投影与权重物化）**：运行 `Projection_Sampling_and_Weight_Estimation_Runner.py`。每个工作负载与聚合类型（`COUNT` / `SUM`）**仅需执行一次**。此步骤用于确定拓扑投影空间 $\hat{\Psi}$，并离线物化所有投影的结构扩展权重 $\hat{w}(\psi)$。
>    * **步骤 2（采样算法与基线评测）**：完成物化 $\hat{\Psi}$ 后，您可以高效地多次运行 `Proxy_Guided_Stratified_Importance_Sampling_Runner.py` 以及各类基线脚本，从而快速评估不同的采样策略、消融变体和多次运行的随机方差。
> 3. **后续重构计划**：我们的团队将持续对代码库进行模块化解耦、精简 I/O 流程，并进一步优化执行效率。敬请关注后续更新。

---

## (5) 环境配置

在运行代码之前，请按照以下步骤配置 Python 虚拟环境并编译底层的 C++ 采样引擎。

### 1. Python 环境 (Conda)
推荐使用 Python 3.10：
```bash
# 1. 创建并激活 conda 虚拟环境
conda create -n iogs python=3.10 -y
conda activate iogs

# 2. 一键安装所有依赖项
pip install -r requirements.txt
```

### 2. C++ 采样引擎编译
底层的图匹配和候选空间树采样引擎采用 C++20 开发，依赖 CMake、Boost 和 GSL：
```bash
# 1. 安装系统依赖 (Ubuntu/Debian)
conda install -c conda-forge cmake gxx_linux-64 boost gsl -y

# 2. 编译生成 Fastest 二进制可执行文件（项目中已包含预编译文件；以下步骤展示如何重新编译）
cd cProject
mkdir -p build && cd build
cmake ..
make -j$(nproc)
cd ../..
```
*编译成功后，二进制可执行文件将位于 `cProject/build/Fastest`。*

---

## 0. 实验基础：数据集与机器学习谓词架构

在运行实验前，请先熟悉底层的图数据集、合成查询负载以及机器学习谓词架构。

---

### 0.1. 数据集与查询负载

实验在三个真实世界的属性/多模态图数据集上进行：

1. **`Parler`：** 文本属性社交网络数据集，包含 **3 种不同的顶点标签/类型**：用户（`user`）、帖子（`post`）和评论（`comment`）。
2. **`Parler-E`：** 改编自 `Parler`。通过将顶点类型细分为互斥的子类型来加剧标签稀疏性，将标签集扩展至 **6 种不同的顶点标签**。这大幅降低了查询选择率，用于评估极端低选择率场景下的性能。
3. **`Amazon`：** 包含用户（`user`）、文本评论（`review`）和商品图像（`product`）的多模态异构图。通过将顶点类型随机划分为互斥子类型，共具有 **11 种不同的顶点标签**，用于评估结构复杂性与多模态特征融合能力。

#### 查询生成与聚合约束
查询图 $Q$ 是通过在数据图上进行**随机游走**生成的，并附加了聚合属性和机器学习谓词：
* **查询拓扑：** 随机游走生成了多样的结构模体。由于底层数据图的高连通性，**复杂环状结构在负载中占主导地位（>90% 为环）**，而树状、路径和星状模体仅占少数。
* **聚合约束：** 为确保在 $Q$ 上执行有效的 `SUM` 和 `AVG` 聚合，每个查询必须包含至少一个数值属性：
  * **`Parler` / `Parler-E`：** `post` 顶点的 `upvotes`（点赞数）属性。
  * **`Amazon`：** `product` 顶点的 `price`（价格）或 `rating`（评分）属性。
* **查询规模与谓词配置：**
  * **`Parler`：** 包含 **~245** 个单谓词查询（$|V(Q)| \in [4, 8]$，$k=1$），谓词随机分配给 1 个 `post` 或 `comment` 顶点。
  * **`Parler-E`：** 包含 **~115** 个多谓词复合查询（$|V(Q)| \in [4, 8]$，$k \ge 2$），谓词同时分配给至少 1 个 `post` 顶点和至少 1 个 `comment` 顶点。
  * **`Amazon`：** 包含 **~750** 个多模态复合多谓词查询（$|V(Q)| \in [3, 8]$，$k \ge 2$），谓词分配给至少 1 个 `product` 图像顶点和至少 1 个 `review` 文本顶点。

---

### 0.2. 机器学习谓词：Oracle 模型对比 Proxy 模型

每个原子机器学习谓词 $\mathcal{P}_i$ 都配备了一个**高精度的 Oracle 模型**（用于精确无偏的验证）和一个**轻量级的 Proxy 模型**（用于高效近似打分并指导分层重要性采样）：

| 数据集 | 顶点类型 | 谓词语义 ($\mathcal{P}$) | Oracle 模型 (参数量) | Proxy 模型 (参数量) | Proxy $F_1$ | Proxy 加速比 |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| **`Amazon`** | `product` | **图像纹理分类**<br>*(木质/塑料/金属/织物/玻璃？)* | `siglip-so400m-patch14-384`<br>*(878M)* | `siglip-base`<br>*(84M)* | 0.7546 | **$26.8\times$** |
| | `review` | **情感分析**<br>*(正面/负面？)* | `roberta-large-sst2`<br>*(355M)* | `bert-mini-finetuned-sst2`<br>*(11M)* | 0.8890 | **$22.1\times$** |
| **`Parler`** /<br>**`Parler-E`** | `post` | **观点推断**<br>*(支持/反对唐纳德·特朗普？)* | `deberta-v2-xxlarge-mnli`<br>*(1.5B)* | `deberta-v3-base-mnli`<br>*(184M)* | 0.7720 | **$42.5\times$** |
| | `comment` | **情感分析**<br>*(正面/负面？)* | `roberta-large-sst2`<br>*(355M)* | `bert-mini-finetuned-sst2`<br>*(11M)* | 0.7876 | **$22.1\times$** |

* **代理质量梯队（$M_{P1} \sim M_{P4}$）：** 为评估 $\text{PROXY}$ 对代理精度的敏感性（RQ3），我们通过微调或采用更轻量的架构，为每个任务构建了 4 个代理质量梯队。它们的相对 $F_1$ 分数在 $[0.65, 0.89]$ 范围内单调递减，而推理速度单调递增。

---

### 0.3. 硬件配置

所有实验均在配置如下的高性能服务器上进行：
* **操作系统：** Ubuntu 22.04 LTS
* **处理器 (CPU)：** 双路 Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz
* **内存 (RAM)：** 503 GB
* **显卡 (GPU)：** 4 张 NVIDIA GeForce RTX 3090 GPU（每张 24GB 显存）

> **⚠️ 注意：** 每个工作负载都包含数百个复杂的子图同构匹配和机器学习谓词评估。运行完全精确匹配需要数小时，并且每个工作负载至少需要 100 GB 的可用磁盘空间。**我们为所有工作负载提供了预先计算好的真实值（Ground Truth, GT）文件**，允许您跳过昂贵的全图匹配和详尽的 Oracle 评估。

---

### 0.4. 代码库结构

本项目底层采用高性能 **C++ 采样引擎（`cProject`）**，上层采用 **Python 代理引导采样框架（`pythonProject`）**：

> 💡 **实现说明（理论与系统优化）：** 
> 为了在实际运行中获得更低的方差，我们的执行引擎采用了**不放回采样（WOR）**，并在分配的预算超出分层大小时合并微小分层（$|S_i| < 2$）。这是一种严格的系统级安全回退机制，旨在防止越界错误；而我们的理论证明为了求得闭式解，建模时采用了*有放回采样（WR）*。论文中报告的所有表格均与该代码库完全对应。

```text
PROXY/
├── cProject/                                   # [已更新] [C++ 核心引擎] CS 构建、树采样及语义投影权重估计 
│   ├── build/                                  # 预编译二进制目录（包含已编译好的 'Fastest' 可执行文件）
│   ├── driver/                                 # C++ 入口点 (subgraph-cardinality-estimation.cc)
│   ├── lib/                                    # 图数据结构、CS 构建器、均匀树采样器等
│   └── CMakeLists.txt                          # CMake 配置文件
│
├── datasets/                                   # [已更新] [数据与结果存储] 三个工作负载的数据图、查询图及结果
│   ├── parler/                                 # Parler 单谓词工作负载 (data_graph / query_graph / ground_truth / results)
│   ├── parler-e/                               # Parler-E 多谓词扩展工作负载
│   └── amazon/                                 # Amazon 多模态图工作负载
│
├── Model/                                      # [更新中] [机器学习模型库] Oracle 与 Proxy 模型权重及配置
│
├── pythonProject/                              # [Python 实验框架] PROXY 采样算法、基线评测与绘图
│   └── src/
│       ├── algorithms/                         # PROXY 核心算法实现
│       │   ├── exact_subgraph_match.py         # 精确子图同构匹配脚本
│       │   ├── compute_truth.py                # 真实值 (GT) 计算与谓词验证类
│       │   └── proxy_sample.py                 # 代理引导分层采样 (POSSA) 及其消融变体
│       │
│       ├── baseline/                           # 基线算法库
│       │   ├── ...                             # 核心基线方法 (ENUM, FASTEST-ORACLE, WEE 等)
│       │   └── ...                             # 对比方法 (如 PRO-ABAE, PSF 级联过滤器等)
│       │
│       ├── runner/                             # 执行运行器
│       │   └── ...                             # 连接 C++ 与 Python 模块的执行脚本
│       │
│       └── RQS-plot/                           # 论文图表可视化与绘图脚本 (RQ1 ~ RQ4)
│
├── scripts/                                    # [自动化脚本] 一键基准测试与复现 Shell 脚本
│   ├── run_get_all_structural_matching.sh      # [已更新] 精确子图匹配：获取所有 structural_matching         
│   ├── run_RQ1.sh                              # [已更新] RQ1: PROXY 端到端效率与运行耗时分解评测      
│   ├── run_RQ2.sh                              # [已更新] RQ2: PROXY 在各预算步长下的精度收敛情况 (COUNT, SUM, AVG) 
│   ├── run_RQ3.sh                              # [已更新] RQ3: 代理模型质量敏感性与鲁棒性消融          
│   ├── run_RQ4.sh                              # [已更新] RQ4: 分配策略消融 (UN, PO, WO, MAB, POSS)
│   ├── run_all_gt.sh                           # [已更新] 真实值计算
│   ├── run_Enum.sh                             # [已更新] RQ1&RQ2: 基线方法：论文中的 Algorithm 1        
│   ├── run_Fastest-Oracle.sh                   # [已更新] RQ1&RQ2: 基线方法：并行 FaSTest-Oracle 运行器
│   ├── run_WEE.sh                              # [已更新] RQ2: 基线方法：权重估计误差 (WEE) 下限   
│   ├── run_Pro_Abae.sh                         # [已更新] Rebuttal: 基线方法：投影空间 + ABAE   
│   ├── run_Pro_Cacade_Filter.sh                # [已更新] Rebuttal: 基线方法：投影空间 + 级联过滤器 (SUPG.ScaleDoc 风格)              
│   ├── run_parler_{count,sum}.sh               # [已更新] Parler（单谓词）模块化一键脚本         
│   ├── run_parler_e_{count,sum}.sh             # [已更新] Parler-E（多谓词）模块化一键脚本        
│   ├── run_amazon_{count,sum}.sh               # [已更新] Amazon（多模态图）模块化一键脚本     
│   ├── run_all_avg.sh                          # [已更新] 基于 COUNT 和 SUM 结果的离线比率式 AVG 综合    
└── ...                                         
```

---

## 1. 一键复现

您既可以通过运行跨所有数据集的总主脚本来复现实验结果，也可以执行针对特定工作负载的专属一键脚本。

> ⚠️ **复现性重要提示：**
>
> 本仓库默认**已打包所有预先计算好的精确基数与真实值映射（`results/T_true_*.json`）**，使您可以立即运行所有下游的采样与近似基准测试（RQ1–RQ4）。
>
> **但是，如果您希望从头生成真实值，或复现精确枚举基线（ENUM / EXACT）：**
> 获取所有精确子图匹配属于 #P-hard 问题，伴随巨大的内存和 CPU 开销（每个工作负载可能需要数小时）。我们提供了一个**交互式的单工作负载生成脚本** `run_get_all_structural_matching.sh`，以便每次安全地针对单个数据集执行精确匹配：

```bash
cd pythonProject/scripts
chmod +x *.sh
# 获取数据集中所有查询的结构匹配（不进行 Oracle 验证，即不验证机器学习谓词）。一次只能选择并运行一个工作负载。
./run_get_all_structural_matching.sh

# EXACT: 获取真实值 (Ground Truth)
./run_all_gt.sh

# 基线 ENUM: 详情请参考论文中的 Algorithm 1
./run_Enum.sh
```
---

### 1.1. 主脚本（所有工作负载与聚合类型）
若要在所有三个工作负载（`Parler`、`Parler-E`、`Amazon`）上自动运行整个流水线（包括 C++ 权重物化和分层采样），并生成绘图所需的所有数据：
<!-- 
```bash
# 确保已激活 conda 环境
conda activate iogs

# 赋予执行权限并运行主脚本
chmod +x scripts/run_all_experiments.sh
bash scripts/run_all_experiments.sh
``` -->

---

### 1.2. 针对特定工作负载的一键脚本
如果您希望评估或调试特定数据集，而不想运行耗时数小时的完整测试套件，可以使用我们提供的专用一键自动化脚本。

#### 示例：Parler（`COUNT` 模式）
脚本 `run_parler_count.sh` 实现了 **Parler** 工作负载在 `COUNT` 模式下的端到端全自动流程：
1. **步骤 1（离线投影与物化）：** 调用 C++ 引擎执行均匀树采样，估计投影扩展权重 $\hat{w}(\psi)$，并物化紧凑的核心实例空间。
2. **步骤 2（在线代理采样）：** 在预算梯度 $\alpha \in [1\%, 90\%]$ 上执行代理引导的分层重要性采样，每个步长独立运行 5 次。

```bash
# 1. 激活 conda 环境
conda activate iogs
cd pythonProject/scripts
# 2. 赋予执行权限
chmod +x *.sh

# 3. 执行一键脚本

# RQ 专项复现脚本 (RQ1 – RQ4)

# 生成 RQ1 的 PROXY 结果。在 `pythonProject/src/RQs_plots/RQ1&RQ2.ipynb` 或 `pythonProject/src/RQs_plots/Rebuttal_exp_result.ipynb` 中可视化。
./run_RQ1.sh 

# 生成 RQ2 的 PROXY 结果。在 `pythonProject/src/RQs_plots/RQ1&RQ2.ipynb` 中可视化。
./run_RQ2.sh 

# 生成 RQ3 的所有结果。在 `pythonProject/src/RQs_plots/RQ3:Sensitivity.ipynb` 中可视化。
./run_RQ3.sh

# 生成 RQ4 的所有结果。在 `pythonProject/src/RQs_plots/RQ4:Ablation.ipynb` 中可视化。
./run_RQ4.sh


# 基线: Fastest-Oracle，生成 RQ1&RQ2 的 `Fastest-Oracle` 结果。在 `pythonProject/src/RQs_plots/RQ1&RQ2.ipynb` 中可视化
./run_Fastest-Oracle.sh

# 基线: 权重估计误差 (WEE) 下限
./run_WEE.sh

# Rebuttal 基线: 投影空间 + ABAE。在 `pythonProject/src/RQs_plots/Rebuttal_exp_result.ipynb` 中可视化
# Rebuttal 后该方法经过了系统性改进，其 ARE（平均相对误差）有一定改善，但仍落后于 PROXY
./run_Pro_Abae.sh

# Rebuttal 基线: 投影空间 + 级联过滤器 (SUPG.ScaleDoc 风格)。在 `pythonProject/src/RQs_plots/Rebuttal_exp_result.ipynb` 中可视化
./run_Pro_Cacade_Filter.sh 


./run_all_avg.sh

# 4. 执行单个工作负载测试

./run_parler_count.sh
./run_parler_sum.sh

./run_parler_e_count.sh
./run_parler_e_sum.sh

./run_amazon_count.sh
./run_amazon_sum.sh

./run_all_avg.sh
```

**生成的输出文件：**
  ```text
  例如：datasets/parler/results 目录下：

  1. /efficiency/allocation_strategy_comparison_count.csv: Proxy 结果文件，存储每个查询在各采样率下的估计值和 Oracle 调用次数，用于 RQ1 & RQ2 

  2. FastestO_budget_curve_count.csv: RQ1 & RQ2 的重要基线 FASTEST-ORACLE 结果

  3. allocation_strategy_comparison_ablation_count.csv: 消融实验结果文件，存储每个消融实验下查询在各采样率下的估计值，用于 RQ4
  ```

---

---
## 以下是正在整理和补充的实验细节。上述所有内容现已就绪，完全能够复现论文中的实验以及 Rebuttal 阶段的补充实验。