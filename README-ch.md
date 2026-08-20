***

# 基于代理模型引导采样的机器学习谓词近似图聚合

# 评估与复现指南（V-0.99 持续更新中...）

本仓库提供了复现论文中所有实验结果所需的完整代码与配置文件。用户既可以通过**主控脚本全自动化一键复现**，也可以**按模块逐步执行实验流水线**。

> **开销说明：** 直接调用大语言模型（LLM）或深度神经网络（Oracle 模型）进行实时推理会产生高昂的计算与时间成本。为了便于快速复现与验证，**我们已将所有查询对应的 Oracle 与 Proxy 验证结果预先缓存至各个数据集的 `csv_data/` 目录中，支持即开即用。**

---

> ### (1) 即时验证：1 分钟图表复现
> **无需耗时等待分层重要性采样与图匹配！** 我们已将论文全部实验的真实评测结果完整持久化在各个数据集的 `datasets/{workload}/results/efficiency/` 目录下。
> 如果您希望**立即验证并复现论文中的所有实验图表（RQ1–RQ4）及统计显著性指标**：
> 1. 启动 Jupyter Notebook：
>    ```bash
>    jupyter notebook pythonProject/src/RQS/RQX.ipynb
>    ```
> 2. 点击 **`Run`** 即可直接加载预存数据，瞬间生成论文中展示的所有高分辨率矢量图！

---

> ### (2) 技术报告与补充理论证明
> 有关论文中提及的**案例研究（Case Study）**以及**均匀树采样的完整理论证明（Tree Sampling Proofs）**细节，请参阅仓库根目录下的技术报告：[**`TR.pdf`**](./TR0.1.pdf)。

> ### (3) 代码库架构设计与工作流建议
> 1. **可复现性与调试设计**：当前代码库处于学术开源阶段。为了便于各阶段的状态验证、算法逻辑调试以及支持完整的数据复现流水线，代码中保留了**详尽的中间结果磁盘持久化与 I/O 校验逻辑**。
> 2. **推荐执行策略**：
>    * **步骤 1（结构投影与权重物化）**：运行 `Projection_Sampling_and_Weight_Estimation_Runner.py`。每个工作负载与聚合类型（`COUNT` / `SUM`）**仅需执行一次**。该步骤会固定拓扑投影空间 $\hat{\Psi}$，并离线物化所有投影的结构扩展权重 $\hat{w}(\psi)$。
>    * **步骤 2（采样算法与基线评估）**：在完成 $\hat{\Psi}$ 物化后，您可以多次高效运行 `Proxy_Guided_Stratified_Importance_Sampling_Runner.py` 及各类基线脚本，以快速评估不同的采样策略、消融变体和多轮随机方差。
> 3. **后续重构计划**：我们的团队将持续对代码库进行模块化和解耦，优化 I/O 流程，并提升执行效率，敬请期待后续更新。

---

## (4) 环境配置

在运行代码之前，请按照以下步骤配置 Python 虚拟环境并编译底层的 C++ 采样引擎。

### 1. Python 环境配置 (Conda)
推荐使用 Python 3.10：
```bash
# 1. 创建并激活 conda 虚拟环境
conda create -n iogs python=3.10 -y
conda activate iogs

# 2. 一键安装所有依赖项
pip install -r requirements.txt
```

### 2. C++ 采样引擎编译
底层的图匹配与候选空间树采样引擎采用 C++20 开发，依赖 CMake、Boost 和 GSL：
```bash
# 1. 安装系统依赖项 (Ubuntu/Debian)
conda install -c conda-forge cmake gxx_linux-64 boost gsl -y

# 2. 编译生成 Fastest 二进制可执行文件（仓库内已包含预编译二进制文件；以下步骤展示重新编译过程）
cd cProject
mkdir -p build && cd build
cmake ..
make -j$(nproc)
cd ../..
```
*编译成功后，生成的二进制可执行文件位于 `cProject/build/Fastest`。*

---

## 0. 实验基础：数据集与机器学习谓词架构

在开始运行实验前，请先熟悉底层的图数据集、合成查询工作负载以及机器学习谓词架构。

---

### 0.1. 数据集与查询工作负载

实验在三个真实的属性/多模态图数据集上进行：

1. **`Parler`：** 包含文本属性的社交网络数据集，涵盖 **3 种不同的顶点标签/类型**：用户（`user`）、帖子（`post`）和评论（`comment`）。
2. **`Parler-E`：** 改编自 `Parler`。通过将顶点类型细分为互不相交的子类型来加剧标签稀疏性，将标签集扩展至 **6 种不同的顶点标签**。这极大降低了查询选择率，用于评估极端低选择率场景下的算法性能。
3. **`Amazon`：** 包含用户（`user`）、文本评论（`review`）和商品图片（`product`）的多模态异构图。通过将顶点类型随机划分为不相交的子类型，形成了 **11 种不同的顶点标签**，用于评估结构复杂性与多模态特征融合能力。

#### 查询生成与聚合约束
查询图 $Q$ 是通过在数据图上进行**随机游走（Random Walks）**生成的，并附加了聚合属性与机器学习谓词：
* **聚合约束：** 为了确保在 $Q$ 上进行有效的 `SUM` 和 `AVG` 聚合，每个查询必须至少包含一个数值属性：
  * **`Parler` / `Parler-E`：** `post` 顶点的 `upvotes`（点赞数）属性。
  * **`Amazon`：** `product` 顶点的 `price`（价格）或 `rating`（评分）属性。
* **查询规模与谓词配置：**
  * **`Parler`：** 包含 **245** 个单谓词查询（$|V(Q)| \in [4, 8]$, $k=1$），谓词随机分配给 1 个 `post` 或 `comment` 顶点。
  * **`Parler-E`：** 包含 **115** 个多谓词复合查询（$|V(Q)| \in [4, 8]$, $k \ge 2$），谓词同时分配给至少 1 个 `post` 顶点和至少 1 个 `comment` 顶点。
  * **`Amazon`：** 包含 **750** 个多模态复合多谓词查询（$|V(Q)| \in [3, 8]$, $k \ge 2$），谓词分配给至少 1 个 `product` 图像顶点和至少 1 个 `review` 文本顶点。

---

### 0.2. 机器学习谓词：Oracle 与 Proxy 模型

每个原子机器学习谓词 $\mathcal{P}_i$ 均配备了一个**高精度 Oracle 模型**（用于精确、无偏的验证）和一个**轻量级 Proxy 模型**（用于高效近似打分并指导分层重要性采样）：

| 数据集 | 顶点类型 | 谓词语义 ($\mathcal{P}$) | Oracle 模型 (参数量) | Proxy 模型 (参数量) | Proxy $F_1$ | Proxy 加速比 |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| **`Amazon`** | `product` | **图像纹理分类**<br>*(木质/塑料/金属/织物/玻璃?)* | `siglip-so400m-patch14-384`<br>*(878M)* | `siglip-base`<br>*(84M)* | 0.7546 | **$26.8\times$** |
| | `review` | **情感分析**<br>*(积极/消极?)* | `roberta-large-sst2`<br>*(355M)* | `bert-mini-finetuned-sst2`<br>*(11M)* | 0.8890 | **$22.1\times$** |
| **`Parler`** /<br>**`Parler-E`** | `post` | **观点推断**<br>*(支持/反对 Donald Trump?)* | `deberta-v2-xxlarge-mnli`<br>*(1.5B)* | `deberta-v3-base-mnli`<br>*(184M)* | 0.7720 | **$42.5\times$** |
| | `comment` | **情感分析**<br>*(积极/消极?)* | `roberta-large-sst2`<br>*(355M)* | `bert-mini-finetuned-sst2`<br>*(11M)* | 0.7876 | **$22.1\times$** |

* **Proxy 质量分级 ($M_{P1} \sim M_{P4}$)：** 为了评估 $\text{PROXY}$ 对代理精度的敏感性（RQ3），我们通过微调或采用更简化的架构为每个任务构建了 4 个代理质量梯度。它们的相对 $F_1$ 分数在 $[0.65, 0.89]$ 范围内单调递减，而推理速度则单调递增。

---

### 0.3. 硬件环境

所有实验均在一台具备以下规格的高性能服务器上进行：
* **操作系统：** Ubuntu 22.04 LTS
* **处理器 (CPU)：** 双路 Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz
* **内存 (RAM)：** 503 GB
* **显卡 (GPU)：** $4 \times$ NVIDIA GeForce RTX 3090 GPU（每张显卡 24GB 显存）

> **⚠️ 注意：** 每个工作负载都包含数百个复杂的子图同构匹配和机器学习谓词评估。运行完整的精确匹配需要数小时，且每个工作负载至少需要 100 GB 的可用磁盘空间。**我们为所有工作负载提供了预先计算好的真值（Ground Truth, GT）文件**，使您可以直接跳过高昂的全图匹配与详尽的 Oracle 评估。

---

### 0.4. 代码仓库结构

项目结构分为底层的**高性能 C++ 采样引擎（`cProject`）**与上层的 **Python 代理引导采样框架（`pythonProject`）**：

```text
PROXY/
├── cProject/                                   # [已更新] [C++ 核心引擎] 候选空间(CS)构建、树采样与语义投影权重估计 
│   ├── build/                                  # 预编译二进制目录（包含编译好的 'Fastest' 可执行文件）
│   ├── driver/                                 # C++ 入口文件 (subgraph-cardinality-estimation.cc)
│   ├── lib/                                    # 图数据结构、CS 构建器、均匀树采样器等
│   └── CMakeLists.txt                          # CMake 配置文件
│
├── datasets/                                   # [已更新] [数据与结果存储] 三个工作负载的数据图、查询图及结果
│   ├── parler/                                 # Parler 单谓词工作负载 (data_graph / query_graph / ground_truth / results)
│   ├── parler-e/                               # Parler-E 多谓词扩展工作负载
│   └── amazon/                                 # Amazon 多模态异构图工作负载
│
├── Model/                                      # [更新中] [机器学习模型库] Oracle 和 Proxy 模型权重与配置
│
├── pythonProject/                              # [Python 实验框架] PROXY 采样算法、基准模型评估与绘图
│   └── src/
│       ├── algorithms/                         # PROXY 核心算法实现
│       │   ├── exact_subgraph_match.py         # 精确子图同构匹配脚本
│       │   ├── compute_truth.py                # 真值 (GT) 计算与谓词验证类
│       │   └── proxy_sample.py                 # 代理引导分层采样 (POSSA) 及其消融变体
│       │
│       ├── baseline/                           # 基线算法库
│       │   ├── ...                             # 核心基线方法 (ENUM, FASTEST-ORACLE, WEE 等)
│       │   └── ...                             # 对比方法 (如 PRO-ABAE, PSF 级联过滤器等)
│       │
│       ├── runner/                             # 执行入口 (Runners)
│       │   └── ...                             # 连接 C++ 与 Python 模块的执行脚本
│       │
│       └── RQS-plot/                           # 论文图表可视化与绘图脚本 (RQ1 ~ RQ4)
│
├── scripts/                                    # [自动化脚本] 一键测试与复现 Shell 脚本
│   ├── run_get_all_structural_matching.sh      # [已更新] 精确子图匹配：获取所有 structural_matching         
│   ├── run_RQ1.sh                              # [已更新] RQ1: PROXY 端到端效率与运行时分解基准测试      
│   ├── run_RQ2.sh                              # [已更新] RQ2: PROXY 在各预算点上的精度收敛性 (COUNT, SUM, AVG) 
│   ├── run_RQ3.sh                              # [已更新] RQ3: 代理质量敏感性与鲁棒性消融实验          
│   ├── run_RQ4.sh                              # [已更新] RQ4: 分配策略消融实验 (UN, PO, WO, MAB, POSS)
│   ├── run_all_gt.sh                           # [已更新] 真值 (Ground Truth) 计算
│   ├── run_ENUM.sh                             # [已更新] RQ1&RQ2: 基线方法：论文中的 Algorithm 1        
│   ├── run_Fastest-Oracle.sh                   # [已更新] RQ1&RQ2: 基线方法：并行 FaSTest-Oracle 执行器
│   ├── run_WEE.sh.sh                           # [已更新] RQ2:     基线方法：权重估计误差 (WEE) 理论下界   
│   ├── run_Pro_Abae.sh.sh                      # [已更新] Rebuttal: 基线方法：投影空间 + ABAE   
│   ├── run_Pro_Cacade_Filter.sh                # [已更新] Rebuttal: 基线方法：投影空间 + 级联过滤器 (SUPG.ScaleDoc 风格)              
│   ├── run_parler_{count,sum}.sh               # [已更新] Parler (单谓词) 模块化一键脚本         
│   ├── run_parler_e_{count,sum}.sh             # [已更新] Parler-E (多谓词) 模块化一键脚本        
│   └── run_amazon_{count,sum}.sh               # [已更新] Amazon (多模态图) 模块化一键脚本     
│   ├── run_all_avg.sh                          # [已更新] 基于 COUNT 和 SUM 结果的离线比率式 AVG 合成脚本    
└── ...                                         
```

---

## 1. 一键复现

您可以通过跨数据集运行总控脚本，或执行特定工作负载的专用一键脚本来复现实验结果。

> ⚠️ **复现重要提示：**
>
> 仓库默认**已打包所有预计算好的精确基数与真值映射文件**（`results/T_true_*.json`），使您可以立即运行所有下游采样和近似基准测试（RQ1–RQ4）。
>
> **但是，如果您希望从头开始生成真值，或复现完全枚举基线（ENUM / EXACT）：**
> 获取所有精确子图匹配属于 #P-hard 问题，需要占用大量内存和 CPU 开销（每个工作负载可能需要耗费数小时）。我们提供了一个**交互式单负载生成脚本** `run_get_all_structural_matching.sh`，可安全地逐个数据集执行精确匹配：

```bash
cd pythonProject/scripts
chmod +x *.sh
# 获取数据集中所有查询的结构匹配（不进行 Oracle 验证，即不验证机器学习谓词）。一次只能选择并运行一个工作负载。
./run_get_all_structural_matching.sh

# EXACT: 获取真值 (GroundTruth)
./run_all_gt.sh

# 基线方法 ENUM: 详细信息请参考论文中的 Algorithm 1
./run_enum.sh
```
---

### 1.1. 主控脚本（全工作负载与全聚合模式）
如需在所有三个工作负载（`Parler`、`Parler-E`、`Amazon`）上自动运行完整流水线（包括 C++ 权重物化和分层采样），并生成绘图所需的所有数据：
<!-- 
```bash
# 确保已激活 conda 环境
conda activate iogs

# 赋予执行权限并运行总脚本
chmod +x scripts/run_all_experiments.sh
bash scripts/run_all_experiments.sh
``` -->

---

### 1.2. 针对特定工作负载的一键脚本
如果您希望评测或调试特定数据集，而不想运行耗时数小时的完整基准套件，可以使用我们提供的专用一键自动化脚本。

#### 示例：Parler（`COUNT` 模式）
脚本 `run_parler_count.sh` 自动化执行 **Parler** 工作负载在 `COUNT` 模式下的端到端工作流：
1. **步骤 1（离线投影与物化）：** 调用 C++ 引擎执行均匀树采样，估计投影扩展权重 $\hat{w}(\psi)$，并物化紧凑的核心实例空间。
2. **步骤 2（在线代理采样）：** 在预算梯度 $\alpha \in [1\%, 90\%]$ 上执行代理引导的分层重要性采样，每个预算点独立重复运行 5 次。

```bash
# 1. 激活 conda 环境
conda activate iogs
cd pythonProject/scripts
# 2. 赋予执行权限
chmod +x *.sh

# 3. 执行一键脚本

# 针对各研究问题 (RQ1 – RQ4) 的专项复现脚本

# 生成 RQ1 的 `PROXY` 结果。可在 `pythonProject/src/RQs_plots/RQ1&RQ2.ipynb` 或 `pythonProject/src/RQs_plots/Rebuttal_exp_result.ipynb` 中进行可视化展示。
./run_RQ1.sh 

# 生成 RQ2 的 `PROXY` 结果。可在 `pythonProject/src/RQs_plots/RQ1&RQ2.ipynb` 中进行可视化展示。
./run_RQ2.sh 

# 生成 RQ3 的所有结果。可在 `pythonProject/src/RQs_plots/RQ3:Sensitivity.ipynb` 中进行可视化展示。
./run_RQ3.sh

# 生成 RQ4 的所有结果。可在 `pythonProject/src/RQs_plots/RQ4:Ablation.ipynb` 中进行可视化展示。
./run_RQ4.sh


# 基线方法：Fastest-Oracle，生成 RQ1 & RQ2 的 `Fastest-Oracle` 结果。可在 `pythonProject/src/RQs_plots/RQ1&RQ2.ipynb` 中进行可视化展示。
./run_Fastest-Oracle.sh

# 基线方法：权重估计误差 (WEE) 理论下界
./run_WEE.sh

# Rebuttal 中补充的基线方法：投影空间 + ABAE。可在 `pythonProject/src/RQs_plots/Rebuttal_exp_result.ipynb` 中进行可视化展示。
# （在 Rebuttal 阶段之后，该方法得到了系统性改进，ARE 得到了一定程度的提升，但仍落后于 PROXY）
./run_Pro_Abae.sh

# Rebuttal 中补充的基线方法：投影空间 + 级联过滤器 (SUPG.ScaleDoc 风格)。可在 `pythonProject/src/RQs_plots/Rebuttal_exp_result.ipynb` 中进行可视化展示。
./run_Pro_Cacade_Filter.sh 


./run_all_avg.sh

# 4. 执行单工作负载测试

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
  以 datasets/parler/results 为例:

  1. /efficiency/allocation_strategy_comparison_count.csv: Proxy 结果文件，存储了每个查询在各采样率下的估计值与 Oracle 调用次数，用于 RQ1 & RQ2

  2. FastestO_budget_curve_count.csv: RQ1 & RQ2 的核心基线 FASTEST-ORACLE 结果文件

  3. allocation_strategy_comparison_ablation_count.csv: 消融实验结果文件，存储了各项消融实验下查询在各采样率的估计值，用于 RQ4
  ```

---

---
## 以下为正在整理与补充的实验细节。上述所有内容目前均已就绪，可完全复现论文中的全部实验以及 Rebuttal 阶段的补充实验。
