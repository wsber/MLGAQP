# 基于代理引导采样的机器学习谓词近似图聚合

# 实验评估与复现指南 (V-0.99 持续更新中...)

本仓库提供了复现论文中所有实验结果所需的完整代码与配置。用户可以选择**通过主脚本实现全自动一键复现**，也可以**按模块逐步执行实验流水线**。

> **开销说明：** 直接调用大语言模型（LLM）或深度神经网络（Oracle 模型）进行实时推理会产生极高的计算与时间开销。为了便于快速复现与验证，**我们已将所有查询对应的 Oracle 与 Proxy 验证结果预先缓存至各数据集的 `csv_data/` 目录中，支持即开即用。**

---

> ### (1) 快速验证：1 分钟图表复现
> **无需等待耗时的分层重要性采样与图匹配！** 我们已将论文中所有实验的真实评估结果完整持久化在每个数据集的 `datasets/{workload}/results/efficiency/` 目录下。
> 如果您希望**立即验证并复现论文中的所有实验图表（RQ1–RQ4）及统计显著性指标**：
> 1. 启动 Jupyter Notebook：
>    ```bash
>    jupyter notebook pythonProject/src/RQS/RQX.ipynb
>    ```
> 2. 点击 **`Run`** 即可直接加载预缓存数据，瞬间渲染出论文中呈现的所有高清矢量图表！

---

> ### (2) 技术报告与补充理论证明
> 关于论文中提及的**案例分析（Case Study）**以及**均匀树采样（Uniform Tree Sampling）的完整理论证明**，请参阅代码库根目录下的技术报告：[**`TR.pdf`**](./TR0.1.pdf)。

> ### (3) 代码库架构设计与执行建议
> 1. **可复现性与调试设计**：本代码库目前处于学术开源阶段。为了便于在各个阶段验证状态、调试算法逻辑并支持完整的数据复现流程，代码中保留了**详细的中间结果落盘持久化与 I/O 校验逻辑**。
> 2. **推荐执行策略**：
>    * **第 1 步（结构投影与权重物化）**：运行 `Projection_Sampling_and_Weight_Estimation_Runner.py`。对于每个工作负载（Workload）和聚合类型（`COUNT` / `SUM`），该步骤**仅需执行一次**。此步骤将确定拓扑投影空间 $\hat{\Psi}$，并离线物化所有投影的结构扩展权重 $\hat{w}(\psi)$。
>    * **第 2 步（采样算法与基线评估）**：在物化好 $\hat{\Psi}$ 之后，您可以多次高效运行 `Proxy_Guided_Stratified_Importance_Sampling_Runner.py` 及各类基线脚本，以快速评估不同的采样策略、消融变体以及多次运行的随机方差。
> 3. **后续重构计划**：我们的团队将持续对代码库进行模块化与解耦优化，精简 I/O 流程并提升执行效率，敬请期待后续更新。

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
底层图匹配与候选空间树采样引擎采用 C++20 开发，依赖 CMake、Boost 和 GSL：
```bash
# 1. 安装系统级依赖库 (Ubuntu/Debian)
conda install -c conda-forge cmake gxx_linux-64 boost gsl -y

# 2. 编译生成 Fastest 二进制可执行文件（项目中已包含预编译二进制文件；以下步骤展示如何重新编译）
cd cProject
mkdir -p build && cd build
cmake ..
make -j$(nproc)
cd ../..
```
*编译成功后，二进制可执行文件将生成在 `cProject/build/Fastest`。*

---

## 0. 实验基础：数据集与 ML 谓词架构

在开始运行实验前，请先熟悉底层的图数据集、合成查询负载以及机器学习谓词架构。

---

### 0.1. 数据集与查询负载

实验在三个真实世界的属性图 / 多模态图数据集上进行：

1. **`Parler`：** 文本属性社交网络数据集，包含 **3 种不同的顶点标签/类型**：用户（`user`）、帖子（`post`）和评论（`comment`）。
2. **`Parler-E`：** 改编自 `Parler`。通过将顶点类型细分为不相交的子类型来提升标签稀疏度，将标签集扩展至 **6 种不同的顶点标签**。这极大地降低了查询选择率，用于评估极端低选择率场景下的算法性能。
3. **`Amazon`：** 多模态异构图，包含用户（`user`）、文本评论（`review`）和商品图像（`product`）。通过将顶点类型随机划分为不相交的子类型，包含 **11 种不同的顶点标签**，用于评估结构复杂性与多模态特征融合能力。

#### 查询生成与聚合约束
查询图 $Q$ 是通过在数据图上进行**随机游走（Random Walks）**生成的，并附加了聚合属性和 ML 谓词：
* **聚合约束：** 为确保在 $Q$ 上进行有效的 `SUM` 和 `AVG` 聚合，每个查询必须包含至少一个数值属性：
  * **`Parler` / `Parler-E`：** `post` 顶点的 `upvotes`（点赞数）属性。
  * **`Amazon`：** `product` 顶点的 `price`（价格）或 `rating`（评分）属性。
* **查询规模与谓词配置：**
  * **`Parler`：** 包含 **245** 个单谓词查询（$|V(Q)| \in [4, 8]$，$k=1$），谓词随机分配给 1 个 `post` 或 `comment` 顶点。
  * **`Parler-E`：** 包含 **115** 个多谓词复合查询（$|V(Q)| \in [4, 8]$，$k \ge 2$），谓词同时分配给至少 1 个 `post` 顶点和至少 1 个 `comment` 顶点。
  * **`Amazon`：** 包含 **750** 个多模态复合多谓词查询（$|V(Q)| \in [3, 8]$，$k \ge 2$），谓词分配给至少 1 个 `product` 图像顶点和至少 1 个 `review` 文本顶点。

---

### 0.2. ML 谓词：Oracle 与 Proxy 模型

每个原子 ML 谓词 $\mathcal{P}_i$ 都配备了一个**高精度 Oracle 模型**（用于精确、无偏验证）和一个**轻量级 Proxy 模型**（用于高效近似评分并指导分层重要性采样）：

| 数据集 | 顶点类型 | 谓词语义 ($\mathcal{P}$) | Oracle 模型 (参数量) | Proxy 模型 (参数量) | Proxy $F_1$ 分数 | Proxy 加速比 |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| **`Amazon`** | `product` | **图像纹理分类**<br>*(木质/塑料/金属/织物/玻璃?)* | `siglip-so400m-patch14-384`<br>*(8.78亿)* | `siglip-base`<br>*(8400万)* | 0.7546 | **$26.8\times$** |
| | `review` | **情感分析**<br>*(正面/负面?)* | `roberta-large-sst2`<br>*(3.55亿)* | `bert-mini-finetuned-sst2`<br>*(1100万)* | 0.8890 | **$22.1\times$** |
| **`Parler`** /<br>**`Parler-E`** | `post` | **观点推断**<br>*(支持/反对唐纳德·特朗普?)* | `deberta-v2-xxlarge-mnli`<br>*(15亿)* | `deberta-v3-base-mnli`<br>*(1.84亿)* | 0.7720 | **$42.5\times$** |
| | `comment` | **情感分析**<br>*(正面/负面?)* | `roberta-large-sst2`<br>*(3.55亿)* | `bert-mini-finetuned-sst2`<br>*(1100万)* | 0.7876 | **$22.1\times$** |

* **Proxy 质量梯度 ($M_{P1} \sim M_{P4}$)：** 为了评估 $\text{PROXY}$ 对代理模型精度的敏感性（RQ3），我们通过微调或采用更简单的架构，为每个任务构建了 4 个代理质量梯度。其相对 $F_1$ 分数在 $[0.65, 0.89]$ 范围内单调递减，而推理速度则单调递增。

---

### 0.3. 硬件配置

所有实验均在配置如下的高性能服务器上进行：
* **操作系统：** Ubuntu 22.04 LTS
* **处理器 (CPU)：** 双路 Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz
* **内存 (RAM)：** 503 GB
* **显卡 (GPU)：** $4 \times$ NVIDIA GeForce RTX 3090 GPU（每张显存 24GB）

> **⚠️ 注意：** 每个工作负载都包含数百个复杂的子图同构匹配和 ML 谓词评估。运行完整的精确匹配需耗时数小时，并且每个工作负载至少需要 100 GB 的可用磁盘空间。**我们为所有工作负载提供了预先计算好的真实值（Ground Truth, GT）文件**，允许您跳过昂贵的全图匹配和穷举式 Oracle 评估。

---

### 0.4. 代码仓库目录结构

本项目采用分层架构：底层为高性能 **C++ 采样引擎 (`cProject`)**，上层为 **Python 代理引导采样框架 (`pythonProject`)**：

```text
PROXY/
├── cProject/                                   # [已更新] [C++ 核心引擎] 候选空间构建、树采样及语义投影权重估计
│   ├── build/                                  # 预编译二进制目录（包含编译好的 'Fastest' 可执行文件）
│   ├── driver/                                 # C++ 入口文件 (subgraph-cardinality-estimation.cc)
│   ├── lib/                                    # 图数据结构、候选空间构建器、均匀树采样器等
│   └── CMakeLists.txt                          # CMake 构建配置文件
│
├── datasets/                                   # [已更新] [数据与结果存储] 三个工作负载的数据图、查询图及实验结果
│   ├── parler/                                 # Parler 单谓词工作负载 (data_graph / query_graph / ground_truth / results)
│   ├── parler-e/                               # Parler-E 多谓词扩展工作负载
│   └── amazon/                                 # Amazon 多模态异构图工作负载
│
├── Model/                                      # [更新中] [ML 模型库] Oracle 与 Proxy 模型的权重及配置文件
│
├── pythonProject/                              # [Python 实验框架] PROXY 采样算法、基线评估及绘图
│   └── src/
│       ├── algorithms/                         # PROXY 核心算法实现
│       │   ├── exact_subgraph_match.py         # 精确子图同构匹配脚本
│       │   ├── compute_truth.py                # 真实值（GT）计算与谓词验证类
│       │   └── proxy_sample.py                 # 代理引导分层采样 (POSSA) 及其消融变体
│       │
│       ├── baseline/                           # 基线算法库
│       │   ├── ...                             # 核心基线 (ENUM, FASTEST-ORACLE, WEE 等)
│       │   └── ...                             # 对比方法 (如 PRO-ABAE, PSF 级联过滤等)
│       │
│       ├── runner/                             # 运行执行器
│       │   └── ...                             # 负责连接 C++ 与 Python 模块的执行脚本
│       │
│       └── RQS-plot/                           # 论文实验图表可视化与绘图脚本 (RQ1 ~ RQ4)
│
├── scripts/                                    # [自动化脚本] 一键式基准测试与复现 Shell 脚本
│   ├── run_RQ1.sh                              # [已更新] RQ1: PROXY 端到端效率与运行时分解基准测试
│   ├── run_RQ2.sh                              # [已更新] RQ2: PROXY 在各预算步长下的精度收敛性 (COUNT, SUM, AVG)
│   ├── run_RQ3.sh                              # [已更新] RQ3: 代理质量敏感性与鲁棒性消融实验
│   ├── run_RQ4.sh                              # [已更新] RQ4: 分配策略消融实验 (UN, PO, WO, MAB, POSS)
│   ├── run_fastesto_all.sh                     # [已更新] RQ1 & RQ2 基线: 并行 FaSTest-Oracle 预算曲线运行器
│   ├── run_parler_{count,sum}.sh               # [已更新] Parler（单谓词）模块化一键运行脚本
│   ├── run_parler_e_{count,sum}.sh             # [已更新] Parler-E（多谓词）模块化一键运行脚本
│   └── run_amazon_{count,sum}.sh               # [已更新] Amazon（多模态图）模块化一键运行脚本
│   ├── run_all_avg.sh                          # [已更新] 基于 COUNT 和 SUM 结果的离线比率合成 AVG 脚本
└── ...                                         # [持续更新中]
```

---

## 1. 一键复现

您既可以通过运行跨所有数据集的主脚本来复现实验结果，也可以通过执行特定工作负载的独立一键脚本来完成。

> ⚠️ **复现注意事项：**
>
> 仓库默认**已经打包了所有预先计算好的精确基数与真实值映射（`results/T_true_*.json`）**，您可以立即运行下游的所有采样与近似基准测试（RQ1–RQ4）。
>
> **但是，如果您希望从头生成 Ground Truth 或复现精确枚举基线（ENUM / EXACT）：**
> 获取全部精确子图匹配属于 #P-hard 问题，伴随巨大的内存和 CPU 开销（每个工作负载可能需要数小时）。我们提供了一个**交互式单工作负载生成器** `run_get_all_structural_matching.sh`，用于一次安全执行单个数据集的精确匹配：

```bash
cd pythonProject/scripts
chmod +x *.sh
# 获取数据集中所有查询的结构匹配（不包含 Oracle 验证，即不进行 ML 谓词校验）。一次只能选择并运行一个工作负载。
./run_get_all_structural_matching.sh

# EXACT：获取 Ground Truth
./run_all_gt.sh

# Baseline ENUM：详细信息请参阅论文中的 Algorithm 1
./run_enum.sh
```
---

### 1.1. 主脚本（全工作负载与聚合类型）
若要跨全部三个工作负载（`Parler`、`Parler-E`、`Amazon`）自动执行完整流程（包括 C++ 权重物化与分层采样），并生成绘图所需的全部数据：
<!-- 
```bash
# 确保激活 conda 环境
conda activate iogs

# 赋予执行权限并运行主脚本
chmod +x scripts/run_all_experiments.sh
bash scripts/run_all_experiments.sh
``` -->

---

### 1.2. 各工作负载独立一键脚本
如果您希望单独评估或调试特定数据集，而无需运行耗时数小时的完整基准测试套件，可以使用以下专用的一键自动化脚本。

#### 示例：Parler (`COUNT` 模式)
脚本 `run_parler_count.sh` 自动化执行 **Parler** 工作负载在 `COUNT` 模式下的端到端流程：
1. **第 1 步（离线投影与物化）：** 调用 C++ 引擎执行均匀树采样，估计投影扩展权重 $\hat{w}(\psi)$，并物化紧凑的核心实例空间。
2. **第 2 步（在线代理采样）：** 在预算梯度 $\alpha \in [1\%, 90\%]$ 范围内执行代理引导的分层重要性采样，每个步长独立运行 5 次。

```bash
# 1. 激活 conda 环境
conda activate iogs
cd pythonProject/scripts
# 2. 赋予执行权限
chmod +x *.sh

# 3. 执行一键脚本

# 特定 RQ 复现脚本 (RQ1 – RQ4)

# 生成 RQ1 的 `PROXY` 实验结果。可视化文件位于 `pythonProject/src/RQs_plots/RQ1&RQ2.ipynb` 或 `pythonProject/src/RQs_plots/Rebuttal_exp_result.ipynb`。
./run_RQ1.sh 

# 生成 RQ2 的 `PROXY` 实验结果。可视化文件位于 `pythonProject/src/RQs_plots/RQ1&RQ2.ipynb`。
./run_RQ2.sh 

# 生成 RQ3 的所有实验结果。可视化文件位于 `pythonProject/src/RQs_plots/RQ3:Sensitivity.ipynb`。
./run_RQ3.sh

# 生成 RQ4 的所有实验结果。可视化文件位于 `pythonProject/src/RQs_plots/RQ4:Ablation.ipynb`。
./run_RQ4.sh


# 基线方法：Fastest-Oracle，生成 RQ1 & RQ2 的 `Fastest-Oracle` 结果。可视化文件位于 `pythonProject/src/RQs_plots/RQ1&RQ2.ipynb` 
./run_Fastest-Oracle.sh

# 基线方法：权重估计误差（WEE）下界
./run_WEE.sh

# Rebuttal 新增基线：投影空间 + ABAE。可视化文件位于 `pythonProject/src/RQs_plots/Rebuttal_exp_result.ipynb` 
./run_Pro_Abae.sh

# Rebuttal 新增基线：投影空间 + 级联过滤 (SUPG/ScaleDoc 风格)。可视化文件位于 `pythonProject/src/RQs_plots/Rebuttal_exp_result.ipynb` 
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
  datasets/parler/results/efficiency/allocation_strategy_comparison_count.csv
  ```

---

---
## 下面是实验细节正在整理补充中, 以上内容已全部就绪,可以完整复现论文实验和rebuttal中补充的实验。
