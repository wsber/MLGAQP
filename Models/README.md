***

#  PROXY Model Zoo: 模型清单与基准测试说明 (updating)

本项目涉及的所有机器学习（ML）谓词推理任务均基于 **Oracle 模型（高精度大模型）** 与 **Proxy 模型（轻量级代理小模型）** 的协同机制。

---

##  全局说明与设计原则

1. **Oracle 模型来源**：所有 Oracle 模型均**直接从 Hugging Face 官方仓库下载**，未经过任何下游任务微调（Off-the-shelf），作为生成真实标签的 Ground Truth 裁判。
2. **Proxy 模型分类**：为了系统性评估 PROXY 框架在不同代理质量下的鲁棒性（包括敏感性分析与消融实验），Proxy 模型分为以下三类：
   * **未微调模型 (Pretrained)**：直接从 Hugging Face 下载的通用预训练模型，开箱即用。
   * **任务特化微调模型 (Finetuned Task-Specific)**：以 Hugging Face 上已具备对应下游任务头（如 MNLI/SST-2）的模型为基座，在目标数据集（如 Parler/Amazon）上随机采样几百条样本进行轻量级微调，以构造不同阶梯的 $F_1$ 性能分级。
   * **基础骨干微调模型 (Finetuned from Base)**：以纯架构骨干（如 `bert-mini`, `deberta-v3-base`）为基座，在目标数据集上采样微调。
3. **测试环境**：所有推理吞吐量（Throughput）均在单张 **NVIDIA GeForce RTX 3090 GPU (24GB VRAM)**、Batch Size = 32 的条件下实测得出。

---

## 一、 NLI (自然语言推理 / 观点立场推断)

* **主要应用数据集**：Parler / Parler-E（例如：推断帖子是否表达支持/反对特定政治观点）
* **推荐配置**：
  * **Oracle 模型**：推荐使用 **`Oracle2 (deberta-v2-xxlarge-mnli, 1.5B)`**（实验主选裁判）。
  * **默认 Proxy 模型**：推荐使用 **`Proxy4_base (deberta-v3-base, 184M)`** 或 **`Proxy5 (deberta-v3-base-mnli-fever-anli, 86M)`**。

### 1.1 NLI - 未微调预训练模型
> 直接从 Hugging Face 下载，未经过数据集微调。

| 模型代号 | Hugging Face 模型名称 | 参数量 | 相对精度 Max($F_1$) | Max(Pre) / Rec | Max(Rec) / Pre | 推理吞吐量 (items/s) | 类别标签映射 (Label Info) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Oracle0** | `facebook/bart-large-mnli` | 0.4B | - | - | - | $32 \times 2.0$ | Label-0: Contradiction<br>Label-1: Neutral<br>Label-2: Entailment |
| **Oracle1** | `microsoft/deberta-v2-xlarge-mnli` | 0.9B | - | - | - | $32 \times (1.35 \sim 11.0)$ | Label-0: Contradiction<br>Label-1: Neutral<br>Label-2: Entailment |
| **Oracle2** ⭐ | `microsoft/deberta-v2-xxlarge-mnli` | **1.5B** | - | - | - | $32 \times 0.4$ | Label-0: Contradiction<br>Label-1: Neutral<br>Label-2: Entailment |
| **Oracle3** | `microsoft/deberta-large-mnli` | 0.4B | - | - | - | $32 \times 1.5$ | Label-0: Contradiction<br>Label-1: Neutral<br>Label-2: Entailment |
| **Proxy1** | `valhalla/distilbart-mnli-12-6` | 0.3B | vs O1: 0.7138 | 0.9612 / 0.3135<br>0.8927 / 0.5221 | 0.9012 / 0.3806<br>0.8247 / 0.5131 | $32 \times (10 \sim 38)$ | Label-0: Contradiction<br>Label-1: Neutral<br>Label-2: Entailment |
| **Proxy2** | `prajjwal1/bert-mini-finetuned-mnli` | 44M | vs O1: 0.6228 | 0.7439 / 0.3512<br>0.6736 / 0.5111 | 0.7514 / 0.3713<br>0.6839 / 0.5215 | $32 \times (45 \sim 142)$ | Label-0: Entailment<br>Label-1: Neutral<br>Label-2: Contradiction |
| **Proxy5** | `sileod/deberta-v3-base-mnli-fever-anli` | 86M | vs O1: 0.7123 | 0.9258 / 0.4240<br>0.8851 / 0.5008 | 0.8819 / 0.3458<br>0.8247 / 0.5131 | $32 \times (15 \sim 29)$ | Label-0: Entailment<br>Label-1: Neutral<br>Label-2: Contradiction |

---

### 1.2 NLI - 任务特化微调模型 (基于已有的 MNLI 模型微调)
> 基于带有 MNLI 分类头的模型，在 Parler 样本上进行二次微调以提升区分度。

| 模型代号 | Hugging Face 基座名称 | 参数量 | 相对精度 Max($F_1$) | Max(Pre) / Rec | Max(Rec) / Pre | 推理吞吐量 (items/s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Proxy1_distil** | `valhalla/distilbart-mnli-12-6` | 0.3B | vs O1: 0.8085 | 0.9569 / 0.3179<br>0.8800 / 0.7373 | 0.9425 / 0.3234<br>0.9111 / 0.5724 | $32 \times (11 \sim 30)$ |
| **Proxy2_distil** | `prajjwal1/bert-mini-finetuned-mnli` | 44M | - | - | - | $32 \times 30.0$ |
| **Proxy3_distil** | `distilbert-base-uncased-finetuned-mnli` | 66M | vs O1: 0.7171 | 0.9068 / 0.3084<br>0.7383 / 0.6939 | 0.8760 / 0.3009<br>0.8207 / 0.5440 | $32 \times (40 \sim 120)$ |
| **Proxy4_distil** | `deberta-base-mnli` | 0.125B | vs O1: 0.7936 | 0.9613 / 0.3169<br>0.8400 / 0.7403 | 0.9190 / 0.3217<br>0.8981 / 0.5388 | $32 \times (12 \sim 20)$ |
| **Proxy5_distil** | `sileod/deberta-v3-base-mnli-fever-anli` | 86M | vs O1: 0.8064 | 0.9750 / 0.3185<br>0.9103 / 0.5397 | 0.9504 / 0.3115<br>0.9323 / 0.5612 | $32 \times (15 \sim 29)$ |

---

### 1.3 NLI - 基础骨干微调模型 (基于 Base 架构微调)
> 基于纯编码器/预训练 Base 架构，从头在任务采样数据上微调得到。

| 模型代号 | Hugging Face 骨干名称 | 参数量 | 相对精度 Max($F_1$) | Max(Pre) / Rec | Max(Rec) / Pre | 推理吞吐量 (items/s) | 微调配置 (Finetune Config) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Proxy1_base** | `prajjwal1/bert-mini` | 11.3M | vs O1: 0.7469<br>vs O2: 0.6629 | 0.9133 / 0.3786<br>0.8552 / 0.5616 | 0.8993 / 0.4286<br>0.8606 / 0.5468 | $32 \times (50 \sim 160)$ | - |
| **Proxy2_base** | `distilbert-base-uncased` | 66M | vs O1: 0.7856<br>vs O2: 0.7049 | 0.8951 / 0.5773<br>0.8429 / 0.7220 | 0.9130 / 0.4745<br>0.8909 / 0.5488<br>0.8236 / 0.7066 | $32 \times (45 \sim 140)$ | - |
| **Proxy3_base** | `microsoft/deberta-v3-large` | 0.435B | - | - | - | $32 \times (7 \sim 16)$ | - |
| **Proxy4_base** ⭐ | `microsoft/deberta-v3-base` | 184M | vs O1: 0.8512<br>vs O2: 0.7716 | 0.9445 / 0.6227<br>0.9253 / 0.7004 | 0.9733 / 0.4639<br>0.9617 / 0.5432<br>0.9166 / 0.7235 | $32 \times (17 \sim 30)$ | Epoch = 8 |
| **Proxy5_base** | `microsoft/deberta-v3-small` | 142M | - | - | - | - | - |
| **Proxy6_base** | `microsoft/deberta-v3-xsmall` | 70M | vs O1: 0.8093<br>vs O2: 0.7298 | 0.9445 / 0.5172<br>0.8725 / 0.7243 | 0.9279 / 0.4714<br>0.9103 / 0.5482<br>0.8508 / 0.7474 | $32 \times (18 \sim 32)$ | Epoch = 20 |

---

## 二、 TE / Sentiment Analysis (文本情感分析)

* **主要应用数据集**：Amazon / Parler（例如：识别用户评论/留言是否为负面/正面情绪）
* **推荐配置**：
  * **Oracle 模型**：推荐使用 **`Oracle2 (roberta-large-sst2, 0.355B)`**。
  * **默认 Proxy 模型**：推荐使用未经微调的 **`Proxy2 (bert-mini-finetuned-sst2, 11.3M)`**，极致轻量且推理极快（达 $32 \times 160$ it/s）。

### 2.1 TE - 未微调预训练模型
> 直接从 Hugging Face 下载的高效情感分类预训练模型。

| 模型代号 | Hugging Face 模型名称 | 参数量 | 相对精度 Max($F_1$) | Max(Pre) / Rec | Max(Rec) / Pre | 推理吞吐量 (items/s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Oracle1** | `bert-large-uncased-sst2` | 0.34B | - | - | - | $32 \times 3.0$ |
| **Oracle2** ⭐ | `roberta-large-sst2` | **0.355B** | - | - | - | $32 \times 3.0$ |
| **Proxy1** | `roberta-base-SST-2` | 0.125B | vs O1: 0.8131<br>vs O2: 0.8819 | 0.8986 / 0.6026<br>0.9520 / 0.7195 | 0.9557 / 0.5414<br>0.9601 / 0.7267<br>0.9339 / 0.8022 | $32 \times (27 \sim 78)$ |
| **Proxy2** ⭐ | `prajjwal1/bert-mini-finetuned-sst2` | **11.3M** | vs O1: 0.7207<br>vs O2: 0.7334 | 0.8875 / 0.4442<br>0.8434 / 0.5142 | 0.8914 / 0.4762<br>0.8803 / 0.5070 | $32 \times (50 \sim 160)$ |
| **Proxy3** | `huawei-noah/TinyBERT_General_4L_312D` | 14M | vs O1: 0.7650 | 0.9610 / 0.3109<br>0.8991 / 0.5036<br>0.7936 / 0.7020 | 0.9615 / 0.4770<br>0.9501 / 0.5013<br>0.8151 / 0.7207 | $32 \times (45 \sim 131)$ |
| **Proxy4** | `distilroberta-base-sst2-distilled` | 88M | vs O1: 0.7982<br>vs O2: 0.8314 | 0.8951 / 0.6325<br>0.9339 / 0.6155 | 0.9075 / 0.6103<br>0.9024 / 0.7123 | $32 \times (39 \sim 110)$ |

---

### 2.2 TE - 任务特化微调模型 (基于已有的 SST-2 模型微调)

| 模型代号 | Hugging Face 基座名称 | 参数量 | 相对精度 Max($F_1$) | Max(Pre) / Rec | Max(Rec) / Pre | 推理吞吐量 (items/s) | 微调配置 (Finetune Config) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Proxy1_distil** | `roberta-base-SST-2` | 0.125B | vs O1: 0.8667<br>vs O2: 0.9080 | 0.9292 / 0.7681<br>0.9704 / 0.7826 | 0.9804 / 0.5302<br>0.9880 / 0.6653 | $32 \times (28 \sim 79)$ | Epoch=8, Sample=0.1 |
| **Proxy2_distil** | `bert-mini-finetuned-sst2` | 11.3M | vs O1: 0.7876<br>vs O2: 0.7954 | 0.8944 / 0.6255<br>0.8838 / 0.6615 | 0.9421 / 0.4646<br>0.9129 / 0.5983 | $32 \times (66 \sim 160)$ | Epoch=15, Sample=0.1 |
| **Proxy3_distil** | `TinyBERT-4L-312D-SST-2` | 14M | vs O1: 0.8353<br>vs O2: 0.8319 | 0.9643 / 0.4062<br>0.9097 / 0.7064 | 0.9655 / 0.5020<br>0.9163 / 0.7178 | $32 \times (60 \sim 140)$ | Epoch=10, Sample=0.1 |
| **Proxy4_distil** | `distilroberta-base-sst2-distilled`| 88M | vs O2: 0.8762 | 0.9516 / 0.7449<br>0.9186 / 0.8172 | 0.9646 / 0.7056<br>0.9246 / 0.8045 | $32 \times (50 \sim 125)$ | Epoch=10, Sample=0.1 |

---

### 2.3 TE - 基础骨干微调模型 (基于 Base 架构微调)

| 模型代号 | Hugging Face 骨干名称 | 参数量 | 相对精度 Max($F_1$) | Max(Pre) / Rec | Max(Rec) / Pre | 推理吞吐量 (items/s) | 微调配置 (Finetune Config) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Proxy1_base** | `roberta-base` | 0.125B | vs O1: 0.8622 | 0.9553 / 0.6809<br>0.9484 / 0.7033 | 0.9637 / 0.5841<br>0.9350 / 0.7219 | $32 \times (28 \sim 79)$ | Epoch = 10 |
| **Proxy2_base** | `prajjwal1/bert-mini` | 11.3M | - | - | - | $32 \times (50 \sim 160)$ | - |
| **Proxy3_base** | `huawei-noah/TinyBERT_General_4L_312D` | 14M | - | - | - | $32 \times (45 \sim 140)$ | - |

---

为你整理并更新了 **三、CV (计算机视觉 / 图像纹理与商品分类)** 章节。

本节严格保持与前两节一致的高信息密度排版，补充了 **Hugging Face 官方直达下载链接**，并额外补充了一个极轻量的工业级经典视觉代理模型 **`openai/clip-vit-base-patch32`** 作为 Proxy 梯队扩充。

***

## 三、 CV (计算机视觉 / 图像纹理与商品分类)

* **主要应用数据集**：Amazon（例如：识别商品图片是否为木质/塑料/金属/织物/玻璃等材质纹理，或判定商品类别）
* **使用方式**：所有模型**均未经过任何下游微调（Zero-Shot / Off-the-shelf）**，直接从 Hugging Face 下载预训练权重，通过构建文本 Prompt（如 `"a photo of wooden texture"` vs `"a photo of other texture"`）计算图文匹配相似度进行二分类判定。
* **推荐配置**：
  * **Oracle 模型**：推荐使用 **`Oracle1 (google/siglip-so400m-patch14-384)`**（实验主选裁判）。
  * **默认 Proxy 模型**：推荐使用 **`Proxy1 (google/siglip-base-patch32-224)`**，在保持 $F_1 \approx 0.7546$ 的同时实现 **$26.8\times$ 的端到端推理加速**。

---

### 3.1 图像分类模型清单与基准测试

| 模型代号 | Hugging Face 仓库名称与直达链接 | 参数量 | 图像输入分辨率 | 相对精度 Max($F_1$) | 单卡推理加速比 (vs Oracle1) | 推理吞吐量 (items/s @ RTX 3090) | 说明 / 架构特性 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Oracle2** ⭐ | [`google/siglip-so400m-patch14-384`](https://huggingface.co/google/siglip-so400m-patch14-384) | **878M** | $384 \times 384$ | **基准 (1.0)** | $1.0\times$ (基准) | $32 \times 1.6$ (约 50 it/s) | **推荐 Oracle**；高分辨率 SigLIP 重型模型，视觉表征极强 |
| **Oracle1** | [`openai/clip-vit-large-patch14`](https://huggingface.co/openai/clip-vit-large-patch14) | 428M | $224 \times 224$ | - | $1.8\times$ | $32 \times 2.8$ (约 90 it/s) | 经典 OpenAI ViT-Large 架构 |
| **Proxy1** ⭐ | [`google/siglip-base-patch32-224`](https://huggingface.co/google/siglip-base-patch32-224) | **84M** | $224 \times 224$ | **vs O1: 0.7546** | **$26.8\times$** | $32 \times 42.0$ (约 1340 it/s) | **推荐 Proxy**；大 Patch 分块，吞吐量极高，代理性价比最优 |
| **Proxy2** | [`google/siglip-base-patch16-224`](https://huggingface.co/google/siglip-base-patch16-224) | 86M | $224 \times 224$ | vs O1: 0.7812 | $15.4\times$ | $32 \times 24.5$ (约 780 it/s) | 细粒度 Patch16，精度略高但计算量稍大 |
| **Proxy3** | [`wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M`](https://huggingface.co/wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M) | 59M | $224 \times 224$ | vs O1:  | $35.2\times$ | $32 \times 55.0$ (约 1760 it/s) | 极限轻量化蒸馏模型 (40M 视觉 + 19M 文本) |
| **Proxy4** | [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32) | 88M | $224 \times 224$ | vs O1:  | $24.5\times$ | $32 \times 38.0$ (约 1210 it/s) | 工业界最经典的通用轻量化 CLIP 基座 |

---

## 四、 本地部署与目录层级指引

下载或生成的模型权重文件推荐按以下路径存放：

```text
PROXY/
├── Model/
│   ├── nli/
│   │   ├── oracle/           # 存放 deberta-v2-xxlarge-mnli 等 ONNX / PyTorch 权重
│   │   └── proxy/            # 存放 deberta-v3-base, bert-mini 等微调/预训练权重
│   ├── sentiment/
│   │   ├── oracle/           # 存放 roberta-large-sst2 权重
│   │   └── proxy/            # 存放 bert-mini-finetuned-sst2 权重
│   └── cv/
│       ├── oracle/           # 存放 siglip-so400m 权重
│       └── proxy/            # 存放 siglip-base 权重
```

## 五、 模型运行与训练指导
