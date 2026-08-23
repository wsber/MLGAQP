

***

# PROXY Model Zoo: Model Inventory and Benchmarking Guide

All machine learning (ML) predicate inference tasks in this project rely on a collaborative mechanism between **Oracle models (high-accuracy large models)** and **Proxy models (lightweight proxy small models)**.

---

> ### 💡 Important Reproduction Note: Offline Caching vs. Real-Time Inference
> 
> To ensure statistical significance, all experimental benchmarks (RQ1–RQ4) evaluate each query over multiple budget ticks with **at least 5 independent repetitions (runs $\ge 5$)**. 
> 
> * **Computational & Financial Overhead**: Invoking foundation Oracle models or deep neural Proxy models in real-time during every sampling iteration will incur prohibitive monetary costs (massive token fees) and excessive runtime overhead (hundreds of GPU hours).
> * **Recommended Reproduction Practice**: 
>   1. **Offline Inference & Caching**: We strongly recommend running each Oracle and Proxy model **once offline** on the target columns of each table in `data_graph/*/csv_data/`, appending the predicted probabilities as new columns in the CSV files.
>   2. **Latency Profiling**: Profile the empirical per-item/per-batch inference latency once on your target hardware.
>   3. **Fast Online Simulation**: During online stratified importance sampling, the algorithms simply look up the pre-cached predictions from the CSV while accounting for the profiled execution time.
> * **Benefit**: This guarantees **100% mathematical and algorithmic fidelity** while allowing researchers to reproduce all paper results in minutes, even on standard consumer-grade commodity PCs.

---

## Overview and Design Principles

1. **Oracle Model Sources**: All Oracle models are **downloaded directly from official Hugging Face repositories** off-the-shelf without any downstream task-specific fine-tuning, serving as ground-truth arbiters for generating true labels.
2. **Proxy Model Categories**: To systematically evaluate the robustness of the PROXY framework across varying proxy quality tiers (including sensitivity analysis and ablation studies), Proxy models are categorized into the following three types:
   * **Pretrained Models (Off-the-shelf)**: General-purpose pretrained models downloaded directly from Hugging Face and used out-of-the-box.
   * **Task-Specific Fine-tuned Models**: Initialized from Hugging Face checkpoints that already feature corresponding downstream classification heads (e.g., MNLI/SST-2), then lightly fine-tuned on randomly sampled instances (500–1000) from target datasets (e.g., Parler/Amazon) to construct distinct $F_1$ performance tiers.
   * **Base Backbone Fine-tuned Models**: Fine-tuned on sampled target dataset instances starting from pure architectural backbones (e.g., `bert-mini`, `deberta-v3-base`).
3. **Benchmarking Environment**: All inference throughput metrics (items/s) were empirically benchmarked on a single **NVIDIA GeForce RTX 3090 GPU (24GB VRAM)** with Batch Size = 32.
4. **Scope of Paper Experiments & Extended Model Zoo**: 
   * **Paper Core Benchmark**: In the paper's reported core experimental evaluations, **only the NLI Base Fine-tuned models (specifically `Proxy4_base` in Section 1.3) and the TE Distil model (specifically `Proxy2_distil` in Section 2.2) are utilized**.
   * **Extended Suite**: All other fine-tuned variants (in Sections 1.2,1.3, 2.2, and 2.3) were **not used in the main paper's reported results**. They are released as part of this extended Model Zoo to give researchers and practitioners complete freedom to substitute, compare, and stress-test arbitrary proxy-oracle accuracy/latency trade-offs.

5. **Metric Definitions & Threshold Tuning (`Max(Prec) / Rec` & `Max(Rec) / Prec`)**:
   * **`Max(Prec) / Rec`**: By sweeping the proxy decision threshold against Oracle Ground Truth, this metric reports the **maximum Precision** achievable under the constraint of $\text{Recall} > 0.3$ (top line) or $\text{Recall} > 0.5$ (bottom line), paired with the actual **Recall** achieved at that calibrated threshold.
   * **`Max(Rec) / Prec`**: Symmetrically reports the **maximum Recall** achievable under minimum precision constraints, along with the corresponding **Precision**.

---

## 1. NLI (Natural Language Inference / Opinion & Stance Inference)

* **Primary Datasets**: Parler / Parler-E (e.g., inferring whether a post expresses support for or opposition to a specific political stance)
* **Recommended Configurations**:
  * **Oracle Model**: **`Oracle2 (deberta-v2-xxlarge-mnli, 1.5B)`** (Primary experimental judge).
  * **Default Proxy Models**: **`Proxy4_base (deberta-v3-base, 184M)`** ⭐ (Main paper proxy) or **`Proxy5 (deberta-v3-base-mnli-fever-anli, 86M)`**.

### 1.1 NLI - Pretrained Models (Off-the-Shelf)
> Downloaded directly from Hugging Face without dataset-specific fine-tuning.

| Model ID | Hugging Face Repository & Link | # Params | Relative Accuracy Max($F_1$) | Max(Prec) / Rec >0.3/0.5 | Max(Rec) / Prec >0.3/0.5 | Throughput (items/s) | Label Mapping |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Oracle0** | [`facebook/bart-large-mnli`](https://huggingface.co/facebook/bart-large-mnli) | 0.4B | - | - | - | $32 \times 2.0$ | Label-0: Contradiction<br>Label-1: Neutral<br>Label-2: Entailment |
| **Oracle1** | [`microsoft/deberta-v2-xlarge-mnli`](https://huggingface.co/microsoft/deberta-v2-xlarge-mnli) | 0.9B | - | - | - | $32 \times (1.35 \sim 11.0)$ | Label-0: Contradiction<br>Label-1: Neutral<br>Label-2: Entailment |
| **Oracle2** ⭐ | [`microsoft/deberta-v2-xxlarge-mnli`](https://huggingface.co/microsoft/deberta-v2-xxlarge-mnli) | **1.5B** | - | - | - | $32 \times 0.4$ | Label-0: Contradiction<br>Label-1: Neutral<br>Label-2: Entailment |
| **Oracle3** | [`microsoft/deberta-large-mnli`](https://huggingface.co/microsoft/deberta-large-mnli) | 0.4B | - | - | - | $32 \times 1.5$ | Label-0: Contradiction<br>Label-1: Neutral<br>Label-2: Entailment |
| **Proxy1** | [`valhalla/distilbart-mnli-12-6`](https://huggingface.co/valhalla/distilbart-mnli-12-6) | 0.3B | vs O1: 0.7138 | 0.9612 / 0.3135<br>0.8927 / 0.5221 | 0.9012 / 0.3806<br>0.8247 / 0.5131 | $32 \times (10 \sim 38)$ | Label-0: Contradiction<br>Label-1: Neutral<br>Label-2: Entailment |
| **Proxy2** | [`prajjwal1/bert-mini`](https://huggingface.co/prajjwal1/bert-mini) *(mnli)* | 44M | vs O1: 0.6228 | 0.7439 / 0.3512<br>0.6736 / 0.5111 | 0.7514 / 0.3713<br>0.6839 / 0.5215 | $32 \times (45 \sim 142)$ | Label-0: Entailment<br>Label-1: Neutral<br>Label-2: Contradiction |
| **Proxy5** | [`sileod/deberta-v3-base-mnli-fever-anli`](https://huggingface.co/sileod/deberta-v3-base-mnli-fever-anli) | 86M | vs O1: 0.7123 | 0.9258 / 0.4240<br>0.8851 / 0.5008 | 0.8819 / 0.3458<br>0.8247 / 0.5131 | $32 \times (15 \sim 29)$ | Label-0: Entailment<br>Label-1: Neutral<br>Label-2: Contradiction |

---

### 1.2 NLI - Task-Specific Fine-tuned Models (Fine-tuned from MNLI Checkpoints)

> **Note:**  
> - **HuggingFace Base Checkpoint & Link**: Checkpoints initialized with existing MNLI heads, **before** target dataset fine-tuning.  
> - **Our Fine-tuned Model & Link**: Secondary fine-tuned on Parler NLI samples to enhance discriminability.  
> - *(These models are provided for extended exploration and were not used in the paper's main reported results).*

| Model ID | HuggingFace Base Checkpoint & Link (Pretrained Only) | Our Fine-tuned Model & Link | # Params | Relative Accuracy Max($F_1$) | Max(Prec) / Rec | Max(Rec) / Prec | Throughput (items/s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Proxy1_distil** | [`valhalla/distilbart-mnli-12-6`](https://huggingface.co/valhalla/distilbart-mnli-12-6) | 🚀 Uploading soon | 0.3B | vs O1: 0.8085 | 0.9569 / 0.3179<br>0.8800 / 0.7373 | 0.9425 / 0.3234<br>0.9111 / 0.5724 | $32 \times (11 \sim 30)$ |
| **Proxy2_distil** | [`prajjwal1/bert-mini`](https://huggingface.co/prajjwal1/bert-mini) | 🚀 Uploading soon | 44M | - | - | - | $32 \times 30.0$ |
| **Proxy3_distil** | [`distilbert/distilbert-base-uncased-finetuned-mnli`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-mnli) | 🚀 Uploading soon | 66M | vs O1: 0.7171 | 0.9068 / 0.3084<br>0.7383 / 0.6939 | 0.8760 / 0.3009<br>0.8207 / 0.5440 | $32 \times (40 \sim 120)$ |
| **Proxy4_distil** | [`microsoft/deberta-base-mnli`](https://huggingface.co/microsoft/deberta-base-mnli) | 🚀 Uploading soon | 0.125B | vs O1: 0.7936 | 0.9613 / 0.3169<br>0.8400 / 0.7403 | 0.9190 / 0.3217<br>0.8981 / 0.5388 | $32 \times (12 \sim 20)$ |
| **Proxy5_distil** | [`sileod/deberta-v3-base-mnli-fever-anli`](https://huggingface.co/sileod/deberta-v3-base-mnli-fever-anli) | 🚀 Uploading soon | 86M | vs O1: 0.8064 | 0.9750 / 0.3185<br>0.9103 / 0.5397 | 0.9504 / 0.3115<br>0.9323 / 0.5612 | $32 \times (15 \sim 29)$ |

---

### 1.3 NLI - Base Backbone Fine-tuned Models (Fine-tuned from Base Architectures)

> **Note:**  
> - **HuggingFace Base Backbone & Link**: Official pretrained backbone weights from HuggingFace, **not fine-tuned** on the NLI task.  
> - **Our Fine-tuned Model & Link**: Models fine-tuned from the pure base backbone on Parler task data.  
> - **`Proxy4_base` ⭐ is the primary NLI proxy model evaluated in the paper's experimental results.**

| Model ID | HuggingFace Base Backbone & Link (Pretrained Only) | Our Fine-tuned Model & Link | # Params | Relative Accuracy Max($F_1$) | Max(Prec) / Rec | Max(Rec) / Prec | Throughput (items/s) | Fine-tuning Configuration |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Proxy1_base** | [`prajjwal1/bert-mini`](https://huggingface.co/prajjwal1/bert-mini) | [`wsber123/bert-mini`](https://huggingface.co/wsber123/bert-mini/tree/main) ✅ | 11.3M | vs O1: 0.7469<br>vs O2: 0.6629 | 0.9133 / 0.3786<br>0.8552 / 0.5616 | 0.8993 / 0.4286<br>0.8606 / 0.5468 | $32 \times (50 \sim 160)$ | - |
| **Proxy2_base** | [`distilbert/distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased) | [`wsber123/distilbert-base`](https://huggingface.co/wsber123/distilbert-base/tree/main) ✅ | 66M | vs O1: 0.7856<br>vs O2: 0.7049 | 0.8951 / 0.5773<br>0.8429 / 0.7220 | 0.9130 / 0.4745<br>0.8909 / 0.5488<br>0.8236 / 0.7066 | $32 \times (45 \sim 140)$ | - |
| **Proxy3_base** | [`microsoft/deberta-v3-large`](https://huggingface.co/microsoft/deberta-v3-large) | 🚀 Uploading soon | 0.435B | - | - | - | $32 \times (7 \sim 16)$ | - |
| **Proxy4_base** ⭐ | [`microsoft/deberta-v3-base`](https://huggingface.co/microsoft/deberta-v3-base) | [`wsber123/deberta-v3-base-binary`](https://huggingface.co/wsber123/deberta-v3-base-binary) ✅ | 184M | vs O1: 0.8512<br>vs O2: 0.7716 | 0.9445 / 0.6227<br>0.9253 / 0.7004 | 0.9733 / 0.4639<br>0.9617 / 0.5432<br>0.9166 / 0.7235 | $32 \times (17 \sim 30)$ | Epoch = 8 |
| **Proxy6_base** | [`microsoft/deberta-v3-xsmall`](https://huggingface.co/microsoft/deberta-v3-xsmall) | 🚀 Uploading soon | 70M | vs O1: 0.8093<br>vs O2: 0.7298 | 0.9445 / 0.5172<br>0.8725 / 0.7243 | 0.9279 / 0.4714<br>0.9103 / 0.5482<br>0.8508 / 0.7474 | $32 \times (18 \sim 32)$ | Epoch = 20 |

---

## 2. TE / Sentiment Analysis (Text Classification & Sentiment Analysis)

* **Primary Datasets**: Amazon / Parler (e.g., detecting whether user reviews/comments express positive or negative sentiment)
* **Recommended Configurations**:
  * **Oracle Model**: **`Oracle2 (howey/roberta-large-sst2, 0.355B)`**.
  * **Default Proxy Model**: Pretrained **`Proxy2 (bert-mini, 11.3M)`** ⭐, offering extreme lightweight efficiency and high throughput (up to $32 \times 160$ items/s).
* **Usage Declaration**: The fine-tuned models listed in Section 2.2 and Section 2.3 are provided for model zoo completeness; **the core paper experiments utilize the pretrained off-the-shelf `Proxy2` (Section 2.1)**.

### 2.1 TE - Pretrained Models (Off-the-Shelf)
> High-efficiency sentiment classification models downloaded directly from Hugging Face.

| Model ID | Hugging Face Repository & Link | # Params | Relative Accuracy Max($F_1$) | Max(Prec) / Rec | Max(Rec) / Prec | Throughput (items/s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Oracle1** | [`yoshitomo-matsuda/bert-large-uncased-sst2`](https://huggingface.co/yoshitomo-matsuda/bert-large-uncased-sst2) | 0.34B | - | - | - | $32 \times 3.0$ |
| **Oracle2** ⭐ | [`howey/roberta-large-sst2`](https://huggingface.co/howey/roberta-large-sst2) | **0.355B** | - | - | - | $32 \times 3.0$ |
| **Proxy1** | [`textattack/roberta-base-SST-2`](https://huggingface.co/textattack/roberta-base-SST-2) | 0.125B | vs O1: 0.8131<br>vs O2: 0.8819 | 0.8986 / 0.6026<br>0.9520 / 0.7195 | 0.9557 / 0.5414<br>0.9601 / 0.7267<br>0.9339 / 0.8022 | $32 \times (27 \sim 78)$ |
| **Proxy2** ⭐ | [`prajjwal1/bert-mini`](https://huggingface.co/prajjwal1/bert-mini) *(sst2)* | **11.3M** | vs O1: 0.7207<br>vs O2: 0.7334 | 0.8875 / 0.4442<br>0.8434 / 0.5142 | 0.8914 / 0.4762<br>0.8803 / 0.5070 | $32 \times (50 \sim 160)$ |
| **Proxy3** | [`huawei-noah/TinyBERT_General_4L_312D`](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D) | 14M | vs O1: 0.7650 | 0.9610 / 0.3109<br>0.8991 / 0.5036<br>0.7936 / 0.7020 | 0.9615 / 0.4770<br>0.9501 / 0.5013<br>0.8151 / 0.7207 | $32 \times (45 \sim 131)$ |
| **Proxy4** | [`azizbarank/distilroberta-base-sst2-distilled`](https://huggingface.co/azizbarank/distilroberta-base-sst2-distilled) | 88M | vs O1: 0.7982<br>vs O2: 0.8314 | 0.8951 / 0.6325<br>0.9339 / 0.6155 | 0.9075 / 0.6103<br>0.9024 / 0.7123 | $32 \times (39 \sim 110)$ |

---

### 2.2 TE - Task-Specific Fine-tuned Models (Fine-tuned from SST-2 Checkpoints)

> **Note:**  
> - **HuggingFace Base Checkpoint & Link**: Initialized with existing SST-2 classification heads before dataset fine-tuning.  
> - **Our Fine-tuned Model & Link**: Secondary fine-tuned on task data. *(Released for custom benchmarking; not used in paper core results).*

| Model ID | HuggingFace Base Checkpoint & Link (Pretrained Only) | Our Fine-tuned Model & Link | # Params | Relative Accuracy Max($F_1$) | Max(Prec) / Rec | Max(Rec) / Prec | Throughput (items/s) | Fine-tuning Configuration |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Proxy1_distil** | [`textattack/roberta-base-SST-2`](https://huggingface.co/textattack/roberta-base-SST-2) | 🚀 Uploading soon | 0.125B | vs O1: 0.8667<br>vs O2: 0.9080 | 0.9292 / 0.7681<br>0.9704 / 0.7826 | 0.9804 / 0.5302<br>0.9880 / 0.6653 | $32 \times (28 \sim 79)$ | Epoch=8, Sample=0.1 |
| **Proxy2_distil** | [`prajjwal1/bert-mini`](https://huggingface.co/prajjwal1/bert-mini) | 🚀 Uploading soon | 11.3M | vs O1: 0.7876<br>vs O2: 0.7954 | 0.8944 / 0.6255<br>0.8838 / 0.6615 | 0.9421 / 0.4646<br>0.9129 / 0.5983 | $32 \times (66 \sim 160)$ | Epoch=15, Sample=0.1 |
| **Proxy3_distil** | [`huawei-noah/TinyBERT_General_4L_312D`](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D) | 🚀 Uploading soon | 14M | vs O1: 0.8353<br>vs O2: 0.8319 | 0.9643 / 0.4062<br>0.9097 / 0.7064 | 0.9655 / 0.5020<br>0.9163 / 0.7178 | $32 \times (60 \sim 140)$ | Epoch=10, Sample=0.1 |
| **Proxy4_distil** | [`azizbarank/distilroberta-base-sst2-distilled`](https://huggingface.co/azizbarank/distilroberta-base-sst2-distilled) | 🚀 Uploading soon | 88M | vs O2: 0.8762 | 0.9516 / 0.7449<br>0.9186 / 0.8172 | 0.9646 / 0.7056<br>0.9246 / 0.8045 | $32 \times (50 \sim 125)$ | Epoch=10, Sample=0.1 |

---

### 2.3 TE - Base Backbone Fine-tuned Models (Fine-tuned from Base Architectures)

> **Note:**  
> - **HuggingFace Base Backbone & Link**: Pure pretrained encoder architectures from HuggingFace.  
> - **Our Fine-tuned Model & Link**: Models fine-tuned from scratch on task-sampled data. *(Released for custom benchmarking; not used in paper core results).*

| Model ID | HuggingFace Base Backbone & Link (Pretrained Only) | Our Fine-tuned Model & Link | # Params | Relative Accuracy Max($F_1$) | Max(Prec) / Rec | Max(Rec) / Prec | Throughput (items/s) | Fine-tuning Configuration |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Proxy1_base** | [`FacebookAI/roberta-base`](https://huggingface.co/FacebookAI/roberta-base) | 🚀 Uploading soon | 0.125B | vs O1: 0.8622 | 0.9553 / 0.6809<br>0.9484 / 0.7033 | 0.9637 / 0.5841<br>0.9350 / 0.7219 | $32 \times (28 \sim 79)$ | Epoch = 10 |
| **Proxy2_base** | [`prajjwal1/bert-mini`](https://huggingface.co/prajjwal1/bert-mini) | 🚀 Uploading soon | 11.3M | - | - | - | $32 \times (50 \sim 160)$ | - |
| **Proxy3_base** | [`huawei-noah/TinyBERT_General_4L_312D`](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D) | 🚀 Uploading soon | 14M | - | - | - | $32 \times (45 \sim 140)$ | - |

---

## 3. CV (Computer Vision / Image Texture & Product Classification)

* **Primary Dataset**: Amazon (e.g., identifying whether product images exhibit wooden/plastic/metal/fabric/glass textures, or determining product categories)
* **Usage Methodology**: All models are **used strictly off-the-shelf without any downstream fine-tuning (Zero-Shot / Off-the-shelf)**. Pretrained weights are downloaded directly from Hugging Face, and binary classification is performed by computing image-text similarity via text prompts (e.g., `"a photo of wooden texture"` vs. `"a photo of other texture"`).
* **Recommended Configurations**:
  * **Oracle Model**: **`Oracle1 (google/siglip-so400m-patch14-384)`** (Primary experimental judge).
  * **Default Proxy Model**: **`Proxy1 (google/siglip-base-patch32-224)`**, maintaining $F_1 \approx 0.7546$ while achieving a **$26.8\times$ end-to-end inference speedup**.

### 3.1 Image Classification Model Inventory and Benchmark

| Model ID | Hugging Face Repository & Link | # Params | Input Resolution | Relative Accuracy Max($F_1$) | Speedup (vs. Oracle1) | Throughput (items/s @ RTX 3090) | Notes / Architecture Details |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Oracle1** ⭐ | [`google/siglip-so400m-patch14-384`](https://huggingface.co/google/siglip-so400m-patch14-384) | **878M** | $384 \times 384$ | **Baseline (1.0)** | $1.0\times$ (Baseline) | $32 \times 1.6$ (~50 it/s) | **Recommended Oracle**; Heavyweight high-resolution SigLIP model with superior visual representation. |
| **Oracle2** | [`openai/clip-vit-large-patch14`](https://huggingface.co/openai/clip-vit-large-patch14) | 428M | $224 \times 224$ | - | $1.8\times$ | $32 \times 2.8$ (~90 it/s) | Classic OpenAI ViT-Large architecture. |
| **Proxy1** ⭐ | [`google/siglip-base-patch32-224`](https://huggingface.co/google/siglip-base-patch32-224) | **84M** | $224 \times 224$ | **vs O1: 0.7546** | **$26.8\times$** | $32 \times 42.0$ (~1340 it/s) | **Recommended Proxy**; Large patch size, extremely high throughput, optimal cost-efficiency. |
| **Proxy2** | [`google/siglip-base-patch16-224`](https://huggingface.co/google/siglip-base-patch16-224) | 86M | $224 \times 224$ | vs O1: 0.7812 | $15.4\times$ | $32 \times 24.5$ (~780 it/s) | Fine-grained Patch16; slightly higher accuracy with moderate computational overhead. |
| **Proxy3** | [`wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M`](https://huggingface.co/wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M) | 59M | $224 \times 224$ | vs O1: 0.6840 | $35.2\times$ | $32 \times 55.0$ (~1760 it/s) | Ultra-lightweight distilled model (40M vision + 19M text). |
| **Proxy4** | [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32) | 88M | $224 \times 224$ | vs O1: 0.7235 | $24.5\times$ | $32 \times 38.0$ (~1210 it/s) | Industry-standard classic lightweight general-purpose CLIP backbone. |

---

## 4. Local Deployment and Directory Structure Guide

Downloaded or fine-tuned model checkpoint weights should be placed according to the following directory hierarchy:

```text
PROXY/
└── Model/
    ├── nli/
    │   ├── oracle/           # Stores ONNX / PyTorch weights (e.g., deberta-v2-xxlarge-mnli)
    │   └── proxy/            # Stores fine-tuned / pretrained weights (e.g., deberta-v3-base, bert-mini)
    ├── sentiment/
    │   ├── oracle/           # Stores roberta-large-sst2 weights
    │   └── proxy/            # Stores bert-mini-finetuned-sst2 weights
    └── cv/
        ├── oracle/           # Stores siglip-so400m weights
        └── proxy/            # Stores siglip-base weights
```

---

## 5. Model Execution and Usage Instructions

### 5.1 Quick Download and Inference Example

Models can be automatically fetched and evaluated for binary classification probability scoring via the Python `transformers` library:

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Example using the recommended NLI Proxy model
model_name = "sileod/deberta-v3-base-mnli-fever-anli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval().cuda()

premise = "The user expressed strong support for the candidate."
hypothesis = "This comment is in favor of the topic."

inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=256).to("cuda")
with torch.no_grad():
    logits = model(**inputs).logits
    # Extract positive class probability for Entailment as proxy score
    probs = torch.softmax(logits, dim=-1)
    proxy_score = probs[0][0].item()  # Extracted according to model-specific label mapping index

print(f"Proxy Score: {proxy_score:.4f}")
```