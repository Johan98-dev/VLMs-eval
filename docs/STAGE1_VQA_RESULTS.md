# Stage 1 — Closed-Ended & Binary VQA Benchmark Report (BCNB & CPTAC)

This document summarizes the complete **Stage 1 Closed-Ended Visual Question Answering (VQA)** evaluation across the **BCNB** (`7,274` samples, `1,058` whole-slide images) and **CPTAC** (`240` samples, `240` `.svs` whole-slide images) cohorts in **SlideBench**.

---

## 1. Executive Summary & Overall Benchmark Table

All three target Whole-Slide Image (WSI) Vision-Language Models (**SlideChat**, **PRISM2**, and **MLLM-HWSI**) were evaluated deterministically (`do_sample=False`, `seed=42`) on `2 × NVIDIA GeForce RTX 4090 (24 GB)` GPUs using [`scripts/run_vqa_eval.py`](../scripts/run_vqa_eval.py). Across all `22,542` model-question evaluations (`7,514 × 3`), **zero samples failed answer extraction** (`unparsed_samples = 0`).

| Benchmark / Split | Samples | SlideChat | PRISM2 | MLLM-HWSI | Best Model |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BCNB Full** (`.jpg`, 7 clinical tasks) | 7,274 | **54.07%** (3,933) | 41.61% (3,027) | 47.66% (3,467) | **SlideChat** |
| ├─ *BCNB Binary (`ER`, `PR`, `HER2`)* | 3,174 | **75.20%** (2,387) | 43.19% (1,371) | 61.34% (1,947) | **SlideChat** |
| └─ *BCNB Multiple-Choice (3–4 options)* | 4,100 | 37.71% (1,546) | **40.39%** (1,656) | 37.07% (1,520) | **PRISM2** |
| **CPTAC Full** (`.svs`, 4 tumor cohorts) | 240 | 50.42% (121) | **55.42%** (133) | 23.75% (57) | **PRISM2** |
| **Stage 1 Weighted Overall (`BCNB + CPTAC`)** | **7,514** | **53.95%** (4,054) | 42.05% (3,160) | 46.90% (3,524) | **SlideChat** |
| **Stage 1 Macro Average Across Cohorts** | — | **52.25%** | 48.52% | 35.71% | **SlideChat** |

---

## 2. Cohort 1: BCNB Full Evaluation (`7,274` VQA Samples / `1,058` Slides)

**Dataset Characteristics:** Early breast cancer core-needle biopsy WSIs (`1,058` `.jpg` slides accompanied by polygon ROI annotations in `.json` and pre-extracted 512-d CONCH patch features in `Feat/BCNB_patch_feat/`). Each slide is evaluated across up to 7 clinical tasks covering histological subtyping, hormone receptor status (`ER`, `PR`), `HER2` status and IHC expression (`0`, `1+`, `2+`, `3+`), Nottingham histological grading (`1`, `2`, `3`), and molecular subtyping (`Luminal A`, `Luminal B`, `HER2(+)`, `Triple negative`).

### 2.1 BCNB Accuracy by Question Type

| Question Type | Options | Samples | SlideChat | PRISM2 | MLLM-HWSI |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Binary (`binary`)** | 2 (`Positive` / `Negative`) | 3,174 | **75.20%** (2,387) | 43.19% (1,371) | 61.34% (1,947) |
| **Multiple-Choice (`multiple_choice`)** | 3 or 4 options | 4,100 | 37.71% (1,546) | **40.39%** (1,656) | 37.07% (1,520) |
| **BCNB Total** | — | **7,274** | **54.07%** (3,933) | 41.61% (3,027) | 47.66% (3,467) |

### 2.2 BCNB Accuracy by Clinical Task (`by_task`)

| Clinical Task | Question Type | Options | Samples | SlideChat | PRISM2 | MLLM-HWSI | Best Model |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tumor** (Histological subtype) | Multiple-Choice | 3 | 1,058 | 89.32% (945) | **89.98%** (952) | 57.37% (607) | **PRISM2** |
| **ER** (Estrogen receptor status) | Binary | 2 | 1,058 | **78.64%** (832) | 26.65% (282) | 60.02% (635) | **SlideChat** |
| **PR** (Progesterone receptor status) | Binary | 2 | 1,058 | **70.60%** (747) | 30.06% (318) | 58.60% (620) | **SlideChat** |
| **HER2** (Binary receptor status) | Binary | 2 | 1,058 | **76.37%** (808) | 72.87% (771) | 65.41% (692) | **SlideChat** |
| **HER2 Expression** (`0`, `1+`, `2+`, `3+`) | Multiple-Choice | 4 | 1,058 | 21.93% (232) | 24.01% (254) | **30.91%** (327) | **MLLM-HWSI** |
| **Histological grading** (`Grade 1, 2, 3`) | Multiple-Choice | 3 | 926 | 19.76% (183) | **26.35%** (244) | 23.97% (222) | **PRISM2** |
| **Molecular subtype** (4 intrinsic subtypes) | Multiple-Choice | 4 | 1,058 | 17.58% (186) | 19.47% (206) | **34.40%** (364) | **MLLM-HWSI** |

### Key Findings on BCNB
1. **Morphological Tumor Subtyping (`Tumor`):** Both **PRISM2** (`89.98%`) and **SlideChat** (`89.32%`) achieve near-90% accuracy distinguishing *Invasive ductal carcinoma*, *Invasive lobular carcinoma*, and *Other* subtypes.
2. **Binary Hormone & HER2 Receptor Prediction (`ER`, `PR`, `HER2`):** **SlideChat** leads substantially (`75.20%` binary accuracy), followed by **MLLM-HWSI** (`61.34%`). **PRISM2** performs well on `HER2` (`72.87%`) but exhibits strong negative-class bias on `ER` (`26.65%`) and `PR` (`30.06%`) in zero-shot closed-ended prompting.
3. **Fine-Grained Multi-Class Tasks (`HER2 Expression`, `Molecular subtype`):** **MLLM-HWSI** outperforms both SlideChat and PRISM2 on 4-way `HER2 Expression` (`30.91%` vs `24.01%` / `21.93%`) and 4-way `Molecular subtype` (`34.40%` vs `19.47%` / `17.58%`), benefiting from its explicit cell-level (`CellViT-256` + `DINOv2`) and region-level (`HIPT-4K`) hierarchical representations.

---

## 3. Cohort 2: CPTAC Full Evaluation (`240` VQA Samples / `240` `.svs` Slides)

**Dataset Characteristics:** `240` gigapixel Aperio `.svs` pyramidal whole-slide images (`20×` objective power, `~0.494 µm/px` MPP) spanning four cancer cohorts (`60` slides each):
- **CM:** Cutaneous Melanoma (`60` samples)
- **LSCC:** Lung Squamous Cell Carcinoma (`60` samples)
- **LUAD:** Lung Adenocarcinoma (`60` samples)
- **UCEC:** Uterine Corpus Endometrial Carcinoma (`60` samples)

All `240` questions are 4-option multiple-choice histological subtyping questions (`random baseline = 25.00%`). Pyramid-aware native `libopenslide.so.0` reading ([`WSIReader`](../src/datasets/feature_extractor.py)) was used to detect foreground tissue on low-resolution pyramid levels and extract up to `1,024` `256×256` tissue tiles (`SlideChat` CONCH / `PRISM2` Virchow2) or `16` `4096×4096` tissue-dense regions (`MLLM-HWSI`).

### 3.1 CPTAC Accuracy by Tumor Cohort (`by_task`)

| Tumor Cohort | Description | Options | Samples | SlideChat | PRISM2 | MLLM-HWSI | Best Model |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CM** | Cutaneous Melanoma | 4 | 60 | **51.67%** (31) | 21.67% (13) | 8.33% (5) | **SlideChat** |
| **LSCC** | Lung Squamous Cell Carcinoma | 4 | 60 | 58.33% (35) | **75.00%** (45) | 6.67% (4) | **PRISM2** |
| **LUAD** | Lung Adenocarcinoma | 4 | 60 | 40.00% (24) | **55.00%** (33) | 36.67% (22) | **PRISM2** |
| **UCEC** | Uterine Corpus Endometrial Carcinoma | 4 | 60 | 51.67% (31) | **70.00%** (42) | 43.33% (26) | **PRISM2** |
| **CPTAC Total** | All 4 Cohorts | 4 | **240** | 50.42% (121) | **55.42%** (133) | 23.75% (57) | **PRISM2** |

### Key Findings on CPTAC
1. **PRISM2 Leads Overall on CPTAC (`55.42%`):** Powered by **Virchow2 (`ViT-H/14`)** + Perceiver resampling, **PRISM2** achieves strong zero-shot histological subtyping accuracy on **LSCC** (`75.00%`), **UCEC** (`70.00%`), and **LUAD** (`55.00%`).
2. **SlideChat Shows Consistent Generalization Across All 4 Organs (`50.42%`):** Unlike PRISM2 and MLLM-HWSI, which drop on Cutaneous Melanoma (`CM`), **SlideChat** maintains `40.00%–58.33%` accuracy across all four tumor types, including **51.67%** on `CM`.
3. **MLLM-HWSI Sensitivity on Non-Breast/Lung/Endometrial Cohorts:** While **MLLM-HWSI** performs competitively on `UCEC` (`43.33%`) and `LUAD` (`36.67%`), it struggles on `CM` (`8.33%`) and `LSCC` (`6.67%`).

---

## 4. Computational Performance & Runtime Profile (`2 × RTX 4090`)

Note: On-demand feature extraction was parallelized across both GPUs (`cuda:0` and `cuda:1`) and cached under `../Datasets/SlideBench/{BCNB,CPTAC}/Feat/`. Once features are cached on disk, pure VQA inference latency is `<0.15 s` per question for all three models.

| Model | Visual Encoders | LLM Backbone | BCNB Median Latency (Cached) | BCNB Total Wall Time | CPTAC Total Wall Time (with `.svs` Extraction) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **SlideChat** | CONCH (`ViT-B/16`, 512-d) + LongNet | Qwen2.5-7B-Instruct (`bf16`) | `0.125 s` | `17.2 min` (CSV pre-extracted) | `60.8 min` (1 GPU) |
| **PRISM2** | Virchow2 (`ViT-H/14`, 1280-d) + Perceiver | Phi-3-Mini-3.8B (`bf16`) | `0.084 s` | `35.2 min` (2 GPUs parallel extract) | `35.0 min` (2 GPUs parallel extract) |
| **MLLM-HWSI** | HIPT-256 + HIPT-4K + CONCH + CellViT-256 + DINOv2 | Qwen2.5-7B-Instruct (`fp16`) | `0.050 s` | `203.5 min` (1 GPU extract + eval) | `118.2 min` (2 GPUs parallel extract) |

---

## 5. Output Artifacts & Reproducibility Index

- **Consolidated CSV Table:** [`results/stage1_vqa/stage1_vqa_consolidated_summary.csv`](../results/stage1_vqa/stage1_vqa_consolidated_summary.csv)
- **BCNB Full Runs:**
  - SlideChat: [`slidechat_bcnb_full_summary.json`](../results/stage1_vqa/slidechat_bcnb_full_summary.json) | [`slidechat_bcnb_full_predictions.jsonl`](../results/stage1_vqa/slidechat_bcnb_full_predictions.jsonl)
  - PRISM2: [`prism2_bcnb_full_summary.json`](../results/stage1_vqa/prism2_bcnb_full_summary.json) | [`prism2_bcnb_full_predictions.jsonl`](../results/stage1_vqa/prism2_bcnb_full_predictions.jsonl)
  - MLLM-HWSI: [`mllm-hwsi_bcnb_full_summary.json`](../results/stage1_vqa/mllm-hwsi_bcnb_full_summary.json) | [`mllm-hwsi_bcnb_full_predictions.jsonl`](../results/stage1_vqa/mllm-hwsi_bcnb_full_predictions.jsonl)
- **CPTAC Full Runs:**
  - SlideChat: [`slidechat_cptac_full_summary.json`](../results/stage1_vqa/slidechat_cptac_full_summary.json) | [`slidechat_cptac_full_predictions.jsonl`](../results/stage1_vqa/slidechat_cptac_full_predictions.jsonl)
  - PRISM2: [`prism2_cptac_full_summary.json`](../results/stage1_vqa/prism2_cptac_full_summary.json) | [`prism2_cptac_full_predictions.jsonl`](../results/stage1_vqa/prism2_cptac_full_predictions.jsonl)
  - MLLM-HWSI: [`mllm-hwsi_cptac_full_summary.json`](../results/stage1_vqa/mllm-hwsi_cptac_full_summary.json) | [`mllm-hwsi_cptac_full_predictions.jsonl`](../results/stage1_vqa/mllm-hwsi_cptac_full_predictions.jsonl)
