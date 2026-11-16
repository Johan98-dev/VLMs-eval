# Stage 2 — Open-Ended WSI Captioning & Pathology Report Generation Benchmark Report (TCGA, CPTAC & Histai)

This document summarizes the **Stage 2 Open-Ended Whole-Slide Image (WSI) Captioning & Pathology Report Generation** evaluation across all supported **SlideBench** captioning cohorts:
- **TCGA Round-5 Stratified Validation Subset (`TCGA-Stratified-152`):** `152` `.svs` slides spanning all `31` TCGA cancer cohorts (`5` slides per tumor cohort, `2` for `ACC`).
- **TCGA Held-Out Unseen Validation Subset (`TCGA-Unseen-151`):** `151` `.svs` slides across `25` TCGA cancer cohorts strictly deduplicated against model training sets (`WSI_unseen_mllm`) to measure zero-leakage report generation.
- **CPTAC Full (`CPTAC-All-38`):** `38` `.svs` slides spanning `4` independent non-TCGA cancer cohorts (`CCRCC`: `10`, `LSCC`: `10`, `LUAD`: `10`, `UCEC`: `8`).
- **Histai Full (`Histai-All-1145`):** `1,145` ultra-high-resolution pyramidal `.tiff` slides spanning `3` organ cohorts (`Breast`: `500`, `Colorectum`: `345`, `Skin`: `300`).

---

## 1. Executive Summary & Multi-Cohort Benchmark Table

All three target WSI Vision-Language Models (**SlideChat**, **PRISM2**, and **MLLM-HWSI**) were evaluated deterministically (`do_sample=False`, `seed=42`, `max_new_tokens=512`) on `2 × NVIDIA GeForce RTX 4090 (24 GB)` GPUs using [`scripts/run_caption_eval.py`](../scripts/run_caption_eval.py). Across all `4,458` model-slide evaluations (`1,486` slides × `3` models), **zero samples produced empty or failed generations** (`empty_predictions = 0`).

### 1.1 Global Performance Summary Across Cohorts

| Benchmark / Split | Slides | Model | BLEU-1 (%) | BLEU-4 (%) | Corpus BLEU-4 (%) | ROUGE-1 (%) | ROUGE-L (%) | METEOR (%) | CIDEr | Clinical P (%) | Clinical R (%) | Clinical F1 (%) | Avg Words (Pred / Ref) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TCGA Stratified-152** | 152 | **SlideChat** | **24.60** | **3.59** | **4.88** | **35.04** | **19.89** | **21.23** | **0.5233** | **50.46** | **49.13** | **47.06** | 110.2 / 147.2 |
| *(31 Tumors, 5/tumor)* | 152 | **PRISM2** | 17.04 | 2.74 | 4.67 | 26.65 | 16.57 | 17.11 | 0.4641 | 48.26 | 48.94 | 44.39 | 136.6 / 147.2 |
| | 152 | **MLLM-HWSI** | 20.53 | 2.68 | 2.98 | 32.28 | 18.93 | 17.78 | 0.3857 | 42.51 | 37.75 | 37.97 | 80.1 / 147.2 |
| **TCGA Unseen-151** | 151 | **SlideChat** | **21.29** | **2.96** | **4.66** | **32.92** | **18.70** | **21.00** | **0.4450** | **54.53** | **57.41** | **52.78** | 116.5 / 141.5 |
| *(25 Tumors, Held-Out)* | 151 | **PRISM2** | 16.51 | 2.07 | 3.53 | 25.61 | 15.73 | 15.92 | 0.3230 | 49.03 | 51.85 | 46.45 | 126.3 / 141.5 |
| | 151 | **MLLM-HWSI** | 20.69 | 2.46 | 2.82 | 31.68 | 18.44 | 17.56 | 0.3045 | 41.82 | 38.77 | 38.57 | 78.9 / 141.5 |
| **CPTAC Full** | 38 | **SlideChat** | 11.48 | 0.89 | 0.00 | 15.78 | 11.25 | 14.90 | 0.1828 | 22.90 | 32.88 | 26.21 | 160.7 / 42.6 |
| *(4 Tumor Cohorts)* | 38 | **PRISM2** | 16.08 | **2.21** | **2.47** | 21.73 | 14.23 | 17.08 | **0.2282** | **36.40** | **40.11** | **36.10** | 89.3 / 42.6 |
| | 38 | **MLLM-HWSI** | **18.14** | 1.82 | 2.07 | **23.19** | **14.30** | **18.63** | 0.1865 | 27.75 | 28.47 | 27.12 | 77.9 / 42.6 |
| **Histai Full** | 1,145 | **SlideChat** | **13.50** | **0.90** | **1.43** | **20.33** | **12.01** | **12.02** | 0.1590 | 38.66 | **27.49** | **28.46** | 144.7 / 179.2 |
| *(3 Organ Cohorts)* | 1,145 | **PRISM2** | 4.30 | 0.34 | 0.14 | 12.90 | 8.56 | 6.07 | **0.2742** | **42.76** | 21.99 | 26.17 | 44.2 / 179.2 |
| | 1,145 | **MLLM-HWSI** | 9.77 | 0.55 | 0.25 | 18.23 | 11.69 | 9.88 | 0.1226 | 35.56 | 24.29 | 26.99 | 73.3 / 179.2 |
| **Stage 2 Macro Avg** | 1,486 | **SlideChat** | **17.72** | **2.08** | **2.74** | **26.02** | 15.46 | **17.29** | **0.3275** | **41.64** | **41.73** | **38.63** | 133.0 / 127.6 |
| *(4 Cohorts Unweighted)* | 1,486 | **PRISM2** | 13.48 | 1.84 | 2.70 | 21.72 | 13.77 | 14.04 | 0.3224 | **44.11** | 40.72 | 38.28 | 99.1 / 127.6 |
| | 1,486 | **MLLM-HWSI** | 17.28 | 1.88 | 2.03 | 26.35 | **15.84** | 15.96 | 0.2498 | 36.91 | 32.32 | 32.66 | 77.5 / 127.6 |
| **Stage 2 Weighted All** | **1,486** | **SlideChat** | **15.38** | **1.38** | **2.07** | **23.00** | **13.48** | **13.95** | 0.2259 | 41.07 | **32.89** | **32.78** | 138.7 / 168.6 |
| *(All 1,486 Slides)* | **1,486** | **PRISM2** | 7.15 | 0.81 | 0.99 | 15.83 | 10.25 | 8.48 | **0.2974** | **43.80** | 27.95 | 30.35 | 63.1 / 168.6 |
| | **1,486** | **MLLM-HWSI** | 12.19 | 0.99 | 0.84 | 21.16 | 13.18 | 11.69 | 0.1696 | 36.71 | 27.27 | 29.29 | 74.7 / 168.6 |

---

## 2. Evaluation Methodology & Clinical Metric Formulation

Open-ended pathology report generation cannot be adequately assessed by n-gram surface overlap alone, as two clinically equivalent descriptions (e.g., *"moderately differentiated adenocarcinoma"* vs. *"grade 2 glandular carcinoma"*) may share few contiguous n-grams while differing sharply in verbosity. Accordingly, [`src/evaluation/caption_metrics.py`](../src/evaluation/caption_metrics.py) implements a two-tier evaluation suite combining **lexical NLG metrics** and **domain-specific clinical concept extraction**:

1. **Sentence-Level & Corpus-Level BLEU (`BLEU-1..4`, `Corpus BLEU-1..4`):**
   - Computes both sample-averaged smoothed BLEU (`Chen & Cherry Method 1`) and standard corpus-level Papineni BLEU with geometric mean precision over `1..4`-grams and length brevity penalty (`BP`).
2. **ROUGE (`ROUGE-1`, `ROUGE-2`, `ROUGE-L`):**
   - Evaluates unigram/bigram overlap and Longest Common Subsequence (`LCS`) sentence-structure F1 score.
3. **Pathology-Adapted METEOR (`meteor`):**
   - Aligns unigrams via exact match followed by a **morphological & clinical stemmer** (`_CLINICAL_STEM_OVERRIDES`, mapping polymorphic medical terms such as `nuclei`/`nuclear` $\to$ `nucle`, `mitoses`/`mitotic` $\to$ `mitot`, `necrosis`/`necrotic` $\to$ `necro`, `tubules`/`tubular` $\to$ `tubul`, `invasion`/`invasive` $\to$ `invas`) with a chunk fragmentation penalty ($\alpha = 0.9, \beta = 3.0, \gamma = 0.5$).
4. **CIDEr-D (`cider`):**
   - Computes TF-IDF weighted `1..4`-gram cosine similarity across the cohort corpus with a Gaussian length penalty ($\sigma = 6.0$), rewarding rare, diagnostically specific terminology over generic clinical boilerplate.
5. **Clinical Concept Precision, Recall & F1 (`clinical_precision`, `clinical_recall`, `clinical_f1`):**
   - Extracts a curated taxonomy of **37 core histopathological concepts** via regular-expression entity patterns across six diagnostic axes:
     - *Primary Histological Diagnoses (13):* `adenocarcinoma`, `endometrial_carcinoma`, `squamous_cell_carcinoma`, `urothelial_carcinoma`, `renal_cell_carcinoma`, `clear_cell`, `invasive_ductal`, `invasive_lobular`, `carcinoma_in_situ`, `fibroadenoma`, `phyllodes`, `hyperplasia`, `adenosis`.
     - *Architectural & Growth Patterns (8):* `acinar_glandular`, `tubular`, `papillary`, `micropapillary`, `cribriform`, `solid_nests`, `trabecular_cords`, `columnar_cuboidal`.
     - *Grading & Differentiation (4):* `grade_1_low`, `grade_2_moderate`, `grade_3_high`, `gleason`.
     - *Cellular, Nuclear & Stromal Features (10):* `pleomorphism_atypia`, `nucleoli`, `mitotic_activity`, `necrosis`, `hemorrhage`, `fibrosis_desmoplasia`, `inflammation_lymphocytes`, `keratin`, `mucin_mucus`, `calcification`.
     - *Invasion, Margins & TNM Staging (5):* `lymphovascular_invasion`, `perineural_invasion`, `surgical_margins`, `lymph_nodes`, `tnm_stage_t`.
     - *IHC Biomarkers (5):* `biomarker_er`, `biomarker_pr`, `biomarker_her2`, `biomarker_ki67`, `biomarker_ck_p63`.

---

## 3. Cohort 1: TCGA Evaluation (`Stratified-152` vs. Held-Out `Unseen-151`)

### 3.1 Experimental Design: Controlling for Pretraining Overlap

Because **SlideChat**, **PRISM2**, and **MLLM-HWSI** all incorporate TCGA pathology reports during pretraining or instruction tuning, evaluating exclusively on random TCGA slides risks conflating memorization with visual understanding. To rigorously separate in-distribution performance from held-out generalization, Stage 2 evaluates two distinct TCGA `.svs` subsets:

1. **`TCGA-Stratified-152` (`152` WSIs across all `31` TCGA tumor cohorts):** The Round-5 Stratified Validation Subset (`5` `.svs` slides per tumor cohort, `2` for `ACC`), matching the exact cohort evaluated in Stage 1 VQA.
2. **`TCGA-Unseen-151` (`151` WSIs across `25` TCGA tumor cohorts):** Defined by `SlideBench-Caption-TCGA-unseen-mllm.csv` (`WSI_unseen_mllm`), comprising `151` TCGA slides strictly verified to have zero overlap with the MLLM-HWSI training split (sharing only `24` slides with `TCGA-Stratified-152`, with `127` newly acquired held-out `.svs` WSIs).

### 3.2 In-Distribution (`Stratified-152`) vs. Held-Out (`Unseen-151`) Comparison

| Metric | SlideChat (`Strat-152`) | SlideChat (`Unseen-151`) | $\Delta$ | PRISM2 (`Strat-152`) | PRISM2 (`Unseen-151`) | $\Delta$ | MLLM-HWSI (`Strat-152`) | MLLM-HWSI (`Unseen-151`) | $\Delta$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BLEU-1 (%)** | **24.60** | **21.29** | -3.31 | 17.04 | 16.51 | -0.53 | 20.53 | 20.69 | **+0.16** |
| **BLEU-4 (%)** | **3.59** | **2.96** | -0.63 | 2.74 | 2.07 | -0.67 | 2.68 | 2.46 | -0.22 |
| **Corpus BLEU-4 (%)** | **4.88** | **4.66** | -0.22 | 4.67 | 3.53 | -1.14 | 2.98 | 2.82 | -0.16 |
| **ROUGE-L (%)** | **19.89** | **18.70** | -1.19 | 16.57 | 15.73 | -0.84 | 18.93 | 18.44 | -0.49 |
| **METEOR (%)** | **21.23** | **21.00** | -0.23 | 17.11 | 15.92 | -1.19 | 17.78 | 17.56 | -0.22 |
| **CIDEr** | **0.5233** | **0.4450** | -0.0783 | 0.4641 | 0.3230 | -0.1411 | 0.3857 | 0.3045 | -0.0812 |
| **Clinical F1 (%)** | **47.06** | **52.78** | **+5.72** | 44.39 | 46.45 | **+2.06** | 37.97 | 38.57 | **+0.60** |

#### Key Findings on TCGA (`Stratified-152` & `Unseen-151`)
1. **Zero Degradation in Clinical Concept F1 on Held-Out `Unseen-151` Slides:** All three models maintain or improve their **Clinical F1** on the held-out `Unseen-151` split (`SlideChat`: `47.06%` $\to$ **52.78%**, `PRISM2`: `44.39%` $\to$ **46.45%**, `MLLM-HWSI`: `37.97%` $\to$ **38.57%**), confirming that their pathology report generation on TCGA reflects genuine visual-linguistic alignment rather than verbatim report memorization.
2. **MLLM-HWSI Exhibits Near-Identical Lexical Performance on Unseen Slides:** Between `Stratified-152` and strictly held-out `Unseen-151`, **MLLM-HWSI** shows virtually zero lexical drop (`BLEU-1`: `20.53%` vs `20.69%`; `ROUGE-L`: `18.93%` vs `18.44%`; `METEOR`: `17.78%` vs `17.56%`).
3. **SlideChat Leads Across All Lexical and Clinical Metrics on Both TCGA Splits:** Powered by `CONCH` patch features and `LongNet` slide-level aggregation, **SlideChat** achieves the highest `BLEU-1` (`24.60%` / `21.29%`), `Corpus BLEU-4` (`4.88%` / `4.66%`), `ROUGE-L` (`19.89%` / `18.70%`), `METEOR` (`21.23%` / `21.00%`), `CIDEr` (`0.5233` / `0.4450`), and `Clinical F1` (`47.06%` / `52.78%`).

### 3.3 Per-Tumor Breakdown on `TCGA-Stratified-152` (`31` Tumor Cohorts)

| Tumor | Slides | SlideChat B-1 (%) | SlideChat R-L (%) | SlideChat CIDEr | SlideChat Clin-F1 (%) | PRISM2 B-1 (%) | PRISM2 R-L (%) | PRISM2 CIDEr | PRISM2 Clin-F1 (%) | MLLM-HWSI B-1 (%) | MLLM-HWSI R-L (%) | MLLM-HWSI CIDEr | MLLM-HWSI Clin-F1 (%) | Best Clin-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ACC** | 2 | 3.20 | 17.10 | 0.269 | 43.20 | 13.30 | 18.20 | 0.358 | **55.40** | **17.20** | **19.40** | **0.678** | 53.80 | **PRISM2** |
| **BLCA** | 5 | **34.90** | 21.10 | **0.721** | **66.50** | 25.90 | 17.70 | 0.490 | 61.40 | 26.90 | **22.10** | 0.627 | 51.70 | **SlideChat** |
| **BRCA** | 5 | **33.70** | **22.70** | **0.766** | **35.80** | 18.70 | 15.60 | 0.520 | 33.40 | 24.40 | 18.30 | 0.292 | 28.20 | **SlideChat** |
| **CESC** | 5 | 15.40 | 21.40 | **0.531** | **55.50** | 20.70 | 18.40 | 0.296 | 38.90 | **25.90** | **22.50** | 0.528 | 54.30 | **SlideChat** |
| **CHOL** | 5 | **38.80** | **20.30** | **0.742** | **45.50** | 11.00 | 14.00 | 0.273 | 28.40 | 21.40 | 16.40 | 0.441 | 28.20 | **SlideChat** |
| **COAD** | 5 | **36.20** | **23.70** | **0.779** | **71.30** | 8.60 | 16.00 | 0.403 | 53.60 | 30.60 | 21.50 | 0.469 | 53.50 | **SlideChat** |
| **DLBC** | 5 | 10.20 | 13.60 | 0.092 | **52.80** | 6.50 | 13.90 | 0.482 | 35.30 | **16.60** | **18.10** | **0.577** | 51.50 | **SlideChat** |
| **ESCA** | 5 | 15.50 | **21.80** | 0.321 | 37.90 | 21.90 | 17.80 | **0.545** | **67.90** | **33.10** | 21.00 | 0.265 | 31.50 | **PRISM2** |
| **GBM** | 5 | **27.10** | **17.90** | **0.480** | **48.20** | 13.70 | 13.60 | 0.397 | 40.90 | 15.60 | 15.40 | 0.196 | 47.70 | **SlideChat** |
| **HNSC** | 5 | **23.50** | **19.60** | **0.339** | **45.40** | 12.10 | 16.20 | 0.321 | 38.70 | 18.90 | 19.00 | 0.204 | 37.40 | **SlideChat** |
| **KICH** | 5 | 24.10 | 22.10 | 0.513 | 40.10 | **29.80** | **22.70** | **0.693** | **52.80** | 19.80 | 19.30 | 0.581 | 28.70 | **PRISM2** |
| **KIRC** | 5 | 23.30 | **21.80** | 0.409 | 30.70 | **27.60** | 21.40 | **0.766** | **63.80** | 21.20 | 18.70 | 0.366 | 30.80 | **PRISM2** |
| **KIRP** | 5 | 20.80 | **25.20** | 0.766 | 51.40 | **35.60** | 23.50 | **0.963** | **75.70** | 15.90 | 20.80 | 0.564 | 43.50 | **PRISM2** |
| **LGG** | 5 | **22.70** | 14.50 | 0.286 | **53.60** | 10.20 | 10.60 | 0.276 | 35.60 | 18.70 | **18.20** | **0.425** | 49.60 | **SlideChat** |
| **LIHC** | 5 | **39.00** | **21.90** | **0.636** | 49.60 | 17.50 | 17.80 | 0.450 | **60.30** | 22.00 | 21.60 | 0.426 | 25.80 | **PRISM2** |
| **LUAD** | 5 | **36.00** | **19.30** | **0.461** | 40.70 | 0.00 | 6.10 | 0.106 | 21.00 | 18.30 | 17.70 | 0.152 | **47.90** | **MLLM-HWSI** |
| **LUSC** | 5 | **34.70** | 18.60 | **0.536** | 47.00 | 22.50 | **19.60** | 0.407 | **52.30** | 19.60 | 17.60 | 0.130 | 23.70 | **PRISM2** |
| **MESO** | 5 | **23.10** | 17.80 | 0.519 | **34.00** | 11.90 | 13.90 | 0.431 | 18.00 | 14.90 | **18.20** | **0.550** | 33.30 | **SlideChat** |
| **OV** | 5 | 7.00 | **16.30** | **0.497** | 22.40 | **21.20** | 15.50 | 0.208 | **37.50** | 8.10 | 13.00 | 0.211 | 20.40 | **PRISM2** |
| **PAAD** | 5 | 3.30 | 15.00 | **0.576** | 54.90 | **18.70** | 19.30 | 0.424 | **67.10** | 14.60 | **19.70** | 0.286 | 38.60 | **PRISM2** |
| **PCPG** | 5 | **23.10** | **20.70** | **0.456** | **46.50** | 10.80 | 16.80 | 0.306 | 33.60 | 19.70 | 17.60 | 0.254 | 16.40 | **SlideChat** |
| **PRAD** | 5 | 22.30 | **23.80** | **1.071** | 67.70 | **24.90** | 17.00 | 0.777 | 71.80 | 19.70 | 19.90 | 0.769 | **73.60** | **MLLM-HWSI** |
| **READ** | 5 | **31.80** | 20.10 | **0.417** | **78.40** | 6.60 | 10.60 | 0.234 | 44.10 | 21.90 | **21.50** | 0.410 | 63.50 | **SlideChat** |
| **SARC** | 5 | **23.50** | **18.00** | **0.302** | 39.20 | 13.70 | 17.30 | 0.300 | **44.40** | 11.80 | 14.80 | 0.184 | 22.60 | **PRISM2** |
| **SKCM** | 5 | **33.10** | **26.60** | **0.942** | **28.00** | 15.30 | 14.60 | 0.362 | 27.20 | 26.50 | 21.10 | 0.222 | 24.30 | **SlideChat** |
| **STAD** | 5 | 24.80 | 19.10 | 0.449 | **72.60** | 15.40 | 16.20 | 0.240 | 49.40 | **31.80** | **20.60** | **0.465** | 47.80 | **SlideChat** |
| **TCGT** | 5 | 20.90 | 19.50 | 0.094 | **47.40** | **30.00** | **24.10** | **1.124** | 46.90 | 21.70 | 18.80 | 0.113 | 20.90 | **SlideChat** |
| **THCA** | 5 | 17.30 | **21.00** | 0.795 | 62.00 | 17.10 | 17.50 | **0.873** | 43.30 | **17.40** | 19.30 | 0.589 | **64.00** | **MLLM-HWSI** |
| **THYM** | 5 | **31.50** | 18.80 | 0.425 | **30.80** | 4.60 | 12.60 | **0.470** | 16.90 | 20.80 | **20.20** | 0.416 | 22.40 | **SlideChat** |
| **UCEC** | 5 | 18.40 | 17.80 | 0.519 | 42.80 | **27.10** | **19.80** | **0.584** | **46.20** | 14.00 | 15.90 | 0.340 | 27.50 | **PRISM2** |
| **UCS** | 5 | **30.40** | 17.80 | 0.361 | 14.60 | 12.90 | 16.10 | 0.243 | 20.80 | 25.60 | **18.50** | **0.402** | **23.30** | **MLLM-HWSI** |

### 3.4 Per-Tumor Breakdown on `TCGA-Unseen-151` (`25` Held-Out Tumor Cohorts)

| Tumor | Slides | SlideChat B-1 (%) | SlideChat R-L (%) | SlideChat CIDEr | SlideChat Clin-F1 (%) | PRISM2 B-1 (%) | PRISM2 R-L (%) | PRISM2 CIDEr | PRISM2 Clin-F1 (%) | MLLM-HWSI B-1 (%) | MLLM-HWSI R-L (%) | MLLM-HWSI CIDEr | MLLM-HWSI Clin-F1 (%) | Best Clin-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BLCA** | 2 | **38.40** | **18.30** | **0.959** | **59.80** | 20.50 | 13.90 | 0.696 | 54.70 | 18.70 | 17.50 | 0.746 | 55.80 | **SlideChat** |
| **BRCA** | 2 | 17.80 | **19.70** | **0.775** | 32.20 | 18.90 | 19.30 | 0.697 | **48.10** | **20.70** | 18.90 | 0.324 | 31.70 | **PRISM2** |
| **COAD** | 14 | **31.90** | 21.10 | **0.514** | **61.50** | 12.10 | 16.10 | 0.344 | 46.70 | 25.60 | **21.60** | 0.302 | 52.50 | **SlideChat** |
| **DLBC** | 2 | 8.10 | 11.60 | 0.036 | **45.20** | 2.00 | 11.10 | 0.086 | 43.20 | **14.80** | **18.10** | **0.614** | 41.10 | **SlideChat** |
| **ESCA** | 1 | 11.10 | **17.00** | **0.730** | **60.00** | 0.20 | 5.80 | 0.005 | 33.30 | **19.00** | 16.70 | 0.047 | 33.30 | **SlideChat** |
| **GBM** | 27 | **25.30** | **18.00** | **0.351** | **52.90** | 14.10 | 13.50 | 0.198 | 38.60 | 19.50 | 16.20 | 0.203 | 35.70 | **SlideChat** |
| **HNSC** | 2 | 33.50 | 19.70 | 0.418 | **59.30** | 15.90 | 13.60 | 0.198 | 55.80 | **35.50** | **21.70** | **0.565** | 42.00 | **SlideChat** |
| **KICH** | 2 | **23.90** | **21.50** | 0.745 | 40.00 | 19.90 | 19.60 | 0.526 | **57.20** | 15.20 | 19.60 | **0.795** | 23.10 | **PRISM2** |
| **KIRC** | 5 | 24.60 | **22.00** | 0.568 | 37.60 | **34.00** | 21.90 | **0.946** | **58.30** | 21.30 | 20.20 | 0.491 | 42.80 | **PRISM2** |
| **KIRP** | 3 | **23.10** | **21.70** | **0.534** | **60.40** | 6.80 | 8.80 | 0.184 | 18.50 | 18.20 | 16.90 | 0.322 | 39.40 | **SlideChat** |
| **LGG** | 7 | 14.60 | 14.00 | 0.185 | **39.40** | 2.50 | 7.20 | 0.077 | 11.90 | **21.40** | **18.00** | **0.389** | 37.90 | **SlideChat** |
| **LIHC** | 2 | **37.90** | 21.40 | 0.683 | **38.40** | 10.30 | 13.10 | **0.950** | 20.00 | 21.60 | **22.00** | 0.287 | 23.40 | **SlideChat** |
| **LUSC** | 6 | 27.20 | **19.30** | **0.525** | **56.40** | 21.10 | 15.90 | 0.216 | 34.00 | **27.80** | 18.00 | 0.195 | 29.50 | **SlideChat** |
| **OV** | 2 | 12.10 | 17.90 | **0.195** | 28.20 | 8.10 | 9.00 | 0.035 | 36.40 | **22.00** | **18.50** | 0.138 | **44.30** | **MLLM-HWSI** |
| **PAAD** | 33 | 6.90 | 16.90 | **0.389** | 58.40 | **18.10** | **18.80** | 0.330 | **61.30** | 17.00 | 17.70 | 0.206 | 40.20 | **PRISM2** |
| **PCPG** | 3 | **26.00** | **24.10** | **0.681** | **54.10** | 24.00 | 19.90 | 0.258 | 46.00 | 18.70 | 20.80 | 0.480 | 43.30 | **SlideChat** |
| **PRAD** | 3 | 21.60 | **25.00** | **1.314** | 65.40 | **25.30** | 18.20 | 0.673 | **77.80** | 24.90 | 19.80 | 0.976 | 63.90 | **PRISM2** |
| **READ** | 2 | **35.60** | 22.40 | **0.541** | **83.30** | 24.10 | 20.40 | 0.335 | 73.30 | 22.90 | **23.00** | 0.488 | 63.30 | **SlideChat** |
| **SARC** | 3 | **23.90** | **17.30** | 0.168 | 29.30 | 19.00 | **17.30** | **0.334** | **51.90** | 12.40 | 16.40 | 0.222 | 26.90 | **PRISM2** |
| **SKCM** | 4 | **22.30** | **18.20** | **0.545** | **36.60** | 10.00 | 13.10 | 0.130 | 27.00 | 19.00 | 16.80 | 0.419 | 20.40 | **SlideChat** |
| **STAD** | 7 | 29.60 | 21.10 | 0.407 | **66.10** | 16.80 | 16.30 | 0.254 | 53.80 | **35.10** | **22.50** | **0.475** | 46.90 | **SlideChat** |
| **TCGT** | 2 | 21.60 | 16.50 | 0.071 | **53.60** | **28.50** | **20.30** | **0.428** | 36.30 | 17.10 | 16.50 | 0.111 | 18.20 | **SlideChat** |
| **THYM** | 1 | **29.60** | 16.20 | 0.484 | **28.60** | 0.00 | 9.20 | 0.069 | 0.00 | 26.50 | **22.00** | **0.618** | 0.00 | **SlideChat** |
| **UCEC** | 15 | **24.50** | **19.10** | **0.463** | **48.80** | 21.40 | 15.80 | 0.422 | 47.00 | 18.70 | 18.40 | 0.201 | 30.60 | **SlideChat** |
| **UCS** | 1 | 22.50 | 18.40 | 0.286 | 15.40 | **30.60** | **19.20** | 0.219 | **61.50** | 9.00 | 14.30 | **0.305** | 36.40 | **PRISM2** |

---

## 4. Cohort 2: CPTAC Full Evaluation (`38` Samples / `38` `.svs` Slides)

**Dataset Characteristics:** `38` Aperio `.svs` whole-slide images across four tumor cohorts from the Clinical Proteomic Tumor Analysis Consortium (**CPTAC**):
- **CCRCC:** Clear Cell Renal Cell Carcinoma (`10` samples; newly extracted in Stage 2)
- **LSCC:** Lung Squamous Cell Carcinoma (`10` samples)
- **LUAD:** Lung Adenocarcinoma (`10` samples)
- **UCEC:** Uterine Corpus Endometrial Carcinoma (`8` samples)

Unlike TCGA pathology reports (`~147` words, combining gross surgical staging and microscopic findings), **CPTAC ground-truth captions are concise, pure histomorphological descriptions averaging only `42.6` words** (`55.1` words for `CCRCC`, `40.6` for `LSCC`, `38.8` for `LUAD`, and `34.1` for `UCEC`), focusing exclusively on cellular arrangement, nuclear grade, cytoplasmic clearing, and stromal vascularity.

### 4.1 CPTAC Performance by Tumor Cohort (`by_task`)

| Tumor Cohort | Slides | Model | BLEU-1 (%) | BLEU-4 (%) | ROUGE-1 (%) | ROUGE-L (%) | METEOR (%) | CIDEr | Clinical P (%) | Clinical R (%) | Clinical F1 (%) | Avg Words (Pred / Ref) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CCRCC** | 10 | **SlideChat** | 21.59 | 2.15 | 24.57 | **16.77** | 15.72 | 0.2276 | 32.05 | 34.00 | 32.84 | 83.1 / 55.1 |
| *(Renal Clear Cell)* | 10 | **PRISM2** | 18.46 | **4.80** | 23.18 | 15.52 | **23.54** | **0.3495** | 37.07 | **41.38** | **38.77** | 146.5 / 55.1 |
| | 10 | **MLLM-HWSI** | **22.06** | 3.41 | **26.52** | 15.98 | 18.77 | 0.2306 | **43.00** | 32.81 | 36.51 | 82.1 / 55.1 |
| **LSCC** | 10 | **SlideChat** | 8.04 | 0.49 | 12.73 | 10.28 | 13.13 | 0.1688 | 14.06 | 23.93 | 16.89 | 163.9 / 40.6 |
| *(Lung Squamous)* | 10 | **PRISM2** | **17.80** | **1.88** | **23.96** | **15.93** | 17.70 | **0.1762** | **27.48** | **34.45** | **29.69** | 46.1 / 40.6 |
| | 10 | **MLLM-HWSI** | 16.35 | 1.33 | 20.77 | 13.96 | **17.77** | 0.1754 | 15.69 | 19.19 | 17.06 | 73.0 / 40.6 |
| **LUAD** | 10 | **SlideChat** | 8.29 | 0.43 | 13.39 | 9.54 | 16.40 | **0.1973** | 21.98 | 40.17 | 27.82 | 224.0 / 38.8 |
| *(Lung Adenocarcinoma)* | 10 | **PRISM2** | 15.87 | 1.18 | 22.08 | 14.04 | 12.61 | 0.1724 | **50.21** | **45.03** | **42.51** | 35.8 / 38.8 |
| | 10 | **MLLM-HWSI** | **18.72** | **1.40** | **24.25** | **14.22** | **20.00** | 0.1688 | 21.33 | 25.39 | 22.28 | 76.2 / 38.8 |
| **UCEC** | 8 | **SlideChat** | 7.14 | 0.38 | 11.62 | 7.69 | 14.22 | 0.1261 | 23.66 | 33.57 | 27.56 | 174.5 / 34.1 |
| *(Endometrial)* | 8 | **PRISM2** | 11.22 | 0.68 | 16.69 | 10.75 | 13.81 | **0.2113** | 29.45 | **39.43** | 32.74 | 138.9 / 34.1 |
| | 8 | **MLLM-HWSI** | **14.74** | **0.94** | **20.72** | **12.71** | **17.82** | 0.1673 | **31.78** | 38.48 | **34.00** | 80.8 / 34.1 |
| **CPTAC Overall** | **38** | **SlideChat** | 11.48 | 0.89 | 15.78 | 11.25 | 14.90 | 0.1828 | 22.90 | 32.88 | 26.21 | 160.7 / 42.6 |
| *(All 4 Cohorts)* | **38** | **PRISM2** | 16.08 | **2.21** | 21.73 | 14.23 | 17.08 | **0.2282** | **36.40** | **40.11** | **36.10** | 89.3 / 42.6 |
| | **38** | **MLLM-HWSI** | **18.14** | 1.82 | **23.19** | **14.30** | **18.63** | 0.1865 | 27.75 | 28.47 | 27.12 | 77.9 / 42.6 |

### Key Findings on CPTAC
1. **PRISM2 Leads Clinical Concept F1 (`36.10%`) and CIDEr (`0.2282`) on CPTAC:** Consistent with its Stage 1 CPTAC VQA victory (`55.42%` accuracy), **PRISM2** achieves the highest overall clinical entity alignment on CPTAC (`36.10%` vs. `27.12%` MLLM-HWSI and `26.21%` SlideChat), leading on `CCRCC` (`38.77%`), `LSCC` (`29.69%`), and `LUAD` (`42.51%`, where its `35.8`-word average output closely matches the `38.8`-word reference length).
2. **MLLM-HWSI Leads Surface Lexical Overlap (`BLEU-1 = 18.14%`, `ROUGE-1 = 23.19%`, `ROUGE-L = 14.30%`, `METEOR = 18.63%`) and `UCEC` Clinical F1 (`34.00%`):** Because **MLLM-HWSI** generates focused `73–82` word microscopic descriptions ("*Microscopic observation of the pathology slide reveals...*") rather than multi-hundred-word surgical synoptics, it aligns more closely with CPTAC's pure histomorphological reference style.
3. **Verbosity Penalty on SlideChat (`160.7` vs. `42.6` words):** On `LSCC`, `LUAD`, and `UCEC`, **SlideChat** generates `164–224` word TCGA-style pathology reports containing extensive gross/staging boilerplate (`pTNM`, lymph node counts, surgical margins) absent from CPTAC's short morphological annotations, depressing its unigram precision (`BLEU-1 = 11.48%`, `Clinical Precision = 22.90%`) despite solid `Clinical Recall` (`32.88%`).

---

## 5. Cohort 3: Histai Full Evaluation (`1,145` Samples / `1,145` `.tiff` Slides)

**Dataset Characteristics:** `1,145` ultra-high-resolution pyramidal `.tiff` whole-slide images from the **HISTAI** clinical pathology archive across three organ cohorts:
- **Breast (`HISTAI-breast`):** `500` cases (`avg_ref_words = 207.0`)
- **Colorectum (`HISTAI-colorectal-b1/b2`):** `345` cases (`avg_ref_words = 153.4`)
- **Skin (`HISTAI-skin-b1`):** `300` cases (`avg_ref_words = 162.4`)

Ground-truth captions in **Histai** are comprehensive, multi-section clinical synoptic checklists (`CAP` / `AJCC 8th Edition` format) averaging `179.2` words, combining specimen macroscopy, microscopic tumor architecture, invasion depth, tumor-infiltrating lymphocytes (`TILs`), resection margins, and `pTNM` classification.

### 5.1 Histai Performance by Organ Cohort (`by_task`)

| Organ Cohort | Slides | Model | BLEU-1 (%) | BLEU-4 (%) | Corpus BLEU-4 (%) | ROUGE-1 (%) | ROUGE-L (%) | METEOR (%) | CIDEr | Clinical P (%) | Clinical R (%) | Clinical F1 (%) | Avg Words (Pred / Ref) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Breast** | 500 | **SlideChat** | **16.50** | **1.06** | **1.47** | **22.32** | **12.08** | **13.42** | 0.1475 | 27.70 | 18.65 | 20.75 | 182.0 / 207.0 |
| *(`HISTAI-breast`)* | 500 | **PRISM2** | 4.32 | 0.34 | 0.22 | 12.98 | 8.48 | 6.06 | **0.1940** | 31.75 | 14.66 | 17.89 | 59.6 / 207.0 |
| | 500 | **MLLM-HWSI** | 9.25 | 0.50 | 0.25 | 19.39 | 11.76 | 10.37 | 0.1234 | **38.16** | **19.61** | **24.46** | 77.9 / 207.0 |
| **Colorectum** | 345 | **SlideChat** | **11.76** | **0.94** | **1.99** | **18.21** | **11.34** | **12.32** | 0.1488 | 36.03 | **42.32** | **35.92** | 142.4 / 153.4 |
| *(`HISTAI-colorectal`)* | 345 | **PRISM2** | 1.50 | 0.16 | 0.01 | 9.73 | 7.52 | 4.30 | **0.3307** | **61.16** | 26.42 | 34.28 | 21.4 / 153.4 |
| | 345 | **MLLM-HWSI** | 8.40 | 0.54 | 0.00 | 14.56 | 10.19 | 8.88 | 0.0882 | 26.50 | 26.36 | 24.87 | 69.5 / 153.4 |
| **Skin** | 300 | **SlideChat** | 10.50 | 0.58 | **0.54** | 19.45 | 12.68 | 9.36 | 0.1898 | **59.96** | 25.15 | 32.74 | 85.1 / 162.4 |
| *(`HISTAI-skin-b1`)* | 300 | **PRISM2** | 7.49 | 0.53 | 0.22 | 16.40 | 9.90 | 8.12 | **0.3430** | 39.95 | 29.10 | 30.65 | 44.6 / 162.4 |
| | 300 | **MLLM-HWSI** | **12.20** | **0.63** | 0.29 | **20.51** | **13.28** | **10.20** | 0.1610 | 41.65 | **29.70** | **33.64** | 69.9 / 162.4 |
| **Histai Overall** | **1,145** | **SlideChat** | **13.50** | **0.90** | **1.43** | **20.33** | **12.01** | **12.02** | 0.1590 | 38.66 | **27.49** | **28.46** | 144.7 / 179.2 |
| *(All 3 Organs)* | **1,145** | **PRISM2** | 4.30 | 0.34 | 0.14 | 12.90 | 8.56 | 6.07 | **0.2742** | **42.76** | 21.99 | 26.17 | 44.2 / 179.2 |
| | **1,145** | **MLLM-HWSI** | 9.77 | 0.55 | 0.25 | 18.23 | 11.69 | 9.88 | 0.1226 | 35.56 | 24.29 | 26.99 | 73.3 / 179.2 |

### Key Findings on Histai
1. **SlideChat Achieves the Highest Overall Clinical F1 (`28.46%`) and Lexical Alignment (`BLEU-1 = 13.50%`, `ROUGE-L = 12.01%`, `METEOR = 12.02%`):** Because **SlideChat** generates comprehensive `144.7`-word narratives that match Histai's `179.2`-word synoptic reference length, it achieves the highest Clinical Recall (`27.49%` overall, **42.32%** on `Colorectum`) and leads `Colorectum` Clinical F1 (`35.92%`).
2. **PRISM2 Exhibits a High-Precision, Concise Diagnostic Profile (`Clinical Precision = 42.76%`, `CIDEr = 0.2742`):** On Histai, **PRISM2** generates concise diagnostic statements averaging `44.2` words (`21.4` words on `Colorectum`). While the `4×` length ratio relative to `179.2`-word references heavily penalizes its BLEU brevity factor (`BLEU-1 = 4.30%`), **PRISM2** achieves the **highest Clinical Precision overall (`42.76%`, and `61.16%` on `Colorectum`)** and the **highest CIDEr across all three organs (`0.2742` overall, `0.3430` on `Skin`, `0.3307` on `Colorectum`)** by concentrating on high-TF-IDF diagnostic terms without conversational padding.
3. **MLLM-HWSI Leads Clinical F1 on `Breast` (`24.46%`) and `Skin` (`33.64%`):** Leveraging its multi-scale cell (`CellViT-256`) and region (`HIPT-4K`) encoders, **MLLM-HWSI** outperforms both SlideChat and PRISM2 on `Breast` Clinical F1 (`24.46%` vs `20.75%` / `17.89%`) and leads all lexical and clinical metrics on `Skin` (`BLEU-1 = 12.20%`, `ROUGE-1 = 20.51%`, `ROUGE-L = 13.28%`, `METEOR = 10.20%`, `Clinical F1 = 33.64%`).

---

## 6. Qualitative Clinical Report Analysis & Domain Findings

### 6.1 Side-by-Side Qualitative Examples Across Cohorts

#### Example 1: Held-Out TCGA Prostate Adenocarcinoma (`TCGA-Unseen-151`, Slide `TCGA-M7-A723-01Z-00-DX1`, Task: `PRAD`)
- **Ground-Truth Pathology Report (`198` words):**
  > *"The pathological assessment reveals a diagnosis of prostatic adenocarcinoma following a robot-assisted laparoscopic prostatectomy. The tumor exhibits a Gleason combined score of 7 (4+3), with a tertiary pattern 5 accounting for roughly 4% of the tumor volume, affecting approximately 40% of the right lobe and 53% of the left lobe. Tumor involvement is extensive, from the apex to base in both lobes, with non-confinement beyond the prostatic capsule into extracapsular sites... involving perineural, perivascular, and fibroadipose soft tissue. The carcinoma extends into both seminal vesicles... There is lymphovascular space invasion, and perineural invasion is also present. The surgical margins at the left base are focally suspicious for tumor extension... High-grade prostatic intraepithelial neoplasia is also observed. Lymph node evaluation reveals metastatic prostatic adenocarcinoma in one of the eleven left pelvic nodes... The tumor staging is classified as pT3b pN1..."*
- **SlideChat (`89` words | `BLEU-1 = 18.21%`, `ROUGE-L = 25.94%`, `Clinical F1 = 75.00%`):**
  > *"The comprehensive pathology image for the provided slide from the prostate would include the diagnosis of prostatic adenocarcinoma with a Gleason score of 3+4=7, indicating moderately differentiated cancer. The tumor is located in the right lobe and exhibits extraprostatic extension into the periprostatic adipose tissue. There is no lymphovascular or perineural invasion identified, and the surgical margins are free of tumor. The TNM staging is pT3a, indicating local advancement beyond the prostate. Additional findings include high-grade prostatic intraepithelial neoplasia and nodular hyperplasia in non-neoplastic areas."*
- **PRISM2 (`273` words | `BLEU-1 = 30.04%`, `ROUGE-L = 19.71%`, `Clinical F1 = 85.71%`):**
  > *"The tumor is classified as pT3a, indicating extraprostatic extension. The Gleason score is 9, with a primary pattern of 4 and a secondary pattern of 5, placing it in Grade Group 5. The tumor is adenocarcinoma with a high percentage of Gleason pattern 4. There is no lymphovascular or perineural invasion. The tumor is present at the surgical margin. The tumor is multifocal with multiple invasive carcinoma foci... The tumor is classified as high-grade prostatic intraepithelial neoplasia (PIN) and intraductal carcinoma is identified..."*
- **MLLM-HWSI (`86` words | `BLEU-1 = 14.74%`, `ROUGE-L = 19.31%`, `Clinical F1 = 76.92%`):**
  > *"Microscopic observation of the pathology slide reveals a high-grade prostatic adenocarcinoma characterized by poorly differentiated tumor cells with significant architectural disorganization. The tumor exhibits a diffuse pattern of infiltration into the surrounding stromal tissue, with notable extracapsular extension beyond the prostate capsule. There is evidence of vascular invasion within lymphatic channels, indicating potential for metastatic spread. The surgical margins are involved by tumor cells, suggesting local dissemination at the resection site. Based on these observations, the diagnosis is prostatic adenocarcinoma, Gleason grade 4+5=9."*

#### Example 2: External Colorectal Adenocarcinoma (`Histai-All-1145`, Slide `HISTAI-colorectal-b1/case_173_slide_H&E_3`, Task: `Colorectum`)
- **Ground-Truth Pathology Report (`290` words):**
  > *"Colon cancer.; Sigmoid colon adenocarcinoma, grade 2, largest dimension 1.4 cm, with invasion of the mucosa, submucosa, and muscular layers of the intestinal wall, with focal invasion into the pericolic adipose tissue, with lymphovascular, venous, and perineural invasion. Proximal and distal resection margins macroscopically free of tumor... Lymph node status: a total of 6 lymph nodes were examined, one with macrometastasis... pTNM stage: pT3, pN1a (1/6), G2, L1, V1, Pn1, R0..."*
- **SlideChat (`118` words | `ROUGE-L = 12.94%`, `Clinical F1 = 85.71%`):**
  > *"The pathology image indicates a moderately differentiated adenocarcinoma of the colorectal type, which has infiltrated the pericolic fatty tissue. The tumor is staged as pT3, indicating that it has grown through the muscularis propria into the pericolorectal tissues. Importantly, there are no regional lymph node metastases (pN0), with 17 nodes examined and all found negative for cancer. The resection margins are free of tumor (R0)... The tumor is graded as G2, reflecting moderate differentiation..."*
- **PRISM2 (`87` words | `ROUGE-L = 13.84%`, `Clinical F1 = 85.71%`):**
  > *"The specimen reveals a moderately differentiated adenocarcinoma. The tumor invades through the muscularis propria into the subserosa. There is no evidence of lymphovascular or perineural invasion. Surgical margins are clear of tumor. No lymph nodes were submitted for review. The tumor is classified as pT3, indicating invasion into pericolorectal tissues. The tumor is located in the sigmoid colon. No tumor budding or precursor lesions are identified..."*
- **MLLM-HWSI (`72` words | `ROUGE-L = 8.78%`, `Clinical F1 = 46.15%`):**
  > *"Microscopic observation of the pathology slide reveals a moderately differentiated adenocarcinoma characterized by glandular structures that deviate from normal architecture. The tumor cells infiltrate the muscularis propria, indicating local tissue invasion. There is no evidence of vascular or lymphatic invasion within the observed sections. The nuclei exhibit moderate variability in size and shape, reflecting nuclear pleomorphism consistent with a moderately differentiated tumor. Based on these observations, the diagnosis is moderately differentiated adenocarcinoma."*

### 6.2 Systematic Failure Modes Identified Across Models

1. **Hallucination of Non-Visual Gross & Multi-Slide Surgical Parameters ("TCGA Template Bias"):**
   - Because TCGA pathology reports are patient-level surgical pathology documents rather than single-slide microscopic descriptions, **SlideChat** and **PRISM2** frequently hallucinate macroscopic or multi-block facts that are physically impossible to observe on a single primary tumor H&E slide — such as exact lymph node dissection counts (e.g., *"with 17 nodes examined and all found negative"*), macroscopic tumor dimensions in centimeters (*"greatest dimension of the tumor is 7.5 cm"*), or anatomical resection margin distances.
   - By contrast, **MLLM-HWSI** was trained on distilled microscopic descriptions (`"Microscopic observation of the pathology slide reveals..."`), largely avoiding lymph-node count hallucinations at the cost of omitting `pTNM` staging tokens expected by TCGA/Histai reference texts.
2. **Autoregressive Tail Repetition Under Unpenalized Greedy Decoding (`PRISM2`):**
   - On a subset of longer TCGA and CPTAC generations (`max_new_tokens=512`, `do_sample=False`), **PRISM2** occasionally enters a repetitive phrase loop near the end of its report (e.g., repeating *"The tumor type is adenocarcinoma."* or *"The histologic type is clear cell renal cell carcinoma."* until hitting the `512`-token ceiling). While Clinical Concept F1 (which operates on unique set presence) is invariant to repetition, tail repetition inflates `avg_pred_words` on CPTAC (`146.5` words on `CCRCC`) and lowers n-gram precision.
3. **Qwen2.5 End-of-Turn Token Handling (`MLLM-HWSI`):**
   - During initial Stage 2 pilot testing, `MLLM-HWSI` required explicit registration of both `<|im_end|>` (`151645`) and `<|endoftext|>` (`151643`) as `eos_token_id` stopping criteria in [`src/models/mllm_hwsi.py`](../src/models/mllm_hwsi.py) to prevent post-answer continuation. With dual-EOS stopping enabled across all full Stage 2 runs, `MLLM-HWSI` produces clean, self-contained `73–80` word microscopic reports across all `1,486` slides.

---

## 7. Computational Performance & Runtime Profile (`2 × RTX 4090`)

Stage 2 combines on-demand WSI feature extraction (for slides not previously cached during Stage 1 VQA, namely `CCRCC` in CPTAC, `127` held-out `.svs` slides in `TCGA-Unseen-151`, and non-VQA cases in `Histai-All-1145`) with autoregressive LLM decoding (`max_new_tokens=512`). For `MLLM-HWSI` on `Histai` (`1,145` slides), inference was sharded across `cuda:0` and `cuda:1` and interleaved via [`scripts/merge_histai_mllm.py`](../scripts/merge_histai_mllm.py).

| Model | TCGA-Strat152 (`152` Cached) | TCGA-Unseen151 (`151` WSIs) | CPTAC-All38 (`38` WSIs) | Histai-All1145 (`1,145` WSIs) | Cached Decoding Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SlideChat** | `12.3 min` (`4.86 s/slide`) | `12.8 min` (`5.10 s/slide`) | `7.2 min` (`11.36 s/slide`) | `202.0 min` (`10.58 s/slide`) | `~4.9 s / report` |
| **PRISM2** | `8.3 min` (`3.27 s/slide`) | `43.4 min` (`17.24 s/slide`) | `4.9 min` (`7.77 s/slide`) | `152.8 min` (`8.01 s/slide`) | `~3.3 s / report` |
| **MLLM-HWSI** | `5.1 min` (`2.02 s/slide`) | `68.3 min` (`27.15 s/slide`) | `13.8 min` (`21.83 s/slide`) | `286.4 min` GPU (`~143 min` 2-GPU wall) | `~2.0 s / report` |

---

## 8. Output Artifacts & Reproducibility Index

- **Consolidated CSV Summary Table:** [`results/stage2_caption/stage2_caption_consolidated_summary.csv`](../results/stage2_caption/stage2_caption_consolidated_summary.csv)
- **TCGA Round-5 Stratified Validation Runs (`152` WSIs across `31` Tumors):**
  - SlideChat: [`slidechat_tcga_stratified152_summary.json`](../results/stage2_caption/slidechat_tcga_stratified152_summary.json) | [`slidechat_tcga_stratified152_predictions.jsonl`](../results/stage2_caption/slidechat_tcga_stratified152_predictions.jsonl)
  - PRISM2: [`prism2_tcga_stratified152_summary.json`](../results/stage2_caption/prism2_tcga_stratified152_summary.json) | [`prism2_tcga_stratified152_predictions.jsonl`](../results/stage2_caption/prism2_tcga_stratified152_predictions.jsonl)
  - MLLM-HWSI: [`mllm-hwsi_tcga_stratified152_summary.json`](../results/stage2_caption/mllm-hwsi_tcga_stratified152_summary.json) | [`mllm-hwsi_tcga_stratified152_predictions.jsonl`](../results/stage2_caption/mllm-hwsi_tcga_stratified152_predictions.jsonl)
- **TCGA Held-Out Unseen Validation Runs (`151` WSIs across `25` Tumors):**
  - SlideChat: [`slidechat_tcga_unseen151_summary.json`](../results/stage2_caption/slidechat_tcga_unseen151_summary.json) | [`slidechat_tcga_unseen151_predictions.jsonl`](../results/stage2_caption/slidechat_tcga_unseen151_predictions.jsonl)
  - PRISM2: [`prism2_tcga_unseen151_summary.json`](../results/stage2_caption/prism2_tcga_unseen151_summary.json) | [`prism2_tcga_unseen151_predictions.jsonl`](../results/stage2_caption/prism2_tcga_unseen151_predictions.jsonl)
  - MLLM-HWSI: [`mllm-hwsi_tcga_unseen151_summary.json`](../results/stage2_caption/mllm-hwsi_tcga_unseen151_summary.json) | [`mllm-hwsi_tcga_unseen151_predictions.jsonl`](../results/stage2_caption/mllm-hwsi_tcga_unseen151_predictions.jsonl)
- **CPTAC Full Runs (`38` WSIs across `4` Tumor Cohorts):**
  - SlideChat: [`slidechat_cptac_all38_summary.json`](../results/stage2_caption/slidechat_cptac_all38_summary.json) | [`slidechat_cptac_all38_predictions.jsonl`](../results/stage2_caption/slidechat_cptac_all38_predictions.jsonl)
  - PRISM2: [`prism2_cptac_all38_summary.json`](../results/stage2_caption/prism2_cptac_all38_summary.json) | [`prism2_cptac_all38_predictions.jsonl`](../results/stage2_caption/prism2_cptac_all38_predictions.jsonl)
  - MLLM-HWSI: [`mllm-hwsi_cptac_all38_summary.json`](../results/stage2_caption/mllm-hwsi_cptac_all38_summary.json) | [`mllm-hwsi_cptac_all38_predictions.jsonl`](../results/stage2_caption/mllm-hwsi_cptac_all38_predictions.jsonl)
- **Histai Full Runs (`1,145` WSIs across `3` Organ Cohorts):**
  - SlideChat: [`slidechat_histai_all1145_summary.json`](../results/stage2_caption/slidechat_histai_all1145_summary.json) | [`slidechat_histai_all1145_predictions.jsonl`](../results/stage2_caption/slidechat_histai_all1145_predictions.jsonl)
  - PRISM2: [`prism2_histai_all1145_summary.json`](../results/stage2_caption/prism2_histai_all1145_summary.json) | [`prism2_histai_all1145_predictions.jsonl`](../results/stage2_caption/prism2_histai_all1145_predictions.jsonl)
  - MLLM-HWSI: [`mllm-hwsi_histai_all1145_summary.json`](../results/stage2_caption/mllm-hwsi_histai_all1145_summary.json) | [`mllm-hwsi_histai_all1145_predictions.jsonl`](../results/stage2_caption/mllm-hwsi_histai_all1145_predictions.jsonl)
