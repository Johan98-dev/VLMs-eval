# VLMs-eval

Evaluation toolkit and dataset management utilities for Vision-Language Models (VLMs) in computational pathology.

## SlideBench Dataset Setup

This repository provides [`download_slidebench.py`](./download_slidebench.py) to download, extract, and organize the SlideBench evaluation benchmarks (`BCNB`, `CPTAC`, `Histai`, and `TCGA`) from [**General-Medical-AI/SlideChat**](https://huggingface.co/datasets/General-Medical-AI/SlideChat) along with their corresponding Whole-Slide Images (WSIs).

### Target Directory Structure

By default, all dataset files are organized under `../Datasets/SlideBench`:

```text
../Datasets/SlideBench/
├── BCNB/
│   ├── SlideBench-VQA-BCNB.csv          # Closed-ended VQA benchmark (7,274 samples, 1,058 slides)
│   ├── Feat/
│   │   └── BCNB_patch_feat.tar.gz       # Pre-extracted patch features from SlideChat
│   └── WSI/                             # 1,058 WSIs (.jpg) and pathology annotations (.json)
├── CPTAC/
│   ├── SlideBench-Caption-CPTAC.csv     # Caption/Report benchmark (38 samples: CCRCC, LUAD, LSCC, UCEC)
│   ├── SlideBench-VQA-CPTAC.csv         # VQA benchmark (240 samples: CM, LUAD, LSCC, UCEC)
│   └── WSI/                             # 270 unique .svs slides organized by tumor cohort
│       ├── CCRCC/
│       ├── CM/
│       ├── LSCC/
│       ├── LUAD/
│       └── UCEC/
├── Histai/
│   ├── SlideBench-Caption-Histai.csv    # Caption/Report benchmark (1,145 samples: Breast, Colorectum, Skin)
│   ├── SlideBench-VQA-Histai.csv        # VQA benchmark (769 samples: Breast, Colorectum, Skin)
│   └── WSI/                             # 2,217 unique .tiff slides organized by sub-cohort and case
│       ├── HISTAI-breast/
│       ├── HISTAI-colorectal-b1/
│       ├── HISTAI-colorectal-b2/
│       └── HISTAI-skin-b1/
└── TCGA/
    ├── SlideBench-Caption-TCGA.csv      # Caption/Report benchmark (1,555 samples, 1,547 slides)
    ├── SlideBench-VQA-TCGA.csv          # VQA benchmark (3,176 samples, 1,292 slides across 31 tumors)
    ├── Feat/
    │   └── TCGA_patch_feat/             # 1,292 VQA (1,547 total) CONCH patch feature CSVs (~91 GB) [READY]
    └── WSI/                             # 152 Round-5 stratified .svs WSIs across 31 tumors (~147 GB) [READY]
```

### Data Sources

| Cohort | Annotations & Features | WSI Source | Format & Volume |
| :--- | :--- | :--- | :--- |
| **BCNB** | `General-Medical-AI/SlideChat` | [`Anonymous192234/BCNB`](https://huggingface.co/datasets/Anonymous192234/BCNB) (18 `.zip` archives) | `1,058` `.jpg` WSIs + `.json` (`~37.1 GB`) |
| **CPTAC** | `General-Medical-AI/SlideChat` | [TCIA PathDB Server](https://pathdb.cancerimagingarchive.net/) | `270` `.svs` WSIs (`~98.4 GB`) |
| **Histai** | `General-Medical-AI/SlideChat` | Official [`histai`](https://huggingface.co/histai) Hugging Face repositories | `2,217` `.tiff` WSIs (`~420–520 GB`) |
| **TCGA** | `General-Medical-AI/SlideChat` (`1,547` CSVs, `~91 GB`) | [NCI GDC Data Portal API](https://api.gdc.cancer.gov/) | `152` Round-5 stratified `.svs` WSIs (`~147 GB`, `345` VQA samples across 31 tumors) |

---

## Installation & Configuration

1. **Activate the virtual environment and install dependencies:**
   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure Hugging Face credentials in `.env`:**
   Create a `.env` file in the project root containing your Hugging Face token:
   ```env
   HF_TOKEN=hf_your_access_token_here
   ```

3. **Gated Repository Access for `Histai` WSIs:**
   The 4 official `histai` WSI repositories on Hugging Face require manual access approval. Before running `--phase histai`, request access on the following repositories and ensure your `HF_TOKEN` has **"Read access to contents of all public gated repos you can access"** enabled:
   - [`histai/HISTAI-breast`](https://huggingface.co/datasets/histai/HISTAI-breast)
   - [`histai/HISTAI-skin-b1`](https://huggingface.co/datasets/histai/HISTAI-skin-b1)
   - [`histai/HISTAI-colorectal-b1`](https://huggingface.co/datasets/histai/HISTAI-colorectal-b1)
   - [`histai/HISTAI-colorectal-b2`](https://huggingface.co/datasets/histai/HISTAI-colorectal-b2)

---

## Usage

Run `download_slidebench.py` by specifying the desired execution phase:

```bash
# Phase 2: Download all 5 non-TCGA CSV benchmarks and BCNB patch features
.venv/bin/python download_slidebench.py --phase annotations

# Phase 3: Download all 270 CPTAC .svs WSIs from TCIA PathDB
.venv/bin/python download_slidebench.py --phase cptac --workers 4

# Phase 4: Download all 2,217 Histai .tiff WSIs from Hugging Face
.venv/bin/python download_slidebench.py --phase histai --workers 4

# Phase 5: Download and extract all 18 BCNB WSI .zip parts
.venv/bin/python download_slidebench.py --phase bcnb

# Phase 6: Stream-extract SlideBench TCGA CONCH patch features from Hugging Face
.venv/bin/python download_slidebench.py --phase tcga-feat --workers 4

# Phase 7: Download TCGA .svs WSIs from NCI GDC API in stratified round-robin tumor order
.venv/bin/python download_slidebench.py --phase tcga --workers 6

# Verify downloaded CSVs, features, and WSI file counts
.venv/bin/python download_slidebench.py --phase verify
```

### Detached Background Execution (`nohup`)

For long-running WSI downloads (such as `histai`), you can run the script detached from your terminal session:

```bash
nohup .venv/bin/python download_slidebench.py --phase histai --workers 4 > download_histai.log 2>&1 &
```

Monitor progress at any time with:

```bash
tail -f download_histai.log
.venv/bin/python download_slidebench.py --phase verify
```

---

## Model Checkpoints & Weights Setup

This repository provides [`download_models.py`](./download_models.py) to download and verify the target WSI Vision-Language Models (`SlideChat`, `PRISM2`, `MLLM-HWSI`), their base instruction-tuned LLM (`Qwen2.5-7B-Instruct`), and optional patch/tile foundation encoders (`Virchow2`, `CONCH`).

### Target External Models Structure (`../Models`)

All model weights and Hugging Face cache files are stored outside the repository under `../Models`:

```text
../Models/
├── .hf_cache/                       # Hugging Face download cache
├── PRISM2/                          # paige-ai/Prism2 (~16.66 GB)
├── MLLM-HWSI/                       # Bastech/MLLM-HWSI (~15.31 GB)
├── Qwen2.5-7B-Instruct/             # Qwen/Qwen2.5-7B-Instruct (~14.20 GB, base LLM)
├── SlideChat/                       # General-Medical-AI/SlideChat_Weight (~14.58 GB model states + ~85.19 GB optim states)
│   ├── stage1_pth/
│   └── stage2_pth/
└── Encoders/                        # Patch/Tile/Region/Cell foundation encoders
    ├── Virchow2/                    # paige-ai/Virchow2 (~4.70 GB, required by PRISM2)
    ├── CONCH/                       # MahmoodLab/CONCH (~0.75 GB, required by SlideChat & MLLM-HWSI)
    ├── HIPT/                        # vit_256_small_dino.pth & vit_4096_xs_dino.pth (required by MLLM-HWSI)
    └── CellViT/                     # CellViT-256-x40-AMP.pth (required by MLLM-HWSI)
```

### Model Repositories & Access Status

| Component | Source Repository | Size | Access Notes |
| :--- | :--- | :--- | :--- |
| **PRISM2** | [`paige-ai/Prism2`](https://huggingface.co/paige-ai/Prism2) | `~16.66 GB` | Gated HF (Approved) |
| **MLLM-HWSI** | [`Bastech/MLLM-HWSI`](https://huggingface.co/Bastech/MLLM-HWSI) | `~15.31 GB` | Public HF |
| **Qwen2.5-7B-Instruct** | [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | `~14.20 GB` | Public HF (Base LLM for SlideChat & MLLM-HWSI) |
| **SlideChat** | [`General-Medical-AI/SlideChat_Weight`](https://huggingface.co/General-Medical-AI/SlideChat_Weight) | `~99.77 GB` | Gated HF (Approved) |
| **Virchow2** *(Encoder)* | [`paige-ai/Virchow2`](https://huggingface.co/paige-ai/Virchow2) | `~4.70 GB` | Gated HF (Approved — PRISM2 tile encoder) |
| **CONCH** *(Encoder)* | [`MahmoodLab/CONCH`](https://huggingface.co/MahmoodLab/CONCH) | `~0.75 GB` | Gated HF (Approved — SlideChat & MLLM-HWSI patch encoder) |
| **HIPT** *(Encoder)* | [Google Drive (`mahmoodlab/HIPT`)](https://drive.google.com/drive/folders/1dzOOKTHMPbDh59zPEkOZX5ss6A6Vj9R1?usp=sharing) | `~1.02 GB` | Public Google Drive (MLLM-HWSI 256/4096 region/patch ViTs) |
| **CellViT** *(Encoder)* | [Google Drive (`TIO-IKIM/CellViT`)](https://drive.google.com/drive/folders/1dEqwN5NaQ-8oIcc8zDYQo_Q_6VDMS_QU?usp=sharing) | `~0.17 GB` | Public Google Drive (MLLM-HWSI cell-level encoder) |

### Usage

```bash
# Verify current download status of all models and encoders
.venv/bin/python download_models.py --models verify

# Download core model checkpoints first (skipping 87 GB DeepSpeed optimizer states)
.venv/bin/python download_models.py --models prism2 mllm-hwsi qwen2.5-7b slidechat --skip-optim-states

# Download all model files (including SlideChat DeepSpeed optimizer states)
.venv/bin/python download_models.py --models all

# Download patch/tile encoders once gated access is approved
.venv/bin/python download_models.py --models encoders
```

---

## Stage 1 — Closed-Ended & Binary/Yes-No VQA Evaluation

Run [`scripts/run_vqa_eval.py`](./scripts/run_vqa_eval.py) to evaluate `SlideChat`, `PRISM2`, and `MLLM-HWSI` on multiple-choice (`A/B/C/D`), binary (`Positive/Negative`), and `Yes/No` visual question answering benchmarks:

```bash
# 20-sample stratified pilot on BCNB across all 3 models
.venv/bin/python scripts/run_vqa_eval.py --model slidechat --cohorts BCNB --limit 20
.venv/bin/python scripts/run_vqa_eval.py --model prism2    --cohorts BCNB --limit 20
.venv/bin/python scripts/run_vqa_eval.py --model mllm-hwsi --cohorts BCNB --limit 20

# Full cohort evaluation (with incremental JSONL checkpointing/resume)
.venv/bin/python scripts/run_vqa_eval.py --model prism2 --cohorts BCNB CPTAC --limit 0
```


