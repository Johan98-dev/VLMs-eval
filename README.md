# VLMs-eval

Evaluation toolkit and dataset management utilities for Vision-Language Models (VLMs) in computational pathology.

## SlideBench Non-TCGA Dataset Setup

This repository provides [`download_slidebench.py`](./download_slidebench.py) to download, extract, and organize the non-TCGA evaluation benchmarks (`BCNB`, `CPTAC`, and `Histai`) from [**General-Medical-AI/SlideChat**](https://huggingface.co/datasets/General-Medical-AI/SlideChat) along with their corresponding Whole-Slide Images (WSIs).

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
└── Histai/
    ├── SlideBench-Caption-Histai.csv    # Caption/Report benchmark (1,145 samples: Breast, Colorectum, Skin)
    ├── SlideBench-VQA-Histai.csv        # VQA benchmark (769 samples: Breast, Colorectum, Skin)
    └── WSI/                             # 2,217 unique .tiff slides organized by sub-cohort and case
        ├── HISTAI-breast/
        ├── HISTAI-colorectal-b1/
        ├── HISTAI-colorectal-b2/
        └── HISTAI-skin-b1/
```

### Data Sources

| Cohort | Annotations & Features | WSI Source | Format & Volume |
| :--- | :--- | :--- | :--- |
| **BCNB** | `General-Medical-AI/SlideChat` | [`Anonymous192234/BCNB`](https://huggingface.co/datasets/Anonymous192234/BCNB) (18 `.zip` archives) | `1,058` `.jpg` WSIs + `.json` (`~37.1 GB`) |
| **CPTAC** | `General-Medical-AI/SlideChat` | [TCIA PathDB Server](https://pathdb.cancerimagingarchive.net/) | `270` `.svs` WSIs (`~98.4 GB`) |
| **Histai** | `General-Medical-AI/SlideChat` | Official [`histai`](https://huggingface.co/histai) Hugging Face repositories | `2,217` `.tiff` WSIs (`~420–520 GB`) |

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

# Phase 6: Verify downloaded CSVs, features, and WSI file counts
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
