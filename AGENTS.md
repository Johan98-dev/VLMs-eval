# AGENTS.md — AI Coding Assistant Context & Guidelines

## 1. Project Overview

**VLMs-eval** is a modular, research-grade evaluation framework for benchmarking Whole-Slide Image (WSI) Vision-Language Models (VLMs) and Multimodal Large Language Models (MLLMs) in computational pathology.

### Evaluation Roadmap
- **Stage 1 — Closed-Ended VQA (Current Focus):**
  - Evaluate models on multiple-choice and binary (`Yes/No`) visual question answering benchmarks.
  - Note: `Yes/No` questions are explicitly grouped into and evaluated as part of the closed-ended VQA pipeline.
- **Stage 2 — Slide Captioning & Report Generation (Upcoming):**
  - Evaluate models on open-ended pathology report and caption generation (`SlideBench-Caption-*`).
- **Future Stages:**
  - Additional tasks and cohorts to be defined after Stage 1 and Stage 2 are complete.

---

## 2. Target Models (Initial Phase)

The framework currently targets three pathology WSI multimodal models through unified model adapters:
1. **SlideChat** (`General-Medical-AI/SlideChat`)
2. **PRISM2**
3. **MLLM-HWSI**

Each model must be encapsulated behind a clean, standardized inference interface so that dataset loading, prompt formatting, output parsing, and metric computation remain model-agnostic.

---

## 3. Datasets & External Storage Layout

> **CRITICAL RULE:** Raw datasets, Whole-Slide Images (WSIs), extracted patch/slide features, Hugging Face caches, and model weights **MUST NEVER** be stored inside the repository root. They reside **one directory level above** the project root (`../`) to prevent repository bloat and accidental commits.

### External Directory Structure (`../`)
```text
../
├── Datasets/
│   └── SlideBench/
│       ├── .hf_cache/                       # Hugging Face download cache
│       ├── BCNB/
│       │   ├── SlideBench-VQA-BCNB.csv      # 7,274 VQA samples (1,058 slides) [READY]
│       │   ├── Feat/
│       │   │   └── BCNB_patch_feat.tar.gz   # Pre-extracted patch features [READY]
│       │   └── WSI/                         # 1,058 .jpg slides + .json [READY]
│       ├── CPTAC/
│       │   ├── SlideBench-Caption-CPTAC.csv # 38 caption samples (Stage 2) [READY]
│       │   ├── SlideBench-VQA-CPTAC.csv     # 240 VQA samples [READY]
│       │   └── WSI/                         # 270 .svs slides by cohort [READY]
│       └── Histai/
│           ├── SlideBench-Caption-Histai.csv# 1,145 caption samples (Stage 2) [CSV READY]
│           ├── SlideBench-VQA-Histai.csv    # 769 VQA samples [CSV READY]
│           └── WSI/                         # 2,217 .tiff slides [PENDING GATED HF ACCESS]
├── Models/                                  # Model checkpoints & weights (SlideChat, PRISM2, MLLM-HWSI)
└── Thesis/ (or repo root "VLMs-eval")                   # This Git repository
```

### Dataset Status Notes
- **BCNB & CPTAC:** Fully downloaded and verified on the primary workstation/server linked to this repository.
- **Histai:** CSV annotations are downloaded; `.tiff` WSIs are pending Hugging Face gated repository access approval (`histai/HISTAI-*`). Code must gracefully handle missing cohorts or allow cohort filtering (e.g., `--cohorts BCNB CPTAC`).

---

## 4. Software Architecture & Code Quality Standards

To keep this codebase maintainable, reproducible, and publication-ready, all agents must strictly follow these engineering principles:

1. **No Spaghetti Code:**
   - Avoid monolithic scripts that mix data I/O, model loading, prompt engineering, and metric calculation.
   - Enforce strict separation of concerns with small, single-responsibility modules and functions.

2. **Recommended Repository Structure:**
   ```text
   .
   ├── AGENTS.md                  # AI agent context and engineering rules
   ├── README.md                  # User-facing documentation
   ├── download_slidebench.py     # Dataset & WSI acquisition CLI
   ├── requirements.txt           # Project dependencies
   ├── configs/                   # YAML/JSON experiment, model, and path configurations
   ├── src/
   │   ├── datasets/              # SlideBench CSV parsers, WSI/feature loaders, prompt builders
   │   ├── models/                # Base VLM interface + SlideChat, PRISM2, MLLM-HWSI adapters
   │   ├── evaluation/            # Closed-ended VQA answer extraction & metrics (Stage 1)
   │   └── utils/                 # Logging, reproducibility, I/O, and path resolution helpers
   ├── scripts/                   # Clean CLI entrypoints (e.g., run_vqa_eval.py, extract_features.py)
   └── results/                   # Lightweight JSON/CSV predictions & aggregated metrics
   ```

3. **Type Safety & Documentation:**
   - Use Python type hints (`typing` / `pathlib.Path` / `dataclasses` or `pydantic`) across all function signatures.
   - Include concise Google-style or NumPy-style docstrings for classes and public functions.
   - Use `pathlib.Path` instead of hardcoded string path concatenation for cross-platform compatibility (Windows development environment + Linux GPU execution environment).

4. **Reproducibility & Fault Tolerance:**
   - Long-running inference or feature extraction jobs must support **checkpointing/resuming** (e.g., saving per-sample predictions incrementally to `.jsonl` or `.csv`) so interrupted runs do not lose progress.
   - Always log model configuration, prompt templates, and random seeds alongside evaluation outputs.

---

## 5. Language & Git Conventions

- **English Only for Code & Docs:**
  - Even if the user communicates in Spanish during chat sessions, **all code, variable/function names, comments, docstrings, README/markdown documentation, CLI help messages, and Git commit messages MUST be written in English**.
- **Commit Messages:**
  - Follow **Conventional Commits** in English (e.g., `feat(vqa): add closed-ended VQA evaluator for SlideBench`, `refactor(models): extract base VLM interface`, `docs: add AGENTS.md context file`).
- **Environment & Secrets:**
  - Never hardcode credentials, tokens (`HF_TOKEN`), or machine-specific absolute paths. Load secrets via `.env` (`python-dotenv`) and resolve external directories relative to `PROJECT_ROOT.parent` or via environment variables/CLI flags.
