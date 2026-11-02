"""SlideBench Closed-Ended VQA dataset loader (Multiple-Choice & Binary/Yes-No)."""

import ast
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class VQASample:
    """Standardized representation of a closed-ended or binary/Yes-No VQA sample."""

    sample_uid: str
    cohort: str
    row_id: str
    slide_id: str
    task: str
    question_type: str  # "binary" (2 options / Yes-No / Pos-Neg) or "multiple_choice" (3-4 options)
    question: str
    options: Dict[str, str]  # e.g., {"A": "Negative", "B": "Positive"} or {"A": "Yes", "B": "No"}
    answer: str  # Ground-truth option key ("A", "B", "C", "D") or binary label
    answer_text: str  # Ground-truth option text
    feature_path: Optional[Path] = None
    wsi_path: Optional[Path] = None

    def format_closed_prompt(self) -> str:
        """Format the question and options exactly as expected by closed-ended VQA models.

        Returns:
            Formatted multi-line question + option list string.
        """
        option_lines = [f"{key}. {val}" for key, val in self.options.items() if val]
        options_block = "\n".join(option_lines)
        return f"{self.question}\n{options_block}" if options_block else self.question


def _classify_question_type(options: Dict[str, str]) -> str:
    """Determine whether a closed-ended VQA sample is binary (Yes/No or Pos/Neg) or multiple-choice.

    Args:
        options: Non-empty option dictionary mapping letter keys to option strings.

    Returns:
        'binary_yes_no', 'binary', or 'multiple_choice'.
    """
    valid_vals = [v.strip().lower() for v in options.values() if v.strip()]
    if len(valid_vals) == 2:
        if set(valid_vals) == {"yes", "no"}:
            return "binary_yes_no"
        return "binary"
    return "multiple_choice"


def load_slidebench_vqa_cohort(
    datasets_dir: Path,
    cohort: str,
    require_features: bool = False,
) -> List[VQASample]:
    """Load VQASample items for a given SlideBench cohort (BCNB, CPTAC, or Histai).

    Args:
        datasets_dir: Base SlideBench directory (../Datasets/SlideBench).
        cohort: Cohort name ('BCNB', 'CPTAC', or 'Histai').
        require_features: If True, filter out samples without pre-extracted feature files.

    Returns:
        List of parsed VQASample instances.
    """
    cohort_dir = datasets_dir / cohort
    csv_path = cohort_dir / f"SlideBench-VQA-{cohort}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing VQA benchmark CSV: {csv_path}")

    samples: List[VQASample] = []
    with csv_path.open("r", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            row_id = str(row.get("ID", idx)).strip()
            raw_slide = str(row.get("Slide", "")).strip()
            if raw_slide.startswith("["):
                parsed_slides = ast.literal_eval(raw_slide)
                slide_id = str(parsed_slides[0]) if parsed_slides else raw_slide
            else:
                slide_id = raw_slide

            task = (
                str(row.get("Task") or row.get("Tumor") or row.get("Organ") or "General")
                .strip()
            )
            question = str(row.get("Question", "")).strip()
            options: Dict[str, str] = {}
            for opt_key in ("A", "B", "C", "D"):
                val = str(row.get(opt_key, "") or "").strip()
                if val and val.lower() != "none":
                    options[opt_key] = val

            answer = str(row.get("Answer", "")).strip().upper()
            answer_text = options.get(answer, answer)
            q_type = _classify_question_type(options)

            feature_path: Optional[Path] = None
            wsi_path: Optional[Path] = None
            if cohort == "BCNB":
                cand_feat = cohort_dir / "Feat" / "BCNB_patch_feat" / f"{slide_id}.csv"
                if cand_feat.exists():
                    feature_path = cand_feat
                cand_wsi = cohort_dir / "WSI" / f"{slide_id}.jpg"
                if cand_wsi.exists():
                    wsi_path = cand_wsi
            elif cohort == "CPTAC":
                tumor = str(row.get("Tumor", "")).strip()
                cand_feat = cohort_dir / "Feat" / tumor / f"{slide_id}.csv"
                if cand_feat.exists():
                    feature_path = cand_feat
                cand_wsi = cohort_dir / "WSI" / tumor / f"{slide_id}.svs"
                if cand_wsi.exists():
                    wsi_path = cand_wsi
            elif cohort == "Histai":
                if "/" in slide_id and "_slide_" in slide_id:
                    repo_short, slide_str = slide_id.split("/", 1)
                    case_id, slide_suffix = slide_str.split("_slide_", 1)
                    cand_wsi = (
                        cohort_dir / "WSI" / repo_short / case_id / f"slide_{slide_suffix}.tiff"
                    )
                    if cand_wsi.exists():
                        wsi_path = cand_wsi

            if require_features and feature_path is None:
                continue

            samples.append(
                VQASample(
                    sample_uid=f"{cohort}_{row_id}_{slide_id}",
                    cohort=cohort,
                    row_id=row_id,
                    slide_id=slide_id,
                    task=task,
                    question_type=q_type,
                    question=question,
                    options=options,
                    answer=answer,
                    answer_text=answer_text,
                    feature_path=feature_path,
                    wsi_path=wsi_path,
                )
            )

    return samples


def convert_binary_to_yes_no_sample(sample: VQASample) -> VQASample:
    """Convert a 2-option binary receptor status sample (e.g., ER/PR/HER2 Positive vs Negative)
    into an explicit Yes/No closed-ended question to test Yes/No VQA capability alongside A/B.

    Args:
        sample: Original binary VQASample with Positive/Negative options.

    Returns:
        New VQASample phrased as a Yes/No closed-ended question with A=Yes, B=No.
    """
    gt_is_positive = sample.answer_text.strip().lower() == "positive"
    yn_answer = "A" if gt_is_positive else "B"
    yn_text = "Yes" if gt_is_positive else "No"
    yn_question = (
        f"Based on the histopathological whole-slide image of breast cancer, "
        f"is the {sample.task} status positive? Please answer Yes or No."
    )
    return VQASample(
        sample_uid=f"{sample.sample_uid}_YN",
        cohort=sample.cohort,
        row_id=f"{sample.row_id}_YN",
        slide_id=sample.slide_id,
        task=f"{sample.task} (Yes/No)",
        question_type="binary_yes_no",
        question=yn_question,
        options={"A": "Yes", "B": "No"},
        answer=yn_answer,
        answer_text=yn_text,
        feature_path=sample.feature_path,
        wsi_path=sample.wsi_path,
    )


def select_stratified_pilot_samples(
    samples: Sequence[VQASample],
    limit: int = 20,
    include_yes_no_variants: bool = True,
) -> List[VQASample]:
    """Select a balanced pilot subset covering all clinical tasks, multiple-choice, and binary/Yes-No questions.

    Args:
        samples: Full list of VQASample objects.
        limit: Total number of samples to return (e.g. 20).
        include_yes_no_variants: If True and dataset has binary receptor tasks (ER/PR/HER2),
            include explicit Yes/No variants for a subset of binary questions.

    Returns:
        Balanced list of up to `limit` VQASample instances across distinct slides and tasks.
    """
    if limit <= 0 or len(samples) <= limit:
        return list(samples)

    by_task: Dict[str, List[VQASample]] = {}
    for item in samples:
        by_task.setdefault(item.task, []).append(item)

    task_names = list(by_task.keys())
    selected: List[VQASample] = []
    used_uids = set()

    # Reserve 4 slots for explicit Yes/No binary questions if requested
    yn_quota = min(4, limit // 4) if include_yes_no_variants else 0
    mc_and_bin_quota = limit - yn_quota

    idx_per_task = {t: 0 for t in task_names}
    while len(selected) < mc_and_bin_quota:
        added_any = False
        for t in task_names:
            if len(selected) >= mc_and_bin_quota:
                break
            task_list = by_task[t]
            while idx_per_task[t] < len(task_list):
                cand = task_list[idx_per_task[t]]
                idx_per_task[t] += 1
                if cand.sample_uid not in used_uids:
                    selected.append(cand)
                    used_uids.add(cand.sample_uid)
                    added_any = True
                    break
        if not added_any:
            break

    if yn_quota > 0:
        binary_candidates = [
            s for s in samples if s.question_type == "binary" and s.task in ("ER", "PR", "HER2")
        ]
        step = max(1, len(binary_candidates) // (yn_quota + 1))
        for i in range(yn_quota):
            if i * step < len(binary_candidates):
                yn_sample = convert_binary_to_yes_no_sample(binary_candidates[i * step])
                selected.append(yn_sample)

    return selected[:limit]


def load_patch_features_csv(feature_csv_path: Path, max_patches: int = 38400) -> torch.Tensor:
    """Load 512-dimensional patch features from a SlideChat CSV file.

    Args:
        feature_csv_path: Path to the slide's `.csv` patch feature file.
        max_patches: Maximum number of patches to sample linearly (default: 38400).

    Returns:
        Float tensor of shape (1, num_patches, 512).
    """
    df = pd.read_csv(feature_csv_path)
    df = df.iloc[:, :512]
    total_rows = df.shape[0]
    if total_rows >= max_patches:
        indices = np.linspace(0, total_rows - 1, max_patches, dtype=int)
        df = df.iloc[indices].iloc[:max_patches]
    arr = df.to_numpy(dtype=np.float32).reshape(1, df.shape[0], 512)
    return torch.from_numpy(arr)
