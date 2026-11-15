"""SlideBench Stage 2 Open-Ended Caption & Pathology Report Generation dataset loader."""

import ast
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
SUPPORTED_CAPTION_COHORTS: Tuple[str, ...] = ("CPTAC", "Histai", "TCGA")
DEFAULT_CAPTION_PROMPT: str = (
    "Please generate a comprehensive and professional pathology report "
    "based on the provided histopathological whole-slide image."
)


@dataclass(frozen=True)
class CaptionSample:
    """Standardized representation of a SlideBench open-ended caption / report sample."""

    sample_uid: str
    cohort: str
    row_id: str
    case_id: str
    slide_id: str
    task: str  # Tumor type (CPTAC/TCGA) or Organ (Histai)
    question: str  # Author-provided pathology report prompt
    reference_caption: str  # Ground-truth pathology report / caption ("Answer" column)
    all_slide_ids: Tuple[str, ...] = ()
    feature_path: Optional[Path] = None
    wsi_path: Optional[Path] = None

    @property
    def answer_text(self) -> str:
        """Return the ground-truth pathology report text (alias for `reference_caption`)."""
        return self.reference_caption

    def format_caption_prompt(self) -> str:
        """Return the pathology report generation prompt for this sample.

        Returns:
            Prompt string from the SlideBench CSV `Question` column, or a standard fallback.
        """
        cleaned = self.question.strip()
        return cleaned if cleaned else DEFAULT_CAPTION_PROMPT


def _parse_slide_list(raw_slide: str) -> Tuple[str, ...]:
    """Parse a raw `Slide` column value into a tuple of slide identifiers.

    Handles both plain strings (e.g., `'C3L-00004-21'`, `'TCGA-PA-A5YG-01Z-00-DX1'`)
    and bracketed Python list strings used in Histai (e.g., `"['HISTAI-breast/case_0003_slide_H&E_1']"`).

    Args:
        raw_slide: Raw string from the CSV `Slide` column.

    Returns:
        Non-empty tuple of stripped slide identifier strings.
    """
    cleaned = raw_slide.strip()
    if cleaned.startswith("["):
        try:
            parsed = ast.literal_eval(cleaned)
            if isinstance(parsed, (list, tuple)):
                slides = tuple(str(s).strip() for s in parsed if str(s).strip())
                if slides:
                    return slides
        except (ValueError, SyntaxError):
            pass
    return (cleaned,) if cleaned else ()


def _resolve_histai_paths(
    cohort_dir: Path, candidate_slides: Tuple[str, ...]
) -> Tuple[str, Optional[Path], Optional[Path]]:
    """Resolve primary `slide_id`, `feature_path`, and `wsi_path` for a Histai sample.

    Checks all slides listed for the case in order and selects the first slide that
    has cached features or a local `.tiff` WSI on disk. Defaults to the first listed
    slide when running in annotation-only / dry-run environments.

    Args:
        cohort_dir: Path to `.../SlideBench/Histai`.
        candidate_slides: Tuple of slide IDs belonging to the case.

    Returns:
        Tuple of `(selected_slide_id, feature_path, wsi_path)`.
    """
    if not candidate_slides:
        return "", None, None

    for cand_slide in candidate_slides:
        feat_path: Optional[Path] = None
        wsi_path: Optional[Path] = None

        cand_feat_pt = (
            cohort_dir / "Feat" / "SlideChat_CONCH" / f"{cand_slide.replace('/', '_')}.pt"
        )
        if cand_feat_pt.exists():
            feat_path = cand_feat_pt

        if "/" in cand_slide and "_slide_" in cand_slide:
            repo_short, slide_str = cand_slide.split("/", 1)
            case_folder, slide_suffix = slide_str.split("_slide_", 1)
            cand_wsi = (
                cohort_dir / "WSI" / repo_short / case_folder / f"slide_{slide_suffix}.tiff"
            )
            if cand_wsi.exists():
                wsi_path = cand_wsi

        if feat_path is not None or wsi_path is not None:
            return cand_slide, feat_path, wsi_path

    return candidate_slides[0], None, None


def resolve_caption_csv_path(datasets_dir: Path, cohort: str) -> Path:
    """Resolve the path to `SlideBench-Caption-<cohort>.csv`.

    Searches in order:
    1. External dataset directory: `<datasets_dir>/<cohort>/SlideBench-Caption-<cohort>.csv`
    2. Local git-ignored annotations fallback: `<project_root>/results/annotations/<cohort>/SlideBench-Caption-<cohort>.csv`
    3. Direct directory fallback: `<datasets_dir>/SlideBench-Caption-<cohort>.csv`

    Args:
        datasets_dir: Base SlideBench directory (`../Datasets/SlideBench`).
        cohort: Cohort name (`'CPTAC'`, `'Histai'`, or `'TCGA'`).

    Returns:
        Resolved `Path` to the existing CSV file.

    Raises:
        ValueError: If `cohort` is not a supported SlideBench captioning cohort (e.g., `'BCNB'`).
        FileNotFoundError: If the CSV cannot be found in any candidate location.
    """
    if cohort not in SUPPORTED_CAPTION_COHORTS:
        raise ValueError(
            f"Cohort '{cohort}' is not supported for Stage 2 Captioning. "
            f"Supported cohorts are: {SUPPORTED_CAPTION_COHORTS} "
            "(Note: BCNB only has a closed-ended VQA benchmark in SlideBench)."
        )

    csv_filename = f"SlideBench-Caption-{cohort}.csv"
    candidates = [
        datasets_dir / cohort / csv_filename,
        PROJECT_ROOT / "results" / "annotations" / cohort / csv_filename,
        datasets_dir / csv_filename,
    ]
    for cand in candidates:
        if cand.is_file():
            return cand

    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Missing caption benchmark CSV for cohort '{cohort}'. Searched: {searched}"
    )


def load_slidebench_caption_cohort(
    datasets_dir: Path,
    cohort: str,
    require_features: bool = False,
    require_wsi: bool = False,
    csv_path: Optional[Path] = None,
) -> List[CaptionSample]:
    """Load `CaptionSample` items for a SlideBench caption cohort (`CPTAC`, `Histai`, or `TCGA`).

    Args:
        datasets_dir: Base SlideBench directory (`../Datasets/SlideBench`).
        cohort: Cohort name (`'CPTAC'`, `'Histai'`, or `'TCGA'`).
        require_features: If True, filter out samples that have neither pre-extracted features nor a local WSI.
        require_wsi: If True, filter out samples without a local WSI file (`.svs` or `.tiff`).
        csv_path: Optional explicit CSV path override (e.g. for unseen/deduplicated evaluation sets).

    Returns:
        List of parsed `CaptionSample` instances.
    """
    if csv_path is None:
        csv_path = resolve_caption_csv_path(datasets_dir=datasets_dir, cohort=cohort)
    cohort_dir = datasets_dir / cohort

    samples: List[CaptionSample] = []
    with csv_path.open("r", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            row_id = str(row.get("ID", idx)).strip()
            case_id = str(
                row.get("Case_ID") or row.get("case") or row.get("ID") or idx
            ).strip()

            raw_slide = str(row.get("Slide", "")).strip()
            slide_candidates = _parse_slide_list(raw_slide)
            if not slide_candidates:
                continue

            task = str(
                row.get("Tumor") or row.get("Organ") or row.get("Task") or "General"
            ).strip()
            question = str(row.get("Question", "") or "").strip()
            reference_caption = str(row.get("Answer", "") or "").strip()

            feature_path: Optional[Path] = None
            wsi_path: Optional[Path] = None
            slide_id = slide_candidates[0]

            if cohort in ("CPTAC", "TCGA"):
                tumor = str(row.get("Tumor", "") or task).strip()
                cand_feat_csv = cohort_dir / "Feat" / tumor / f"{slide_id}.csv"
                cand_feat_tcga_csv = (
                    cohort_dir / "Feat" / "TCGA_patch_feat" / tumor / f"{slide_id}.csv"
                )
                cand_feat_pt = (
                    cohort_dir / "Feat" / "SlideChat_CONCH" / f"{slide_id}.pt"
                )
                if cand_feat_csv.exists():
                    feature_path = cand_feat_csv
                elif cand_feat_tcga_csv.exists():
                    feature_path = cand_feat_tcga_csv
                elif cand_feat_pt.exists():
                    feature_path = cand_feat_pt

                cand_wsi = cohort_dir / "WSI" / tumor / f"{slide_id}.svs"
                cand_wsi_unseen = cohort_dir / "WSI_unseen_mllm" / tumor / f"{slide_id}.svs"
                if cand_wsi.exists():
                    wsi_path = cand_wsi
                elif cand_wsi_unseen.exists():
                    wsi_path = cand_wsi_unseen
                elif (cohort_dir / "WSI_unseen_mllm" / f"{slide_id}.svs").exists():
                    wsi_path = cohort_dir / "WSI_unseen_mllm" / f"{slide_id}.svs"
                elif (cohort_dir / "WSI" / f"{slide_id}.svs").exists():
                    wsi_path = cohort_dir / "WSI" / f"{slide_id}.svs"

            elif cohort == "Histai":
                slide_id, feature_path, wsi_path = _resolve_histai_paths(
                    cohort_dir=cohort_dir,
                    candidate_slides=slide_candidates,
                )

            if require_wsi and wsi_path is None:
                continue
            if require_features and feature_path is None and wsi_path is None:
                continue

            samples.append(
                CaptionSample(
                    sample_uid=f"{cohort}_{idx}_{slide_id}",
                    cohort=cohort,
                    row_id=row_id,
                    case_id=case_id,
                    slide_id=slide_id,
                    task=task,
                    question=question,
                    reference_caption=reference_caption,
                    all_slide_ids=slide_candidates,
                    feature_path=feature_path,
                    wsi_path=wsi_path,
                )
            )

    return samples


def select_stratified_caption_pilot_samples(
    samples: Sequence[CaptionSample],
    limit: int = 20,
) -> List[CaptionSample]:
    """Select a stratified pilot subset across all organs/tumor types (`task`).

    Iterates in round-robin order over distinct `task` values (`Tumor` in CPTAC/TCGA,
    `Organ` in Histai) so every clinical subgroup is represented proportionally.

    Args:
        samples: Sequence of `CaptionSample` objects.
        limit: Maximum number of pilot samples to return (<= 0 returns all samples).

    Returns:
        Stratified list of up to `limit` `CaptionSample` instances.
    """
    if limit <= 0 or len(samples) <= limit:
        return list(samples)

    by_task: Dict[str, List[CaptionSample]] = {}
    for item in samples:
        by_task.setdefault(item.task, []).append(item)

    task_names = sorted(by_task.keys())
    selected: List[CaptionSample] = []
    used_uids = set()
    idx_per_task = {t: 0 for t in task_names}

    while len(selected) < limit:
        added_any = False
        for t in task_names:
            if len(selected) >= limit:
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

    return selected[:limit]
