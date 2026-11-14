"""SlideBench dataset loaders, feature resolvers, and VQA/Caption prompt builders."""

from src.datasets.slidebench_caption import (
    SUPPORTED_CAPTION_COHORTS,
    CaptionSample,
    load_slidebench_caption_cohort,
    resolve_caption_csv_path,
    select_stratified_caption_pilot_samples,
)
from src.datasets.slidebench_vqa import (
    VQASample,
    load_patch_features_csv,
    load_slidebench_vqa_cohort,
    select_stratified_pilot_samples,
)

__all__ = [
    "CaptionSample",
    "SUPPORTED_CAPTION_COHORTS",
    "VQASample",
    "load_patch_features_csv",
    "load_slidebench_caption_cohort",
    "load_slidebench_vqa_cohort",
    "resolve_caption_csv_path",
    "select_stratified_caption_pilot_samples",
    "select_stratified_pilot_samples",
]

