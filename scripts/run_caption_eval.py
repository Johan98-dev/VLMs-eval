#!/usr/bin/env python3
"""Stage 2 Open-Ended WSI Captioning & Pathology Report Generation Evaluation CLI Entrypoint."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, List, Sequence, TypeVar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - lightweight fallback when tqdm is not installed
    T = TypeVar("T")

    def tqdm(iterable: Iterable[T], **kwargs: Any) -> Iterable[T]:  # type: ignore[misc]
        return iterable


from src.datasets.slidebench_caption import (
    SUPPORTED_CAPTION_COHORTS,
    CaptionSample,
    load_slidebench_caption_cohort,
    select_stratified_caption_pilot_samples,
)
from src.datasets.slidebench_vqa import VQASample
from src.evaluation.caption_metrics import (
    compute_caption_metrics,
    compute_sample_lexical_metrics,
)
from src.models.base import BaseWSIVLM
from src.utils.io import (
    append_jsonl_record,
    load_completed_sample_ids,
    load_jsonl_records,
    set_global_seed,
    write_json_file,
)
from src.utils.paths import ProjectPaths


class DryRunCaptionAdapter(BaseWSIVLM):
    """Lightweight mock adapter for verifying the Stage 2 pipeline without GPU weights."""

    def __init__(self, model_key: str, device: str = "cpu") -> None:
        self.model_name = f"{model_key}-dryrun"
        self.device = device

    def load(self) -> None:
        """No-op weight loading for dry-run verification."""

    def predict_vqa(self, sample: VQASample, max_new_tokens: int = 128) -> str:
        """Return a deterministic dummy option choice."""
        return "A"

    def predict_caption(self, sample: CaptionSample, max_new_tokens: int = 512) -> str:
        """Return a deterministic synthetic pathology report tailored to the sample's task."""
        return (
            f"Histopathological evaluation of {sample.task} specimen ({sample.slide_id}). "
            "Microscopic examination demonstrates neoplastic proliferation with characteristic "
            "architectural growth patterns, moderate nuclear atypia, and fibrotic stroma "
            "without definitive lymphovascular invasion."
        )

    def metadata(self) -> Dict[str, Any]:
        """Return dry-run metadata."""
        return {
            "model_name": self.model_name,
            "mode": "dry_run",
            "device": self.device,
        }


def build_model_adapter(
    model_key: str,
    paths: ProjectPaths,
    device: str,
    dry_run: bool = False,
) -> BaseWSIVLM:
    """Instantiate the requested WSI VLM adapter.

    Args:
        model_key: Model identifier ('slidechat', 'prism2', 'mllm-hwsi').
        paths: Resolved project and external storage paths.
        device: CUDA/CPU device string (e.g., 'cuda:0').
        dry_run: If True, return a lightweight `DryRunCaptionAdapter` without loading GPU weights.

    Returns:
        Configured `BaseWSIVLM` adapter instance.
    """
    if dry_run:
        return DryRunCaptionAdapter(model_key=model_key, device=device)

    if model_key == "slidechat":
        from src.models.slidechat import SlideChatAdapter

        return SlideChatAdapter(
            slidechat_ckpt_path=paths.slidechat_dir,
            qwen_dir=paths.qwen_dir,
            device=device,
            paths=paths,
        )
    if model_key == "prism2":
        from src.models.prism2 import PRISM2Adapter

        return PRISM2Adapter(paths=paths, device=device)
    if model_key == "mllm-hwsi":
        from src.models.mllm_hwsi import MLLMHWSIAdapter

        return MLLMHWSIAdapter(paths=paths, device=device)

    raise NotImplementedError(
        f"Model adapter '{model_key}' is not registered in Stage 2 runner."
    )


def print_caption_evaluation_report(
    records: Sequence[Dict[str, Any]],
    metrics: Dict[str, Any],
    show_samples: bool = True,
) -> None:
    """Print a formatted human-readable Stage 2 captioning summary to stdout."""
    total = metrics.get("total_samples", 0)
    print("\n" + "=" * 100)
    print(
        f"STAGE 2 CAPTIONING EVALUATION SUMMARY | Samples: {total} | "
        f"Empty: {metrics.get('empty_predictions', 0)} | "
        f"Avg Words (Pred/Ref): {metrics.get('avg_pred_words', 0.0):.1f}/{metrics.get('avg_ref_words', 0.0):.1f}"
    )
    print("=" * 100)

    print("\n--- Overall Corpus & Macro Metrics ---")
    header = (
        f"  {'BLEU-1':>8} {'BLEU-2':>8} {'BLEU-3':>8} {'BLEU-4':>8} | "
        f"{'cBLEU-4':>8} | {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8} | "
        f"{'METEOR':>8} {'CIDEr':>8}"
    )
    row = (
        f"  {metrics.get('bleu_1', 0.0) * 100:7.2f}% "
        f"{metrics.get('bleu_2', 0.0) * 100:7.2f}% "
        f"{metrics.get('bleu_3', 0.0) * 100:7.2f}% "
        f"{metrics.get('bleu_4', 0.0) * 100:7.2f}% | "
        f"{metrics.get('corpus_bleu_4', 0.0) * 100:7.2f}% | "
        f"{metrics.get('rouge_1', 0.0) * 100:7.2f}% "
        f"{metrics.get('rouge_2', 0.0) * 100:7.2f}% "
        f"{metrics.get('rouge_l', 0.0) * 100:7.2f}% | "
        f"{metrics.get('meteor', 0.0) * 100:7.2f}% "
        f"{metrics.get('cider', 0.0):8.4f}"
    )
    if "bertscore_f1" in metrics:
        header += f" | {'BERT-F1':>8}"
        row += f" | {metrics['bertscore_f1'] * 100:7.2f}%"
    print(header)
    print("  " + "-" * (len(header) - 2))
    print(row)

    by_cohort = metrics.get("by_cohort", {})
    if len(by_cohort) > 1:
        print("\n--- Breakdown by Cohort ---")
        print(
            f"  {'Cohort':<12} {'N':>5} {'BLEU-1':>8} {'BLEU-4':>8} "
            f"{'ROUGE-L':>8} {'METEOR':>8} {'CIDEr':>8}"
        )
        for cohort_name, c_stats in by_cohort.items():
            print(
                f"  {cohort_name:<12} {c_stats['total_samples']:>5} "
                f"{c_stats['bleu_1'] * 100:7.2f}% {c_stats['bleu_4'] * 100:7.2f}% "
                f"{c_stats['rouge_l'] * 100:7.2f}% {c_stats['meteor'] * 100:7.2f}% "
                f"{c_stats['cider']:8.4f}"
            )

    by_task = metrics.get("by_task", {})
    if by_task:
        print("\n--- Breakdown by Clinical Task / Tumor / Organ ---")
        print(
            f"  {'Task/Organ':<18} {'N':>5} {'BLEU-1':>8} {'BLEU-4':>8} "
            f"{'ROUGE-L':>8} {'METEOR':>8} {'CIDEr':>8} {'Words(P/R)':>12}"
        )
        for task_name, t_stats in by_task.items():
            words_str = f"{t_stats['avg_pred_words']:.0f}/{t_stats['avg_ref_words']:.0f}"
            print(
                f"  {task_name:<18} {t_stats['total_samples']:>5} "
                f"{t_stats['bleu_1'] * 100:7.2f}% {t_stats['bleu_4'] * 100:7.2f}% "
                f"{t_stats['rouge_l'] * 100:7.2f}% {t_stats['meteor'] * 100:7.2f}% "
                f"{t_stats['cider']:8.4f} {words_str:>12}"
            )

    if show_samples and len(records) <= 25:
        print("\n--- Sample Predictions Preview ---")
        print(
            f"{'#':<3} {'Cohort':<8} {'Task':<12} {'Slide':<28} "
            f"{'BLEU-1':>7} {'ROUGE-L':>8} {'METEOR':>7} {'Pred Preview'}"
        )
        print("-" * 110)
        for idx, rec in enumerate(records, start=1):
            slide_short = str(rec.get("slide_id", ""))[:26]
            pred_preview = str(rec.get("raw_output", "")).replace("\n", " ")[:32]
            b1 = float(rec.get("bleu_1", 0.0)) * 100
            rl = float(rec.get("rouge_l", 0.0)) * 100
            mt = float(rec.get("meteor", 0.0)) * 100
            print(
                f"{idx:<3} {str(rec.get('cohort', '')):<8} {str(rec.get('task', ''))[:11]:<12} "
                f"{slide_short:<28} {b1:6.1f}% {rl:7.1f}% {mt:6.1f}% {pred_preview}"
            )


def main() -> int:
    """CLI entrypoint for Stage 2 Open-Ended WSI Captioning evaluation."""
    parser = argparse.ArgumentParser(
        description="Run Stage 2 Open-Ended Pathology Report & Captioning Evaluation on SlideBench."
    )
    parser.add_argument(
        "--model",
        choices=["slidechat", "prism2", "mllm-hwsi"],
        default="slidechat",
        help="Target WSI VLM to evaluate (default: slidechat).",
    )
    parser.add_argument(
        "--cohorts",
        nargs="+",
        choices=list(SUPPORTED_CAPTION_COHORTS),
        default=["CPTAC"],
        help="SlideBench caption cohorts to evaluate (default: CPTAC).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of caption samples per cohort (0 = full cohort, default: 0).",
    )
    parser.add_argument(
        "--pilot",
        type=int,
        default=0,
        help="Convenience alias for --limit to run a stratified pilot subset (e.g., --pilot 20).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Target CUDA/CPU device (default: cuda:0).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per pathology report (default: 512).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional custom run identifier for output files in results/stage2_caption/.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing checkpoint predictions for this run_name.",
    )
    parser.add_argument(
        "--require-wsi",
        action="store_true",
        help="Restrict evaluation to samples that have a local WSI (.svs/.tiff) file on disk.",
    )
    parser.add_argument(
        "--bertscore",
        action="store_true",
        help="Compute contextual BERTScore P/R/F1 at the end of evaluation.",
    )
    parser.add_argument(
        "--bertscore-model",
        type=str,
        default="roberta-large",
        help="HuggingFace model ID or local path for BERTScore (default: roberta-large).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run end-to-end pipeline verification with synthetic outputs without loading GPU weights.",
    )
    args = parser.parse_args()

    effective_limit = args.pilot if args.pilot > 0 else args.limit
    set_global_seed(args.seed)
    paths = ProjectPaths()

    all_samples: List[CaptionSample] = []
    for cohort in args.cohorts:
        require_features = False if args.dry_run else (args.model == "slidechat")
        require_wsi = (
            False
            if args.dry_run
            else (args.require_wsi or args.model in ("prism2", "mllm-hwsi"))
        )
        cohort_samples = load_slidebench_caption_cohort(
            datasets_dir=paths.datasets_dir,
            cohort=cohort,
            require_features=require_features,
            require_wsi=require_wsi,
        )
        if effective_limit > 0:
            cohort_samples = select_stratified_caption_pilot_samples(
                cohort_samples,
                limit=effective_limit,
            )
        all_samples.extend(cohort_samples)

    if not all_samples:
        print(
            "No caption samples found matching the requested cohorts and feature/WSI requirements."
        )
        return 1

    cohort_tag = "_".join(args.cohorts).lower()
    limit_tag = f"pilot{effective_limit}" if effective_limit > 0 else "full"
    dry_suffix = "_dryrun" if args.dry_run else ""
    run_name = args.run_name or f"{args.model}_{cohort_tag}_{limit_tag}{dry_suffix}"

    out_dir = paths.results_dir / "stage2_caption"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{run_name}_predictions.jsonl"
    summary_path = out_dir / f"{run_name}_summary.json"

    if args.overwrite and jsonl_path.exists():
        jsonl_path.unlink()

    target_uids = {s.sample_uid for s in all_samples}
    if jsonl_path.exists():
        existing_records = load_jsonl_records(jsonl_path)
        filtered_existing = [
            r for r in existing_records if str(r.get("sample_uid")) in target_uids
        ]
        if len(filtered_existing) != len(existing_records):
            jsonl_path.unlink()
            for rec in filtered_existing:
                append_jsonl_record(jsonl_path, rec)

    completed_uids = load_completed_sample_ids(jsonl_path)
    pending_samples = [s for s in all_samples if s.sample_uid not in completed_uids]

    print(
        f"[Stage 2 Captioning] Run: {run_name} | Total target samples: {len(all_samples)} | "
        f"Already completed: {len(completed_uids)} | Pending: {len(pending_samples)}"
    )

    adapter = build_model_adapter(
        args.model,
        paths=paths,
        device=args.device,
        dry_run=args.dry_run,
    )
    if pending_samples:
        adapter.load()

    for sample in tqdm(pending_samples, desc=f"Captioning ({args.model})"):
        t0 = time.perf_counter()
        raw_output = adapter.predict_caption(
            sample, max_new_tokens=args.max_new_tokens
        )
        latency_sec = round(time.perf_counter() - t0, 3)

        sample_metrics = compute_sample_lexical_metrics(
            prediction=raw_output,
            reference=sample.reference_caption,
        )

        record: Dict[str, Any] = {
            "sample_uid": sample.sample_uid,
            "cohort": sample.cohort,
            "row_id": sample.row_id,
            "case_id": sample.case_id,
            "slide_id": sample.slide_id,
            "all_slide_ids": list(sample.all_slide_ids),
            "task": sample.task,
            "question": sample.question,
            "reference_caption": sample.reference_caption,
            "raw_output": raw_output,
            **sample_metrics,
            "latency_sec": latency_sec,
        }
        append_jsonl_record(jsonl_path, record)

    all_records = load_jsonl_records(jsonl_path)
    metrics = compute_caption_metrics(
        all_records,
        compute_bertscore=args.bertscore,
        bertscore_model=args.bertscore_model,
        device=args.device,
    )
    summary_payload = {
        "run_name": run_name,
        "stage": "stage2_caption",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "cohorts": args.cohorts,
        "limit": effective_limit,
        "max_new_tokens": args.max_new_tokens,
        "dry_run": args.dry_run,
        "model_metadata": adapter.metadata(),
        "metrics": metrics,
        "predictions_file": str(jsonl_path),
    }
    write_json_file(summary_path, summary_payload)
    print_caption_evaluation_report(
        all_records,
        metrics,
        show_samples=(len(all_records) <= 25),
    )
    print(f"\nSaved predictions to: {jsonl_path}")
    print(f"Saved summary to:     {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
