#!/usr/bin/env python3
"""Stage 1 Closed-Ended & Binary/Yes-No VQA Evaluation CLI Entrypoint."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm

from src.datasets.slidebench_vqa import (
    VQASample,
    load_slidebench_vqa_cohort,
    select_stratified_pilot_samples,
)
from src.evaluation.vqa_metrics import compute_vqa_metrics, extract_closed_vqa_choice
from src.models.base import BaseWSIVLM
from src.models.mllm_hwsi import MLLMHWSIAdapter
from src.models.prism2 import PRISM2Adapter
from src.models.slidechat import SlideChatAdapter
from src.utils.io import (
    append_jsonl_record,
    load_completed_sample_ids,
    load_jsonl_records,
    set_global_seed,
    write_json_file,
)
from src.utils.paths import ProjectPaths


def build_model_adapter(
    model_key: str, paths: ProjectPaths, device: str
) -> BaseWSIVLM:
    """Instantiate the requested WSI VLM adapter.

    Args:
        model_key: Model identifier ('slidechat', 'prism2', 'mllm-hwsi').
        paths: Resolved project and external storage paths.
        device: CUDA device string (e.g., 'cuda:0').

    Returns:
        Configured BaseWSIVLM adapter instance.
    """
    if model_key == "slidechat":
        return SlideChatAdapter(
            slidechat_ckpt_path=paths.slidechat_dir,
            qwen_dir=paths.qwen_dir,
            device=device,
            paths=paths,
        )
    if model_key == "prism2":
        return PRISM2Adapter(paths=paths, device=device)
    if model_key == "mllm-hwsi":
        return MLLMHWSIAdapter(paths=paths, device=device)
    raise NotImplementedError(
        f"Model adapter '{model_key}' is not yet registered in Stage 1 runner."
    )


def print_evaluation_report(
    records: List[Dict[str, Any]], metrics: Dict[str, Any], show_samples: bool = True
) -> None:
    """Print a clean human-readable summary and per-sample breakdown to stdout."""
    print("\n" + "=" * 88)
    print(
        f"STAGE 1 VQA EVALUATION SUMMARY | Total: {metrics['total_samples']} | "
        f"Correct: {metrics['correct_samples']} | "
        f"Accuracy: {metrics['overall_accuracy'] * 100:.2f}%"
    )
    print("=" * 88)

    print("\n--- Accuracy by Question Type ---")
    for qtype, stats in metrics.get("by_question_type", {}).items():
        print(
            f"  {qtype:<20}: {stats['correct']:>3}/{stats['total']:<3} "
            f"({stats['accuracy'] * 100:6.2f}%)"
        )

    print("\n--- Accuracy by Clinical Task ---")
    for task_name, stats in metrics.get("by_task", {}).items():
        print(
            f"  {task_name:<24}: {stats['correct']:>3}/{stats['total']:<3} "
            f"({stats['accuracy'] * 100:6.2f}%)"
        )

    if show_samples and len(records) <= 50:
        print("\n--- Per-Sample Predictions ---")
        print(
            f"{'#':<3} {'Slide':<8} {'Task':<22} {'Type':<16} "
            f"{'GT':<14} {'Pred':<14} {'Raw Output':<24} {'Match'}"
        )
        print("-" * 110)
        for idx, rec in enumerate(records, start=1):
            gt_disp = f"{rec['answer']} ({rec['answer_text'][:8]})"
            pred_choice = rec.get("predicted_choice") or "NONE"
            pred_text = rec.get("predicted_text") or ""
            pred_disp = f"{pred_choice} ({pred_text[:8]})" if pred_text else pred_choice
            raw_short = str(rec.get("raw_output", "")).replace("\n", " ")[:22]
            mark = "OK" if rec.get("is_correct") else "MISS"
            print(
                f"{idx:<3} {str(rec['slide_id']):<8} {str(rec['task'])[:21]:<22} "
                f"{str(rec['question_type']):<16} {gt_disp:<14} {pred_disp:<14} "
                f"{raw_short:<24} {mark}"
            )


def main() -> int:
    """CLI entrypoint for Stage 1 Closed-Ended VQA evaluation."""
    parser = argparse.ArgumentParser(
        description="Run Stage 1 Closed-Ended & Binary/Yes-No VQA Evaluation on SlideBench."
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
        choices=["BCNB", "CPTAC", "Histai"],
        default=["BCNB"],
        help="SlideBench cohorts to evaluate (default: BCNB).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Limit number of VQA samples to evaluate (0 = full dataset, default: 20).",
    )
    parser.add_argument(
        "--no-yes-no-variants",
        action="store_true",
        help="Disable explicit Yes/No question variants during pilot sampling.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Target CUDA device (default: cuda:0).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum tokens to generate per VQA answer (default: 64).",
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
        help="Optional custom run identifier for output files in results/stage1_vqa/.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing checkpoint predictions for this run_name.",
    )
    args = parser.parse_args()

    set_global_seed(args.seed)
    paths = ProjectPaths()

    all_samples: List[VQASample] = []
    for cohort in args.cohorts:
        cohort_samples = load_slidebench_vqa_cohort(
            datasets_dir=paths.datasets_dir,
            cohort=cohort,
            require_features=(args.model == "slidechat"),
        )
        if args.limit > 0:
            cohort_samples = select_stratified_pilot_samples(
                cohort_samples,
                limit=args.limit,
                include_yes_no_variants=not args.no_yes_no_variants,
            )
        all_samples.extend(cohort_samples)

    if not all_samples:
        print("No VQA samples found matching the requested cohorts and feature requirements.")
        return 1

    cohort_tag = "_".join(args.cohorts).lower()
    limit_tag = f"pilot{args.limit}" if args.limit > 0 else "full"
    run_name = args.run_name or f"{args.model}_{cohort_tag}_{limit_tag}"

    out_dir = paths.results_dir / "stage1_vqa"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{run_name}_predictions.jsonl"
    summary_path = out_dir / f"{run_name}_summary.json"

    if args.overwrite and jsonl_path.exists():
        jsonl_path.unlink()

    completed_uids = load_completed_sample_ids(jsonl_path)
    pending_samples = [s for s in all_samples if s.sample_uid not in completed_uids]

    print(
        f"[Stage 1 VQA] Run: {run_name} | Total target samples: {len(all_samples)} | "
        f"Already completed: {len(completed_uids)} | Pending: {len(pending_samples)}"
    )

    adapter = build_model_adapter(args.model, paths=paths, device=args.device)
    if pending_samples:
        adapter.load()

    for sample in tqdm(pending_samples, desc=f"Evaluating {args.model}"):
        t0 = time.perf_counter()
        raw_output = adapter.predict_vqa(sample, max_new_tokens=args.max_new_tokens)
        latency_sec = round(time.perf_counter() - t0, 3)

        pred_choice = extract_closed_vqa_choice(raw_output, sample.options)
        pred_text = sample.options.get(pred_choice, "") if pred_choice else ""
        is_correct = pred_choice == sample.answer

        record: Dict[str, Any] = {
            "sample_uid": sample.sample_uid,
            "cohort": sample.cohort,
            "row_id": sample.row_id,
            "slide_id": sample.slide_id,
            "task": sample.task,
            "question_type": sample.question_type,
            "question": sample.question,
            "options": sample.options,
            "answer": sample.answer,
            "answer_text": sample.answer_text,
            "raw_output": raw_output,
            "predicted_choice": pred_choice,
            "predicted_text": pred_text,
            "is_correct": is_correct,
            "latency_sec": latency_sec,
        }
        append_jsonl_record(jsonl_path, record)

    all_records = load_jsonl_records(jsonl_path)
    metrics = compute_vqa_metrics(all_records)
    summary_payload = {
        "run_name": run_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "cohorts": args.cohorts,
        "limit": args.limit,
        "model_metadata": adapter.metadata(),
        "metrics": metrics,
        "predictions_file": str(jsonl_path),
    }
    write_json_file(summary_path, summary_payload)
    print_evaluation_report(all_records, metrics, show_samples=(len(all_records) <= 50))
    print(f"\nSaved predictions to: {jsonl_path}")
    print(f"Saved summary to:     {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
