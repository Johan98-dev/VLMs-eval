#!/usr/bin/env python3
"""Merge MLLM-HWSI Histai shard predictions and compute consolidated Stage 2 captioning metrics."""

import json
from datetime import datetime, timezone
from pathlib import Path

from src.datasets.slidebench_caption import load_slidebench_caption_cohort
from src.evaluation.caption_metrics import compute_caption_metrics
from src.utils.io import load_jsonl_records, write_json_file
from src.utils.paths import ProjectPaths


def main() -> None:
    paths = ProjectPaths()
    caption_dir = paths.results_dir / "stage2_caption"
    even_path = caption_dir / "mllm-hwsi_histai_shard_even_predictions.jsonl"
    odd_path = caption_dir / "mllm-hwsi_histai_shard_odd_predictions.jsonl"
    out_jsonl = caption_dir / "mllm-hwsi_histai_all1145_predictions.jsonl"
    out_summary = caption_dir / "mllm-hwsi_histai_all1145_summary.json"

    print(f"Loading even shard from {even_path}...")
    even_records = load_jsonl_records(even_path)
    print(f"Loading odd shard from {odd_path}...")
    odd_records = load_jsonl_records(odd_path)

    # Load original full Histai cohort to preserve exact original ordering and uids
    all_samples = load_slidebench_caption_cohort(paths.datasets_dir, "Histai")
    print(f"Total target Histai samples: {len(all_samples)}")

    interleaved_records = []
    for i in range(len(even_records)):
        interleaved_records.append(even_records[i])
        if i < len(odd_records):
            interleaved_records.append(odd_records[i])

    assert len(interleaved_records) == len(all_samples), (
        f"Length mismatch: {len(interleaved_records)} vs {len(all_samples)}"
    )

    # Synchronize row_id and sample_uid with official cohort definitions
    for idx, (rec, samp) in enumerate(zip(interleaved_records, all_samples)):
        assert rec["slide_id"] == samp.slide_id, f"Slide mismatch at {idx}: {rec['slide_id']} vs {samp.slide_id}"
        rec["row_id"] = samp.row_id
        rec["sample_uid"] = samp.sample_uid

    print(f"Verified 100% match across all {len(interleaved_records)} samples.")

    # Save merged JSONL
    print(f"Saving merged predictions to {out_jsonl}...")
    with out_jsonl.open("w", encoding="utf-8") as f:
        for record in interleaved_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Compute consolidated metrics
    print("Computing consolidated Stage 2 captioning metrics across 1,145 samples...")
    metrics = compute_caption_metrics(interleaved_records)

    summary_payload = {
        "run_name": "mllm-hwsi_histai_all1145",
        "stage": "stage2_caption",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": 42,
        "cohorts": ["Histai"],
        "limit": 0,
        "max_new_tokens": 512,
        "dry_run": False,
        "model_metadata": {
            "model_name": "mllm-hwsi",
            "mllm_hwsi_dir": str(paths.mllm_hwsi_dir),
            "encoders_dir": str(paths.encoders_dir),
            "device": "multi_gpu (cuda:0 + cuda:1)",
            "dtype": "torch.float16",
        },
        "metrics": metrics,
        "predictions_file": str(out_jsonl),
    }

    write_json_file(out_summary, summary_payload)
    print(f"Saved consolidated summary to {out_summary}")


if __name__ == "__main__":
    main()
