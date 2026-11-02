"""Reproducibility, JSONL checkpointing, and structured result persistence utilities."""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    """Set deterministic random seeds across Python, NumPy, and PyTorch.

    Args:
        seed: Integer random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl_records(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Load existing prediction records from a JSONL checkpoint file.

    Args:
        jsonl_path: Path to the JSONL predictions file.

    Returns:
        List of parsed JSON dictionary records.
    """
    if not jsonl_path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def load_completed_sample_ids(jsonl_path: Path) -> Set[str]:
    """Return the set of unique sample_uid values already evaluated in a JSONL file.

    Args:
        jsonl_path: Path to the JSONL predictions file.

    Returns:
        Set of completed sample_uid strings.
    """
    return {str(rec["sample_uid"]) for rec in load_jsonl_records(jsonl_path) if "sample_uid" in rec}


def append_jsonl_record(jsonl_path: Path, record: Dict[str, Any]) -> None:
    """Append a single prediction record to a JSONL file and flush immediately.

    Args:
        jsonl_path: Destination JSONL path.
        record: Serializable dictionary representing one evaluated sample.
    """
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def write_json_file(json_path: Path, payload: Dict[str, Any]) -> None:
    """Write a dictionary to a formatted JSON file.

    Args:
        json_path: Destination JSON path.
        payload: Dictionary to serialize.
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
