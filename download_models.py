#!/usr/bin/env python3
"""
Model Checkpoint & Weights Downloader for VLMs-eval (SlideChat, PRISM2, MLLM-HWSI).

Downloads and verifies model weights and base LLM dependencies into ../Models/,
strictly outside the repository root to prevent repository bloat.
"""

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = PROJECT_ROOT.parent / "Models"
os.environ.setdefault("HF_HOME", str(DEFAULT_MODELS_DIR / ".hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_MODELS_DIR / ".hf_cache" / "hub"))
os.environ.setdefault("HF_XET_CACHE", str(DEFAULT_MODELS_DIR / ".hf_cache" / "xet"))

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
import requests
from tqdm import tqdm


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a Hugging Face model repository to download locally."""

    key: str
    display_name: str
    repo_id: str
    local_subdir: str
    gated: bool
    description: str


@dataclass(frozen=True)
class GDriveFileSpec:
    """Specification for a checkpoint file hosted on Google Drive."""

    filename: str
    file_id: str
    min_expected_bytes: int


@dataclass(frozen=True)
class GDriveEncoderSpec:
    """Specification for an encoder checkpoint group hosted on Google Drive."""

    key: str
    display_name: str
    local_subdir: str
    folder_url: str
    files: Tuple[GDriveFileSpec, ...]
    description: str


MODEL_SPECS: Dict[str, ModelSpec] = {
    "prism2": ModelSpec(
        key="prism2",
        display_name="PRISM2",
        repo_id="paige-ai/Prism2",
        local_subdir="PRISM2",
        gated=True,
        description="Paige PRISM2 WSI multimodal encoder-decoder (Perceiver + Phi-3)",
    ),
    "mllm-hwsi": ModelSpec(
        key="mllm-hwsi",
        display_name="MLLM-HWSI",
        repo_id="Bastech/MLLM-HWSI",
        local_subdir="MLLM-HWSI",
        gated=False,
        description="Hierarchical WSI MLLM checkpoint and vision-language projectors",
    ),
    "qwen2.5-7b": ModelSpec(
        key="qwen2.5-7b",
        display_name="Qwen2.5-7B-Instruct",
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        local_subdir="Qwen2.5-7B-Instruct",
        gated=False,
        description="Base instruction-tuned LLM required by SlideChat and MLLM-HWSI",
    ),
    "slidechat": ModelSpec(
        key="slidechat",
        display_name="SlideChat",
        repo_id="General-Medical-AI/SlideChat_Weight",
        local_subdir="SlideChat",
        gated=True,
        description="SlideChat Stage-1 and Stage-2 DeepSpeed checkpoints",
    ),
}

ENCODER_SPECS: Dict[str, ModelSpec] = {
    "virchow2": ModelSpec(
        key="virchow2",
        display_name="Virchow2",
        repo_id="paige-ai/Virchow2",
        local_subdir="Encoders/Virchow2",
        gated=True,
        description="Tile-level foundation encoder required for PRISM2 feature extraction",
    ),
    "conch": ModelSpec(
        key="conch",
        display_name="CONCH",
        repo_id="MahmoodLab/CONCH",
        local_subdir="Encoders/CONCH",
        gated=True,
        description="Patch-level vision-language encoder required for SlideChat and MLLM-HWSI",
    ),
}

GDRIVE_ENCODER_SPECS: Dict[str, GDriveEncoderSpec] = {
    "hipt": GDriveEncoderSpec(
        key="hipt",
        display_name="HIPT",
        local_subdir="Encoders/HIPT",
        folder_url="https://drive.google.com/drive/folders/1dzOOKTHMPbDh59zPEkOZX5ss6A6Vj9R1",
        files=(
            GDriveFileSpec(
                filename="vit_256_small_dino.pth",
                file_id="1CLMNNl1--GH4pomsT3qzK82GBq6xP8Oq",
                min_expected_bytes=704_000_000,
            ),
            GDriveFileSpec(
                filename="vit_4096_xs_dino.pth",
                file_id="1M7qWjIHuiya3qcyzgwClpoV9vDna4KIL",
                min_expected_bytes=395_000_000,
            ),
        ),
        description="Hierarchical patch (256) and region (4096) ViT encoders for MLLM-HWSI",
    ),
    "cellvit": GDriveEncoderSpec(
        key="cellvit",
        display_name="CellViT",
        local_subdir="Encoders/CellViT",
        folder_url="https://drive.google.com/drive/folders/1dEqwN5NaQ-8oIcc8zDYQo_Q_6VDMS_QU",
        files=(
            GDriveFileSpec(
                filename="CellViT-256-x40-AMP.pth",
                file_id="1_2A5I5nTZo7EEK35r06NRXw1iQfy2ZpU",
                min_expected_bytes=187_000_000,
            ),
        ),
        description="Cell-level segmentation and feature encoder for MLLM-HWSI",
    ),
}


def setup_models_dir(base_dir: Path) -> Dict[str, Path]:
    """Configure Hugging Face cache environment variables and create model directories.

    Args:
        base_dir: Root directory where model checkpoints will be stored (e.g. ../Models).

    Returns:
        Dictionary mapping directory keys to resolved Path objects.
    """
    hf_cache = base_dir / ".hf_cache"
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["HF_HUB_CACHE"] = str(hf_cache / "hub")
    os.environ["HF_XET_CACHE"] = str(hf_cache / "xet")

    dirs: Dict[str, Path] = {
        "BASE": base_dir,
        "HF_CACHE": hf_cache,
    }
    for spec in (
        *MODEL_SPECS.values(),
        *ENCODER_SPECS.values(),
        *GDRIVE_ENCODER_SPECS.values(),
    ):
        dirs[spec.key] = base_dir / spec.local_subdir

    base_dir.mkdir(parents=True, exist_ok=True)
    hf_cache.mkdir(parents=True, exist_ok=True)
    return dirs


def _prioritize_files(filenames: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    """Order repository files so configs, model states, and smaller files download before optimizer shards.

    Args:
        filenames: List of (relative_filename, size_in_bytes) tuples.

    Returns:
        Sorted list of (relative_filename, size_in_bytes) tuples.
    """
    def sort_key(item: Tuple[str, int]) -> Tuple[int, int, str]:
        fname, size = item
        is_optim = 1 if "optim_states" in fname else 0
        return (is_optim, size, fname)

    return sorted(filenames, key=sort_key)


def fetch_repo_manifest(
    api: HfApi, repo_id: str
) -> Tuple[Optional[List[Tuple[str, int]]], Optional[str]]:
    """Retrieve the list of files and expected byte sizes for a Hugging Face model repo.

    Args:
        api: Authenticated HfApi client.
        repo_id: Hugging Face model repository identifier.

    Returns:
        Tuple of (list of (filename, size_bytes), error_message).
    """
    try:
        info = api.model_info(repo_id=repo_id, files_metadata=True)
        manifest: List[Tuple[str, int]] = []
        for sibling in info.siblings or []:
            size = int(getattr(sibling, "size", 0) or 0)
            manifest.append((sibling.rfilename, size))
        return _prioritize_files(manifest), None
    except Exception as exc:
        return None, str(exc).splitlines()[0]


def download_model_spec(
    spec: ModelSpec,
    base_dir: Path,
    token: Optional[str],
    skip_optim_states: bool = False,
) -> bool:
    """Download and verify all files for a single ModelSpec into base_dir / spec.local_subdir.

    Args:
        spec: ModelSpec describing the target repository.
        base_dir: Base external Models directory.
        token: Hugging Face authentication token.
        skip_optim_states: If True, skip DeepSpeed '*_optim_states.pt' files.

    Returns:
        True if all requested files were downloaded and verified, False otherwise.
    """
    dirs = setup_models_dir(base_dir)
    target_dir = dirs[spec.key]
    target_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi(token=token)
    manifest, err = fetch_repo_manifest(api, spec.repo_id)
    if manifest is None:
        print(f"[{spec.display_name}] ERROR querying {spec.repo_id}: {err}")
        return False

    if skip_optim_states:
        manifest = [(f, s) for f, s in manifest if "optim_states" not in f]

    total_bytes = sum(s for _, s in manifest)
    print(
        f"\n=== Downloading {spec.display_name} ({spec.repo_id}) -> {target_dir} "
        f"[{len(manifest)} files, {total_bytes / (1024 ** 3):.2f} GB] ==="
    )

    all_ok = True
    for idx, (rel_path, expected_size) in enumerate(manifest, start=1):
        local_file = target_dir / rel_path
        if local_file.exists():
            actual_size = local_file.stat().st_size
            if expected_size > 0 and actual_size == expected_size:
                print(
                    f"  [{idx}/{len(manifest)}] Skipped (verified {actual_size / (1024 ** 2):.2f} MB): {rel_path}"
                )
                continue
            if expected_size == 0 and actual_size >= 0:
                print(f"  [{idx}/{len(manifest)}] Skipped (exists): {rel_path}")
                continue

        print(
            f"  [{idx}/{len(manifest)}] Fetching {rel_path} "
            f"({expected_size / (1024 ** 2):.2f} MB)..."
        )
        try:
            hf_hub_download(
                repo_id=spec.repo_id,
                filename=rel_path,
                repo_type="model",
                local_dir=target_dir,
                cache_dir=dirs["HF_CACHE"] / "hub",
                token=token,
            )
            # Clean up temporary Xet reconstruction cache after large files (>100 MB)
            if expected_size > 100 * 1024 * 1024:
                shutil.rmtree(dirs["HF_CACHE"] / "xet", ignore_errors=True)
                shutil.rmtree(target_dir / ".cache", ignore_errors=True)
        except Exception as exc:
            err_line = str(exc).splitlines()[0]
            print(f"  [{idx}/{len(manifest)}] FAILED {rel_path}: {err_line}")
            all_ok = False

    shutil.rmtree(target_dir / ".cache", ignore_errors=True)
    return all_ok


def download_gdrive_encoder_spec(spec: GDriveEncoderSpec, base_dir: Path) -> bool:
    """Download public Google Drive encoder checkpoints for MLLM-HWSI.

    Args:
        spec: GDriveEncoderSpec describing the target files and Google Drive IDs.
        base_dir: Base external Models directory.

    Returns:
        True if all files were downloaded and verified, False otherwise.
    """
    dirs = setup_models_dir(base_dir)
    target_dir = dirs[spec.key]
    target_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n=== Downloading {spec.display_name} (Google Drive) -> {target_dir} "
        f"[{len(spec.files)} files] ==="
    )
    all_ok = True
    for idx, fspec in enumerate(spec.files, start=1):
        dest = target_dir / fspec.filename
        if dest.exists() and dest.stat().st_size >= fspec.min_expected_bytes:
            actual_mb = dest.stat().st_size / (1024 ** 2)
            print(
                f"  [{idx}/{len(spec.files)}] Skipped (verified {actual_mb:.2f} MB): {fspec.filename}"
            )
            continue

        url = (
            f"https://drive.usercontent.google.com/download"
            f"?id={fspec.file_id}&export=download&confirm=t"
        )
        temp_dest = dest.with_suffix(dest.suffix + ".part")
        print(f"  [{idx}/{len(spec.files)}] Fetching {fspec.filename}...")
        try:
            with requests.get(url, stream=True, timeout=180) as response:
                response.raise_for_status()
                total_len = int(response.headers.get("Content-Length", 0) or 0)
                with temp_dest.open("wb") as out_f, tqdm(
                    total=total_len if total_len > 0 else None,
                    unit="B",
                    unit_scale=True,
                    desc=fspec.filename,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            out_f.write(chunk)
                            pbar.update(len(chunk))
            if temp_dest.stat().st_size < fspec.min_expected_bytes:
                raise RuntimeError(
                    f"Downloaded file {fspec.filename} is smaller than expected "
                    f"({temp_dest.stat().st_size} bytes)."
                )
            temp_dest.rename(dest)
        except Exception as exc:
            print(f"  [{idx}/{len(spec.files)}] FAILED {fspec.filename}: {exc}")
            all_ok = False

    return all_ok


def verify_models(base_dir: Path, token: Optional[str]) -> None:
    """Verify local model files against remote Hugging Face and Google Drive manifests.

    Args:
        base_dir: Base external Models directory.
        token: Hugging Face authentication token.
    """
    dirs = setup_models_dir(base_dir)
    api = HfApi(token=token)

    print("=== VLMs-eval Model Checkpoints Verification Summary ===")
    for group_name, specs in (
        ("Core VLM Checkpoints & Base LLM", MODEL_SPECS),
        ("Patch/Tile Foundation Encoders (Hugging Face)", ENCODER_SPECS),
    ):
        print(f"\n--- {group_name} ---")
        for spec in specs.values():
            target_dir = dirs[spec.key]
            manifest, err = fetch_repo_manifest(api, spec.repo_id)
            if manifest is None:
                print(f"  [{spec.display_name}] ({spec.repo_id}): REMOTE ERROR ({err})")
                continue

            matched_files = 0
            matched_bytes = 0
            total_bytes = sum(s for _, s in manifest)
            missing_non_optim: List[str] = []
            missing_optim: List[str] = []

            for rel_path, expected_size in manifest:
                local_file = target_dir / rel_path
                if local_file.exists() and (
                    expected_size == 0 or local_file.stat().st_size == expected_size
                ):
                    matched_files += 1
                    matched_bytes += local_file.stat().st_size
                else:
                    if "optim_states" in rel_path:
                        missing_optim.append(rel_path)
                    else:
                        missing_non_optim.append(rel_path)

            status = "READY" if matched_files == len(manifest) else "PARTIAL/MISSING"
            if matched_files > 0 and not missing_non_optim and missing_optim:
                status = "MODEL STATES READY (optim states pending/skipped)"

            print(
                f"  [{spec.display_name}] {spec.repo_id} -> {target_dir}\n"
                f"    Status: {status} | Files: {matched_files}/{len(manifest)} | "
                f"Size: {matched_bytes / (1024 ** 3):.2f} / {total_bytes / (1024 ** 3):.2f} GB"
            )
            if missing_non_optim:
                preview = ", ".join(missing_non_optim[:4])
                more = (
                    f" (+{len(missing_non_optim) - 4} more)"
                    if len(missing_non_optim) > 4
                    else ""
                )
                print(f"    Missing model files: {preview}{more}")

    print("\n--- Hierarchical Region/Cell Encoders (Google Drive) ---")
    for gspec in GDRIVE_ENCODER_SPECS.values():
        target_dir = dirs[gspec.key]
        matched = 0
        matched_bytes = 0
        missing: List[str] = []
        for fspec in gspec.files:
            local_file = target_dir / fspec.filename
            if local_file.exists() and local_file.stat().st_size >= fspec.min_expected_bytes:
                matched += 1
                matched_bytes += local_file.stat().st_size
            else:
                missing.append(fspec.filename)
        status = "READY" if matched == len(gspec.files) else "PARTIAL/MISSING"
        print(
            f"  [{gspec.display_name}] {gspec.folder_url} -> {target_dir}\n"
            f"    Status: {status} | Files: {matched}/{len(gspec.files)} | "
            f"Size: {matched_bytes / (1024 ** 3):.2f} GB"
        )
        if missing:
            print(f"    Missing files: {', '.join(missing)}")


def main() -> int:
    """CLI entrypoint for downloading and verifying VLMs-eval model weights."""
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Download and verify SlideChat, PRISM2, and MLLM-HWSI checkpoints into ../Models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[
            "prism2",
            "mllm-hwsi",
            "qwen2.5-7b",
            "slidechat",
            "encoders",
            "all",
            "verify",
        ],
        required=True,
        help="Target models to download or 'verify' to inspect status.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help=f"External directory for model weights (default: {DEFAULT_MODELS_DIR}).",
    )
    parser.add_argument(
        "--skip-optim-states",
        action="store_true",
        help="Skip downloading DeepSpeed '*_optim_states.pt' files for SlideChat (~87.6 GB).",
    )
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    selected = set(args.models)

    if "verify" in selected and len(selected) == 1:
        verify_models(args.models_dir, token)
        return 0

    targets: List[ModelSpec] = []
    gdrive_targets: List[GDriveEncoderSpec] = []
    if "all" in selected:
        targets.extend(MODEL_SPECS.values())
        targets.extend(ENCODER_SPECS.values())
        gdrive_targets.extend(GDRIVE_ENCODER_SPECS.values())
    else:
        for key in ("prism2", "mllm-hwsi", "qwen2.5-7b", "slidechat"):
            if key in selected:
                targets.append(MODEL_SPECS[key])

    if "encoders" in selected and "all" not in selected:
        targets.extend(ENCODER_SPECS.values())
        gdrive_targets.extend(GDRIVE_ENCODER_SPECS.values())

    for spec in targets:
        download_model_spec(
            spec=spec,
            base_dir=args.models_dir,
            token=token,
            skip_optim_states=args.skip_optim_states,
        )

    for gspec in gdrive_targets:
        download_gdrive_encoder_spec(spec=gspec, base_dir=args.models_dir)

    verify_models(args.models_dir, token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
