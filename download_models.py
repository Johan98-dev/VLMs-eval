#!/usr/bin/env python3
"""
Model Checkpoint & Weights Downloader for VLMs-eval (SlideChat, PRISM2, MLLM-HWSI, ALPaCA).

Downloads and verifies model weights, base LLMs, upstream code/configs, and reference sample
features into ../Models/, strictly outside the repository root to prevent repository bloat.
"""

import argparse
import os
import pickle
import shutil
import subprocess
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
import numpy as np
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
    repo_type: str = "model"


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


ALPACA_GIT_URL: str = "https://github.com/ZeyuGaoAi/ALPaCA.git"
ALPACA_SPECIALIST_CKPTS: Tuple[str, ...] = (
    "Llama-slideQA-BRCA.bin",
    "Llama-slideQA-Morphology.bin",
)

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
    "alpaca": ModelSpec(
        key="alpaca",
        display_name="ALPaCA (Llama-slideQA)",
        repo_id="CNX-PathLLM/Llama-slideQA",
        local_subdir="ALPaCA",
        gated=False,
        description="ALPaCA Hybrid (LongNet + GMM) WSI VLM checkpoints (Llama-slideQA*.bin)",
    ),
    "llama-3.1-8b": ModelSpec(
        key="llama-3.1-8b",
        display_name="Llama-3.1-8B-Instruct",
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        local_subdir="Llama-3.1-8B-Instruct",
        gated=True,
        description="Base instruction-tuned LLM required by ALPaCA (WPathVLM)",
    ),
    "alpaca-sample-features": ModelSpec(
        key="alpaca-sample-features",
        display_name="ALPaCA Sample Features (40 WSIs)",
        repo_id="CNX-PathLLM/Llama-slideQA-Sample-Features",
        local_subdir="ALPaCA/sample_features",
        gated=False,
        description="Official 40-slide (20 TCGA + 20 GTEx) 3-scale CONCH features for ALPaCA validation",
        repo_type="dataset",
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
        description="Patch-level vision-language encoder required for SlideChat, MLLM-HWSI, and ALPaCA",
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
    """Order repository files so configs, general model states, and smaller files download first.

    Args:
        filenames: List of (relative_filename, size_in_bytes) tuples.

    Returns:
        Sorted list of (relative_filename, size_in_bytes) tuples.
    """
    def sort_key(item: Tuple[str, int]) -> Tuple[int, int, str]:
        fname, size = item
        if "optim_states" in fname:
            priority = 2
        elif fname in ALPACA_SPECIALIST_CKPTS:
            priority = 1
        elif fname == "Llama-slideQA.bin":
            priority = 0
        else:
            priority = 0
        return (priority, size if fname != "Llama-slideQA.bin" else 0, fname)

    return sorted(filenames, key=sort_key)


def fetch_repo_manifest(
    api: HfApi, repo_id: str, repo_type: str = "model"
) -> Tuple[Optional[List[Tuple[str, int]]], Optional[str]]:
    """Retrieve the list of files and expected byte sizes for a Hugging Face repository.

    Args:
        api: Authenticated HfApi client.
        repo_id: Hugging Face repository identifier.
        repo_type: Repository type ('model' or 'dataset').

    Returns:
        Tuple of (list of (filename, size_bytes), error_message).
    """
    try:
        info = api.repo_info(repo_id=repo_id, repo_type=repo_type, files_metadata=True)
        manifest: List[Tuple[str, int]] = []
        for sibling in info.siblings or []:
            # Skip duplicate raw PyTorch consolidated checkpoint in meta-llama repos
            if sibling.rfilename.startswith("original/"):
                continue
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
    skip_specialist_ckpts: bool = False,
) -> bool:
    """Download and verify all files for a single ModelSpec into base_dir / spec.local_subdir.

    Args:
        spec: ModelSpec describing the target repository.
        base_dir: Base external Models directory.
        token: Hugging Face authentication token.
        skip_optim_states: If True, skip DeepSpeed '*_optim_states.pt' files.
        skip_specialist_ckpts: If True, skip ALPaCA Stage-3 specialist checkpoints
            ('Llama-slideQA-BRCA.bin', 'Llama-slideQA-Morphology.bin').

    Returns:
        True if all requested files were downloaded and verified, False otherwise.
    """
    dirs = setup_models_dir(base_dir)
    target_dir = dirs[spec.key]
    target_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi(token=token)
    manifest, err = fetch_repo_manifest(api, spec.repo_id, repo_type=spec.repo_type)
    if manifest is None:
        print(f"[{spec.display_name}] ERROR querying {spec.repo_id}: {err}")
        return False

    if skip_optim_states:
        manifest = [(f, s) for f, s in manifest if "optim_states" not in f]
    if skip_specialist_ckpts and spec.key == "alpaca":
        manifest = [(f, s) for f, s in manifest if f not in ALPACA_SPECIALIST_CKPTS]

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
                repo_type=spec.repo_type,
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


def setup_alpaca_repo_and_prototypes(base_dir: Path) -> bool:
    """Clone upstream ZeyuGaoAi/ALPaCA code/configs and ensure valid GMM prototype PKLs.

    Follows the authors' PANTHER K-Means clustering formulation (`n_clusters=P, max_iter=50,
    n_init=5, random_state=1`, storing `{'prototypes': cluster_centers_[np.newaxis, ...]}`)
    when GitHub LFS pointer files cannot be smudged due to upstream LFS bandwidth limits.

    Args:
        base_dir: Base external Models directory (`../Models`).

    Returns:
        True if the upstream repository and all 6 multi-scale CONCH prototype PKLs are ready.
    """
    alpaca_dir = base_dir / "ALPaCA"
    repo_dir = alpaca_dir / "repo"
    sample_feats_dir = alpaca_dir / "sample_features"
    alpaca_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"

    if not (repo_dir / "model" / "my_model.py").exists():
        print(f"\n=== Cloning upstream ALPaCA repository ({ALPACA_GIT_URL}) -> {repo_dir} ===")
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        subprocess.run(
            ["git", "clone", ALPACA_GIT_URL, str(repo_dir)],
            env=env,
            check=True,
        )
    else:
        print(f"\n=== Upstream ALPaCA repository already present at {repo_dir} ===")

    proto_base = repo_dir / "GMM_feature_extraction" / "GTEx-TCGA-prototypes"
    proto_specs = [
        # (subdir, scale_suffix, res_token, n_clusters)
        ("GTEx_TCGA_merged_output_more", "0_1024", 1024, 256),
        ("GTEx_TCGA_merged_output_more", "1_512", 2048, 128),
        ("GTEx_TCGA_merged_output_more", "1_1024", 4096, 64),
        ("GTEx_TCGA_merged_output_less", "0_1024", 1024, 32),
        ("GTEx_TCGA_merged_output_less", "1_512", 2048, 16),
        ("GTEx_TCGA_merged_output_less", "1_1024", 4096, 8),
    ]

    needs_build = False
    for subdir, _, res_token, n_clusters in proto_specs:
        pkl_path = (
            proto_base
            / subdir
            / "prototypes"
            / f"prototypes_c{n_clusters}_kmeans_num_1.0e+05_{res_token}_conch.pkl"
        )
        if not pkl_path.exists() or pkl_path.stat().st_size < 1024:
            needs_build = True
            break

    if not needs_build:
        print("  [ALPaCA Prototypes] All 6 multi-scale CONCH GMM prototype PKLs verified.")
        return True

    if not sample_feats_dir.exists() or not list(sample_feats_dir.rglob("*_0_1024.npy")):
        print(
            "  [ALPaCA Prototypes] Sample features not yet downloaded; "
            "run with 'alpaca-sample-features' to fit PANTHER K-Means prototypes."
        )
        return False

    print(
        "\n=== Fitting PANTHER K-Means Multi-Scale CONCH Prototypes from GTEx+TCGA Features ==="
    )
    import torch

    def _fit_kmeans_torch(
        x_np: np.ndarray, n_clusters: int, max_iter: int = 50, n_init: int = 5, seed: int = 1
    ) -> np.ndarray:
        """Deterministic K-Means++ in PyTorch matching PANTHER cluster(mode='kmeans')."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
        n_samples, dim = x.shape
        x_norm_sq = (x * x).sum(dim=1, keepdim=True)  # (N, 1)

        best_inertia = float("inf")
        best_centers: Optional[torch.Tensor] = None

        for init_idx in range(n_init):
            rng = np.random.RandomState(seed + init_idx)
            # K-Means++ initialization
            centers = torch.empty((n_clusters, dim), device=device, dtype=torch.float32)
            first_idx = int(rng.randint(0, n_samples))
            centers[0] = x[first_idx]
            closest_dist_sq = ((x - centers[0:1]) ** 2).sum(dim=1)

            for c_idx in range(1, n_clusters):
                probs = closest_dist_sq.detach().cpu().numpy().astype(np.float64)
                prob_sum = probs.sum()
                if prob_sum > 0:
                    probs /= prob_sum
                    next_idx = int(rng.choice(n_samples, p=probs))
                else:
                    next_idx = int(rng.randint(0, n_samples))
                centers[c_idx] = x[next_idx]
                new_dist_sq = ((x - centers[c_idx : c_idx + 1]) ** 2).sum(dim=1)
                closest_dist_sq = torch.minimum(closest_dist_sq, new_dist_sq)

            # Lloyd's iterations
            for _ in range(max_iter):
                c_norm_sq = (centers * centers).sum(dim=1).unsqueeze(0)  # (1, K)
                dist_sq = torch.clamp(x_norm_sq + c_norm_sq - 2.0 * (x @ centers.t()), min=0.0)
                min_dist_sq, labels = torch.min(dist_sq, dim=1)

                new_centers = torch.zeros_like(centers)
                counts = torch.bincount(labels, minlength=n_clusters).to(torch.float32)
                new_centers.index_add_(0, labels, x)
                non_empty = counts > 0
                new_centers[non_empty] /= counts[non_empty].unsqueeze(1)
                if (~non_empty).any():
                    empty_indices = torch.where(~non_empty)[0]
                    farthest = torch.topk(min_dist_sq, k=len(empty_indices)).indices
                    new_centers[empty_indices] = x[farthest]

                shift = torch.norm(new_centers - centers)
                centers = new_centers
                if float(shift.item()) < 1e-4:
                    break

            c_norm_sq = (centers * centers).sum(dim=1).unsqueeze(0)
            dist_sq = torch.clamp(x_norm_sq + c_norm_sq - 2.0 * (x @ centers.t()), min=0.0)
            inertia = float(torch.min(dist_sq, dim=1).values.sum().item())
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.detach().clone()

        assert best_centers is not None
        return best_centers.cpu().numpy()

    scale_features: Dict[str, np.ndarray] = {}
    for scale_suffix in ("0_1024", "1_512", "1_1024"):
        npy_files = sorted(sample_feats_dir.rglob(f"*_{scale_suffix}.npy"))
        arrays: List[np.ndarray] = []
        for npy_file in npy_files:
            obj = np.load(npy_file, allow_pickle=True)
            if isinstance(obj, np.ndarray) and obj.shape == () and isinstance(obj.item(), dict):
                feat = np.asarray(obj.item()["feature"], dtype=np.float32)
            else:
                feat = np.asarray(obj, dtype=np.float32)
            if feat.ndim == 3 and feat.shape[0] == 1:
                feat = feat[0]
            arrays.append(feat)
        pooled = np.concatenate(arrays, axis=0)
        scale_features[scale_suffix] = pooled
        print(
            f"  [Scale {scale_suffix}] Aggregated {pooled.shape[0]:,} CONCH patches "
            f"(dim={pooled.shape[1]}) across {len(npy_files)} GTEx+TCGA slides."
        )

    for subdir, scale_suffix, res_token, n_clusters in proto_specs:
        pkl_path = (
            proto_base
            / subdir
            / "prototypes"
            / f"prototypes_c{n_clusters}_kmeans_num_1.0e+05_{res_token}_conch.pkl"
        )
        if pkl_path.exists() and pkl_path.stat().st_size >= 1024:
            continue
        pkl_path.parent.mkdir(parents=True, exist_ok=True)
        feats = scale_features[scale_suffix]
        centers = _fit_kmeans_torch(
            feats, n_clusters=n_clusters, max_iter=50, n_init=5, seed=1
        )
        weights = centers[np.newaxis, ...].astype(np.float64)
        with pkl_path.open("wb") as handle:
            pickle.dump({"prototypes": weights}, handle)
        print(
            f"  [Saved Prototype] {pkl_path.name} -> shape={weights.shape} "
            f"({pkl_path.stat().st_size / 1024:.1f} KB)"
        )

    return True


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
        ("Core VLM Checkpoints & Base LLMs", MODEL_SPECS),
        ("Patch/Tile Foundation Encoders (Hugging Face)", ENCODER_SPECS),
    ):
        print(f"\n--- {group_name} ---")
        for spec in specs.values():
            target_dir = dirs[spec.key]
            manifest, err = fetch_repo_manifest(api, spec.repo_id, repo_type=spec.repo_type)
            if manifest is None:
                print(f"  [{spec.display_name}] ({spec.repo_id}): REMOTE ERROR ({err})")
                continue

            matched_files = 0
            matched_bytes = 0
            total_bytes = sum(s for _, s in manifest)
            missing_non_optim: List[str] = []
            missing_optim: List[str] = []
            missing_specialist: List[str] = []

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
                    elif spec.key == "alpaca" and rel_path in ALPACA_SPECIALIST_CKPTS:
                        missing_specialist.append(rel_path)
                    else:
                        missing_non_optim.append(rel_path)

            status = "READY" if matched_files == len(manifest) else "PARTIAL/MISSING"
            if matched_files > 0 and not missing_non_optim and missing_optim:
                status = "MODEL STATES READY (optim states pending/skipped)"
            elif matched_files > 0 and not missing_non_optim and missing_specialist:
                status = "GENERAL MODEL READY (specialist BRCA/Morphology ckpts skipped)"

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

    print("\n--- ALPaCA Upstream Repository & GMM Prototypes ---")
    repo_dir = base_dir / "ALPaCA" / "repo"
    proto_more_dir = (
        repo_dir
        / "GMM_feature_extraction"
        / "GTEx-TCGA-prototypes"
        / "GTEx_TCGA_merged_output_more"
        / "prototypes"
    )
    proto_files = sorted(proto_more_dir.glob("*.pkl")) if proto_more_dir.exists() else []
    valid_protos = [p for p in proto_files if p.stat().st_size >= 1024]
    repo_ready = (repo_dir / "model" / "my_model.py").exists() and len(valid_protos) == 3
    print(
        f"  [ALPaCA Repo & Prototypes] {ALPACA_GIT_URL} -> {repo_dir}\n"
        f"    Status: {'READY' if repo_ready else 'PARTIAL/MISSING'} | "
        f"Valid More-Version CONCH Prototypes (256/128/64): {len(valid_protos)}/3"
    )


def main() -> int:
    """CLI entrypoint for downloading and verifying VLMs-eval model weights."""
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Download and verify SlideChat, PRISM2, MLLM-HWSI, and ALPaCA checkpoints into ../Models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[
            "prism2",
            "mllm-hwsi",
            "qwen2.5-7b",
            "slidechat",
            "alpaca",
            "llama-3.1-8b",
            "alpaca-sample-features",
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
    parser.add_argument(
        "--skip-specialist-ckpts",
        action="store_true",
        help="Skip downloading ALPaCA Stage-3 specialist checkpoints (BRCA & Morphology, ~38.6 GB).",
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
        for key in (
            "prism2",
            "mllm-hwsi",
            "qwen2.5-7b",
            "slidechat",
            "alpaca",
            "llama-3.1-8b",
            "alpaca-sample-features",
        ):
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
            skip_specialist_ckpts=args.skip_specialist_ckpts,
        )

    if any(k in selected for k in ("alpaca", "alpaca-sample-features", "all")):
        setup_alpaca_repo_and_prototypes(args.models_dir)

    for gspec in gdrive_targets:
        download_gdrive_encoder_spec(spec=gspec, base_dir=args.models_dir)

    verify_models(args.models_dir, token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
