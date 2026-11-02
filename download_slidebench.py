#!/usr/bin/env python3
"""
SlideBench Non-TCGA Dataset & WSI Downloader (BCNB, CPTAC, Histai).

Organizes CSV benchmarks, pre-extracted features, and Whole-Slide Images (WSIs)
into ../Datasets/SlideBench/{BCNB,CPTAC,Histai}.
"""

import argparse
import ast
import csv
import os
import shutil
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = PROJECT_ROOT.parent / "Datasets" / "SlideBench"
os.environ.setdefault("HF_HOME", str(DEFAULT_DATASET_DIR / ".hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_DATASET_DIR / ".hf_cache" / "hub"))
os.environ.setdefault("HF_XET_CACHE", str(DEFAULT_DATASET_DIR / ".hf_cache" / "xet"))

import requests
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

SLIDECHAT_REPO = "General-Medical-AI/SlideChat"
BCNB_WSI_REPO = "Anonymous192234/BCNB"
TCIA_CPTAC_BASE_URL = (
    "https://pathdb.cancerimagingarchive.net/system/files/wsi/ross/CPTAC"
)

SLIDEBENCH_CSVS = {
    "BCNB": ["SlideBench-VQA-BCNB.csv"],
    "CPTAC": ["SlideBench-Caption-CPTAC.csv", "SlideBench-VQA-CPTAC.csv"],
    "Histai": ["SlideBench-Caption-Histai.csv", "SlideBench-VQA-Histai.csv"],
}


def setup_directories(base_dir: Path) -> Dict[str, Path]:
    """Create and return the structured directory paths for SlideBench."""
    hf_cache = base_dir / ".hf_cache"
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["HF_HUB_CACHE"] = str(hf_cache / "hub")
    os.environ["HF_XET_CACHE"] = str(hf_cache / "xet")

    dirs = {
        "BCNB": base_dir / "BCNB",
        "BCNB_WSI": base_dir / "BCNB" / "WSI",
        "BCNB_FEAT": base_dir / "BCNB" / "Feat",
        "CPTAC": base_dir / "CPTAC",
        "CPTAC_WSI": base_dir / "CPTAC" / "WSI",
        "Histai": base_dir / "Histai",
        "Histai_WSI": base_dir / "Histai" / "WSI",
        "HF_CACHE": hf_cache,
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def download_annotations_and_features(
    base_dir: Path, include_bcnb_feat: bool = True
) -> None:
    """Phase 2: Download non-TCGA SlideBench CSVs and BCNB patch features."""
    dirs = setup_directories(base_dir)
    token = os.getenv("HF_TOKEN")

    print("=== Downloading SlideBench Non-TCGA CSV Benchmarks ===")
    for cohort, filenames in SLIDEBENCH_CSVS.items():
        cohort_dir = dirs[cohort]
        for filename in filenames:
            target_path = cohort_dir / filename
            print(f"[{cohort}] Fetching {filename} -> {target_path}")
            downloaded = hf_hub_download(
                repo_id=SLIDECHAT_REPO,
                filename=filename,
                repo_type="dataset",
                token=token,
            )
            shutil.copy2(downloaded, target_path)

    if include_bcnb_feat:
        feat_rel_path = "Feat/BCNB_patch_feat.tar.gz"
        target_feat = dirs["BCNB_FEAT"] / "BCNB_patch_feat.tar.gz"
        if target_feat.exists() and target_feat.stat().st_size > 0:
            print(f"[BCNB] Feature archive already exists: {target_feat}")
        else:
            print(f"[BCNB] Fetching {feat_rel_path} -> {target_feat}")
            downloaded_feat = hf_hub_download(
                repo_id=SLIDECHAT_REPO,
                filename=feat_rel_path,
                repo_type="dataset",
                token=token,
            )
            shutil.copy2(downloaded_feat, target_feat)


def parse_cptac_slides(cptac_dir: Path) -> List[Tuple[str, str]]:
    """Extract unique (Tumor, Slide_ID) pairs from CPTAC CSVs."""
    slides: Set[Tuple[str, str]] = set()
    for csv_name in SLIDEBENCH_CSVS["CPTAC"]:
        csv_path = cptac_dir / csv_name
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing {csv_path}. Run `--phase annotations` first."
            )
        with csv_path.open("r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tumor = row["Tumor"].strip()
                slide = row["Slide"].strip()
                if tumor and slide:
                    slides.add((tumor, slide))
    return sorted(slides)


def _download_http_file(
    url: str, dest: Path, timeout: int = 180, retries: int = 3
) -> Tuple[bool, str]:
    """Download a file over HTTP/HTTPS with resume/atomic temporary file and retries."""
    if dest.exists() and dest.stat().st_size > 0:
        return True, f"Skipped (already exists): {dest.name}"

    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = dest.with_suffix(dest.suffix + ".part")

    last_err = ""
    for attempt in range(1, retries + 1):
        headers = {}
        mode = "wb"
        existing_size = temp_dest.stat().st_size if temp_dest.exists() else 0
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            mode = "ab"

        try:
            with requests.get(
                url, headers=headers, stream=True, timeout=timeout
            ) as r:
                if r.status_code == 416:  # Range already satisfied
                    temp_dest.rename(dest)
                    return True, f"Completed: {dest.name}"
                if r.status_code not in (200, 206):
                    last_err = f"HTTP {r.status_code} for {url}"
                    continue
                if r.status_code == 200 and existing_size > 0:
                    mode = "wb"
                with temp_dest.open(mode) as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            temp_dest.rename(dest)
            return True, f"Downloaded: {dest.name}"
        except Exception as exc:
            last_err = f"Error downloading {url} (attempt {attempt}/{retries}): {exc}"

    return False, last_err


def download_cptac_wsis(
    base_dir: Path, max_workers: int = 4, limit: int = 0
) -> None:
    """Phase 3: Download CPTAC .svs WSIs from TCIA PathDB."""
    dirs = setup_directories(base_dir)
    slides = parse_cptac_slides(dirs["CPTAC"])
    if limit > 0:
        slides = slides[:limit]

    print(f"=== Downloading CPTAC WSIs ({len(slides)} unique slides) ===")
    failures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for tumor, slide_id in slides:
            url = f"{TCIA_CPTAC_BASE_URL}/{tumor}/{slide_id}.svs"
            dest = dirs["CPTAC_WSI"] / tumor / f"{slide_id}.svs"
            futures[executor.submit(_download_http_file, url, dest)] = (
                tumor,
                slide_id,
            )

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="CPTAC WSIs"
        ):
            ok, msg = future.result()
            if not ok:
                failures.append(msg)

    if failures:
        print(f"Completed with {len(failures)} failed downloads:")
        for err in failures[:20]:
            print(f"  - {err}")
    else:
        print("All requested CPTAC WSIs downloaded successfully.")


def parse_histai_slides(histai_dir: Path) -> List[Tuple[str, str]]:
    """
    Extract unique (hf_repo_id, repo_relative_tiff_path) pairs from Histai CSVs.

    Example CSV entry: 'HISTAI-breast/case_0003_slide_H&E_1'
      -> repo_id: 'histai/HISTAI-breast'
      -> rel_path: 'case_0003/slide_H&E_1.tiff'
    """
    entries: Set[Tuple[str, str]] = set()
    for csv_name in SLIDEBENCH_CSVS["Histai"]:
        csv_path = histai_dir / csv_name
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing {csv_path}. Run `--phase annotations` first."
            )
        with csv_path.open("r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_slide = row["Slide"].strip()
                items = (
                    ast.literal_eval(raw_slide)
                    if raw_slide.startswith("[")
                    else [raw_slide]
                )
                for item in items:
                    repo_short, slide_str = item.split("/", 1)
                    case_id, slide_suffix = slide_str.split("_slide_", 1)
                    repo_id = f"histai/{repo_short}"
                    rel_path = f"{case_id}/slide_{slide_suffix}.tiff"
                    entries.add((repo_id, rel_path))
    return sorted(entries)


def _download_histai_single(
    repo_id: str, rel_path: str, repo_dest_dir: Path, cache_dir: Path, token: str | None
) -> Tuple[bool, str]:
    """Download a single Histai WSI .tiff from Hugging Face directly into repo_dest_dir."""
    dest = repo_dest_dir / rel_path
    if dest.exists() and dest.stat().st_size > 0:
        return True, f"Skipped: {dest.name}"
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=repo_id,
            filename=rel_path,
            repo_type="dataset",
            local_dir=repo_dest_dir,
            cache_dir=cache_dir,
            token=token,
        )
        return True, f"Downloaded: {dest.name}"
    except Exception as exc:
        if dest.parent.exists() and not any(dest.parent.iterdir()):
            dest.parent.rmdir()
        return False, f"Failed {repo_id}/{rel_path}: {exc}"


def download_histai_wsis(
    base_dir: Path, max_workers: int = 4, limit: int = 0
) -> None:
    """Phase 4: Download Histai .tiff WSIs from Hugging Face `histai/*` repos."""
    dirs = setup_directories(base_dir)
    slides = parse_histai_slides(dirs["Histai"])
    if limit > 0:
        slides = slides[:limit]

    token = os.getenv("HF_TOKEN")
    print(f"=== Downloading Histai WSIs ({len(slides)} unique slides) ===")

    # Pre-check gated file access per repository to avoid thousands of 403 requests
    sample_per_repo: Dict[str, str] = {}
    for repo_id, rel_path in slides:
        sample_per_repo.setdefault(repo_id, rel_path)

    approved_repos: Set[str] = set()
    for repo_id, sample_rel_path in sorted(sample_per_repo.items()):
        repo_short = repo_id.split("/", 1)[1]
        repo_dest_dir = dirs["Histai_WSI"] / repo_short
        ok, msg = _download_histai_single(
            repo_id,
            sample_rel_path,
            repo_dest_dir,
            dirs["HF_CACHE"] / "hub",
            token,
        )
        if ok:
            approved_repos.add(repo_id)
            print(f"  [Access OK] {repo_id}")
        else:
            first_line = msg.splitlines()[0]
            print(f"  [Access Pending/Denied] {repo_id}: {first_line}")

    active_slides = [
        (repo_id, rel_path)
        for repo_id, rel_path in slides
        if repo_id in approved_repos
    ]
    if not active_slides:
        print(
            "No Histai repositories currently have approved gated access. "
            "Re-run `--phase histai` once your Hugging Face access request is approved."
        )
        return

    failures = []
    batch_size = 40
    with tqdm(total=len(active_slides), desc="Histai WSIs") as pbar:
        for start in range(0, len(active_slides), batch_size):
            batch = active_slides[start : start + batch_size]
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for repo_id, rel_path in batch:
                    repo_short = repo_id.split("/", 1)[1]
                    repo_dest_dir = dirs["Histai_WSI"] / repo_short
                    futures[
                        executor.submit(
                            _download_histai_single,
                            repo_id,
                            rel_path,
                            repo_dest_dir,
                            dirs["HF_CACHE"] / "hub",
                            token,
                        )
                    ] = (repo_id, rel_path)

                for future in as_completed(futures):
                    ok, msg = future.result()
                    if not ok:
                        failures.append(msg)
                    pbar.update(1)

            # Periodically clean temporary Xet reconstruction caches between batches
            shutil.rmtree(dirs["HF_CACHE"] / "xet", ignore_errors=True)
            for sub in dirs["Histai_WSI"].iterdir():
                if sub.is_dir():
                    shutil.rmtree(sub / ".cache", ignore_errors=True)

    if failures:
        print(f"Completed with {len(failures)} failed downloads:")
        for err in failures[:20]:
            print(f"  - {err}")
    else:
        print("All requested Histai WSIs downloaded successfully.")


def download_bcnb_wsis(
    base_dir: Path, limit: int = 0, keep_zips: bool = False
) -> None:
    """Phase 5: Download and extract BCNB WSIs from Hugging Face mirror."""
    dirs = setup_directories(base_dir)
    token = os.getenv("HF_TOKEN")

    repo_files = list_repo_files(BCNB_WSI_REPO, repo_type="dataset", token=token)
    zip_files = sorted(
        f for f in repo_files if f.startswith("WSIs-") and f.endswith(".zip")
    )
    if limit > 0:
        zip_files = zip_files[:limit]

    tmp_zip_dir = dirs["BCNB_WSI"] / "_tmp_zips"
    tmp_zip_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Downloading & Extracting BCNB WSI Archives ({len(zip_files)} parts) ===")
    for zip_name in zip_files:
        marker = dirs["BCNB_WSI"] / f".extracted_{zip_name}"
        if marker.exists():
            print(f"[BCNB] Already extracted {zip_name}, skipping.")
            continue

        print(f"[BCNB] Downloading {zip_name}...")
        local_zip = Path(
            hf_hub_download(
                repo_id=BCNB_WSI_REPO,
                filename=zip_name,
                repo_type="dataset",
                local_dir=tmp_zip_dir,
                cache_dir=dirs["HF_CACHE"] / "hub",
                token=token,
            )
        )
        print(f"[BCNB] Extracting {zip_name} -> {dirs['BCNB_WSI']}...")
        with zipfile.ZipFile(local_zip, "r") as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                target_name = Path(member.filename).name
                if not target_name:
                    continue
                target_file = dirs["BCNB_WSI"] / target_name
                with zf.open(member) as src, target_file.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        marker.touch()
        if not keep_zips:
            local_zip.unlink(missing_ok=True)
            shutil.rmtree(dirs["HF_CACHE"] / "xet", ignore_errors=True)

    if not keep_zips:
        shutil.rmtree(tmp_zip_dir, ignore_errors=True)


def verify_dataset(base_dir: Path) -> None:
    """Phase 6: Verify CSV benchmarks and downloaded WSI counts."""
    dirs = setup_directories(base_dir)
    print("=== SlideBench Non-TCGA Verification Summary ===")

    for cohort, csv_list in SLIDEBENCH_CSVS.items():
        for csv_name in csv_list:
            p = dirs[cohort] / csv_name
            status = f"OK ({p.stat().st_size:,} bytes)" if p.exists() else "MISSING"
            print(f"  [{cohort} CSV] {csv_name}: {status}")

    bcnb_feat = dirs["BCNB_FEAT"] / "BCNB_patch_feat.tar.gz"
    print(
        f"  [BCNB Feat] BCNB_patch_feat.tar.gz: "
        f"{'OK (' + f'{bcnb_feat.stat().st_size:,} bytes)' if bcnb_feat.exists() else 'MISSING'}"
    )

    if (dirs["CPTAC"] / "SlideBench-VQA-CPTAC.csv").exists():
        cptac_expected = parse_cptac_slides(dirs["CPTAC"])
        cptac_found = sum(
            1
            for tumor, slide_id in cptac_expected
            if (dirs["CPTAC_WSI"] / tumor / f"{slide_id}.svs").exists()
        )
        print(f"  [CPTAC WSIs] {cptac_found} / {len(cptac_expected)} .svs files present")

    if (dirs["Histai"] / "SlideBench-VQA-Histai.csv").exists():
        histai_expected = parse_histai_slides(dirs["Histai"])
        histai_found = sum(
            1
            for repo_id, rel_path in histai_expected
            if (dirs["Histai_WSI"] / repo_id.split("/", 1)[1] / rel_path).exists()
        )
        print(
            f"  [Histai WSIs] {histai_found} / {len(histai_expected)} .tiff files present"
        )

    bcnb_wsi_files = [
        p
        for p in dirs["BCNB_WSI"].rglob("*")
        if p.is_file()
        and not p.name.startswith(".")
        and ".cache" not in p.parts
        and "_tmp_zips" not in p.parts
    ]
    print(f"  [BCNB WSIs] {len(bcnb_wsi_files)} extracted files in {dirs['BCNB_WSI']}")


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Download and organize SlideBench non-TCGA cohorts (BCNB, CPTAC, Histai)."
    )
    parser.add_argument(
        "--phase",
        choices=["annotations", "cptac", "histai", "bcnb", "verify", "all"],
        required=True,
        help="Execution phase to run.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Target SlideBench directory (default: {DEFAULT_DATASET_DIR})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent download workers (default: 4).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on number of WSIs/archives to download (0 = all).",
    )
    parser.add_argument(
        "--skip-bcnb-feat",
        action="store_true",
        help="Skip downloading Feat/BCNB_patch_feat.tar.gz in annotations phase.",
    )
    args = parser.parse_args()

    if args.phase in ("annotations", "all"):
        download_annotations_and_features(
            args.dataset_dir, include_bcnb_feat=not args.skip_bcnb_feat
        )
    if args.phase in ("cptac", "all"):
        download_cptac_wsis(args.dataset_dir, max_workers=args.workers, limit=args.limit)
    if args.phase in ("histai", "all"):
        download_histai_wsis(args.dataset_dir, max_workers=args.workers, limit=args.limit)
    if args.phase in ("bcnb", "all"):
        download_bcnb_wsis(args.dataset_dir, limit=args.limit)
    if args.phase in ("verify", "all"):
        verify_dataset(args.dataset_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
