"""Download RMBench demo_clean zips from Hugging Face and extract them.

Remote layout (https://huggingface.co/datasets/TianxingChen/RMBench):

    data/<task>/demo_clean.zip

Local layout after extraction (same as native collection):

    demo_clean/<task>/aloha_agilex/data/episode_0000000.hdf5
    demo_clean/<task>/aloha_agilex/video/episode_0000000.mp4
    demo_clean/<task>/aloha_agilex/instruction/episode_0000000.json

Usage (from the repo root or from this directory):

    python data/_download.py
    python data/_download.py battery_try swap_T
    bash scripts/_download_data.sh
    bash scripts/_download_data.sh battery_try
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import tempfile
from pathlib import Path
from zipfile import ZipFile


def _hub():
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install it with: pip install huggingface_hub"
        ) from exc
    return HfApi, hf_hub_download

DATA_DIR = Path(__file__).resolve().parent
DEFAULT_REPO = "TianxingChen/RMBench"
REMOTE_PREFIX = "data"
ARCHIVE_NAME = "demo_clean.zip"
TASK_CONFIG = "demo_clean"
EMBODIMENT = "aloha_agilex"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tasks",
        nargs="*",
        help="Task names to download (default: every data/<task>/demo_clean.zip on the hub)",
    )
    parser.add_argument("--repo", default=os.environ.get("HF_REPO_ID", DEFAULT_REPO))
    parser.add_argument("--revision", default=os.environ.get("HF_REVISION", "main"))
    parser.add_argument(
        "--cache",
        type=Path,
        default=DATA_DIR / "download_cache",
        help="Where to store downloaded zips (default: data/download_cache)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download archives even if they already exist in the cache",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Extract again even if demo_clean/<task>/aloha_agilex/data already has hdf5",
    )
    parser.add_argument(
        "--keep-archives",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep downloaded zips in the cache (default: keep)",
    )
    return parser.parse_args()


def discover_remote_tasks(repo_id: str, revision: str, requested: list[str]) -> list[str]:
    HfApi, _ = _hub()
    suffix = f"/{ARCHIVE_NAME}"
    files = HfApi().list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
    available = sorted(
        {
            parts[1]
            for path in files
            if (parts := path.split("/"))
            and len(parts) == 3
            and parts[0] == REMOTE_PREFIX
            and path.endswith(suffix)
        }
    )
    if requested:
        wanted = list(dict.fromkeys(requested))
        missing = [name for name in wanted if name not in available]
        if missing:
            raise SystemExit(
                f"no {REMOTE_PREFIX}/<task>/{ARCHIVE_NAME} on {repo_id}@{revision} "
                f"for: {', '.join(missing)}"
            )
        return wanted
    if not available:
        raise SystemExit(
            f"no {REMOTE_PREFIX}/<task>/{ARCHIVE_NAME} archives found in "
            f"{repo_id}@{revision}"
        )
    return available


def already_extracted(task: str) -> bool:
    data_dir = DATA_DIR / TASK_CONFIG / task / EMBODIMENT / "data"
    return data_dir.is_dir() and any(data_dir.glob("*.hdf5"))


def validate_members(zip_file: ZipFile) -> None:
    members = [info for info in zip_file.infolist() if not info.is_dir()]
    if not members:
        raise ValueError("archive is empty")
    for info in members:
        member_path = Path(info.filename)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError(f"unsafe archive member path: {info.filename!r}")
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise ValueError(f"symbolic links are not allowed: {info.filename!r}")


def locate_payload(extract_root: Path, task: str) -> Path:
    candidates = sorted(
        path
        for path in extract_root.rglob(EMBODIMENT)
        if path.is_dir() and path.parent.name == task and (path / "data").is_dir()
    )
    if not candidates:
        raise ValueError(
            f"archive does not contain {task}/{EMBODIMENT}/data/"
        )
    if len(candidates) > 1:
        relative = ", ".join(str(path.relative_to(extract_root)) for path in candidates)
        raise ValueError(f"archive contains multiple payloads for {task}: {relative}")
    return candidates[0]


def merge_payload(source: Path, destination: Path, force: bool) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source_path in sorted(source.rglob("*")):
        relative = source_path.relative_to(source)
        if ".cache" in relative.parts or "_traj_data" in relative.parts:
            continue
        dest_path = destination / relative
        if source_path.is_dir():
            dest_path.mkdir(parents=True, exist_ok=True)
            continue
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if dest_path.exists() and not force:
            continue
        shutil.copy2(source_path, dest_path)


def download_zip(
    repo_id: str,
    revision: str,
    task: str,
    cache: Path,
    force_download: bool,
) -> Path:
    _, hf_hub_download = _hub()
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            filename=f"{REMOTE_PREFIX}/{task}/{ARCHIVE_NAME}",
            local_dir=str(cache),
            force_download=force_download,
        )
    )


def extract_zip(task: str, archive: Path, force_extract: bool) -> Path:
    destination = DATA_DIR / TASK_CONFIG / task / EMBODIMENT
    print(f"[extract] {task}: {archive} -> {destination}")
    with tempfile.TemporaryDirectory(prefix=f".{TASK_CONFIG}-{task}-", dir=archive.parent) as tmp:
        extract_root = Path(tmp)
        with ZipFile(archive) as zip_file:
            validate_members(zip_file)
            zip_file.extractall(extract_root)
        payload = locate_payload(extract_root, task)
        merge_payload(payload, destination, force=force_extract)

    data_dir = destination / "data"
    if not data_dir.is_dir() or not any(data_dir.glob("*.hdf5")):
        raise ValueError(f"{archive} did not produce hdf5 under {data_dir}")
    return destination


def main() -> None:
    args = parse_args()
    tasks = discover_remote_tasks(args.repo, args.revision, args.tasks)
    args.cache.mkdir(parents=True, exist_ok=True)

    print(f"Repository: hf://datasets/{args.repo}@{args.revision}")
    print(f"Remote: {REMOTE_PREFIX}/<task>/{ARCHIVE_NAME}")
    print(f"Local: {DATA_DIR / TASK_CONFIG / '<task>' / EMBODIMENT}")
    print(f"Tasks: {len(tasks)}")

    downloaded: list[Path] = []
    for task in tasks:
        if already_extracted(task) and not args.force_extract:
            print(f"[skip] {task}: already extracted")
            continue
        print(f"[download] {REMOTE_PREFIX}/{task}/{ARCHIVE_NAME}")
        archive = download_zip(
            args.repo, args.revision, task, args.cache, args.force_download
        )
        downloaded.append(archive)
        print(f"[downloaded] {task}: {archive}")
        extract_zip(task, archive, args.force_extract)
        print(f"[done] {task}")

    if downloaded and not args.keep_archives:
        for archive in downloaded:
            archive.unlink(missing_ok=True)
        print(f"Removed downloaded archives from {args.cache}")

    print(f"Complete. Trajectories are under {DATA_DIR / TASK_CONFIG}")


if __name__ == "__main__":
    main()
