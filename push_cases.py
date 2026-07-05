#!/usr/bin/env python3
"""
push_cases.py - Push oso_airfoils/data/cases_* to GitHub in <LIMIT_KB chunks.

The cases directories can be up to 136GB each, containing nested subdirs of
individual run directories, each of which contains ~2001 txt files (~1MB each).

Strategy:
  - Walk top-level cases_* dirs
  - If a dir fits in LIMIT_KB: add + commit + push as one unit
  - If too big: recurse into subdirs
  - If a leaf run-dir is too big: batch its individual files

Each commit is immediately pushed before the next is started, keeping
pack sizes manageable and making the process resumable.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO = Path("/Users/codykarcher/Dropbox/research/oso-airfoils")
LOCAL_BRANCH = "repackaging-clean"
REMOTE_BRANCH = "repackaging"
LIMIT_KB = 1_300_000  # 1.3 GB per commit (text files compress well, stays <2GB pack)
LOG_FILE = REPO / "push_cases.log"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str):
    import time
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def run(cmd, check=True, **kwargs):
    return subprocess.run(cmd, cwd=str(REPO), check=check, **kwargs)


def get_size_kb(path: Path) -> int:
    """Disk usage in KB (using du -sk for directories, os.path.getsize for files)."""
    if path.is_file():
        return max(1, os.path.getsize(path) // 1024)
    result = subprocess.run(
        ["du", "-sk", str(path)],
        capture_output=True, text=True, cwd=str(REPO)
    )
    if result.returncode != 0:
        return 0
    return int(result.stdout.split()[0])


def is_committed(path: Path) -> bool:
    """Return True if ALL files under path are already in HEAD."""
    result = run(
        ["git", "ls-files", str(path)],
        capture_output=True, text=True, check=False
    )
    tracked = set(result.stdout.splitlines())
    if not tracked:
        return False
    # Check total files on disk vs tracked
    disk_files = set(str(f.relative_to(REPO)) for f in path.rglob("*") if f.is_file())
    return disk_files.issubset(tracked)


def has_staged() -> bool:
    result = run(["git", "diff", "--cached", "--quiet"], check=False)
    return result.returncode != 0


def git_add(path: Path):
    run(["git", "add", str(path)])


def push_commit(label: str):
    if not has_staged():
        log(f"  [SKIP] Nothing to commit: {label}")
        return
    log(f"  [COMMIT] {label}")
    run(["git", "commit", "-m", f"Add cases data: {label}"])
    log(f"  [PUSH]   {label}")
    run(["git", "push", "origin", f"{LOCAL_BRANCH}:{REMOTE_BRANCH}"])
    log(f"  [DONE]   {label}")


# ---------------------------------------------------------------------------
# Recursive batched-add logic
# ---------------------------------------------------------------------------

def push_dir(path: Path, label: str):
    """
    Add `path` to git and push, splitting into <LIMIT_KB chunks if needed.
    Handles three levels:
      cases_* -> case_N -> run_dirs -> files
    """
    if is_committed(path):
        log(f"  [SKIP] Already committed: {label}")
        return

    total_kb = get_size_kb(path)
    log(f"  Sizing {label}: {total_kb:,} KB")

    if total_kb <= LIMIT_KB:
        git_add(path)
        push_commit(label)
        return

    # Too big — split by first-level children (subdirs)
    subdirs = sorted([d for d in path.iterdir() if d.is_dir()])
    direct_files = sorted([f for f in path.iterdir() if f.is_file()])

    if subdirs:
        _push_subdirs(subdirs, direct_files, label)
    elif direct_files:
        # Leaf dir with only files (e.g. a run dir whose subdirs are all files)
        _push_files(direct_files, label)


def _push_subdirs(subdirs, direct_files, label):
    """Batch subdirs into <LIMIT_KB chunks; recurse into any oversized subdir."""
    batch = []
    batch_size = 0
    batch_num = 1

    for sub in subdirs:
        if is_committed(sub):
            continue
        sub_kb = get_size_kb(sub)

        if sub_kb > LIMIT_KB:
            # Flush current batch first
            if batch:
                for p in batch:
                    git_add(p)
                push_commit(f"{label} batch {batch_num}")
                batch = []
                batch_size = 0
                batch_num += 1
            # Recurse into this oversized subdir
            push_dir(sub, f"{sub.name} in {label}")
            continue

        if batch_size + sub_kb > LIMIT_KB and batch:
            for p in batch:
                git_add(p)
            push_commit(f"{label} batch {batch_num}")
            batch = []
            batch_size = 0
            batch_num += 1

        batch.append(sub)
        batch_size += sub_kb

    # Remaining subdirs + any direct files in the same final batch
    if direct_files:
        for f in direct_files:
            if not is_committed(f):
                batch.append(f)

    if batch:
        for p in batch:
            git_add(p)
        push_commit(f"{label} batch {batch_num}")


def _push_files(files, label):
    """Batch individual files into <LIMIT_KB chunks."""
    batch = []
    batch_size = 0
    batch_num = 1

    for f in files:
        if is_committed(f):
            continue
        f_kb = max(1, os.path.getsize(f) // 1024)

        if batch_size + f_kb > LIMIT_KB and batch:
            for p in batch:
                git_add(p)
            push_commit(f"{label} files batch {batch_num}")
            batch = []
            batch_size = 0
            batch_num += 1

        batch.append(f)
        batch_size += f_kb

    if batch:
        for p in batch:
            git_add(p)
        push_commit(f"{label} files batch {batch_num}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.chdir(REPO)
    log("=" * 60)
    log("Starting cases push")
    log("=" * 60)

    cases_base = REPO / "oso_airfoils" / "data"
    cases_dirs = sorted(cases_base.glob("cases_*"))

    if not cases_dirs:
        log("No cases_* directories found under oso_airfoils/data/")
        sys.exit(1)

    for cases_dir in cases_dirs:
        if not cases_dir.is_dir():
            continue
        log(f"\n{'='*60}")
        log(f"Processing: {cases_dir.name}")
        log("=" * 60)
        push_dir(cases_dir, cases_dir.name)

    log("\nAll cases pushed successfully!")


if __name__ == "__main__":
    main()
