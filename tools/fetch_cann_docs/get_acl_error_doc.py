"""
Fetch aclError.md from the CANN runtime repository,
save it to docs/cann/ and prepend source metadata (repo, branch, commit, date).

Usage:
    python get_acl_error_doc.py
    python get_acl_error_doc.py --local-repo /path/to/runtime
"""

from __future__ import annotations

import argparse
import datetime
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

REPO = {
    "name": "runtime",
    "https": "https://gitcode.com/cann/runtime.git",
    "ssh": "git@gitcode.com:cann/runtime.git",
}

TEMP_DIR = SCRIPT_DIR / "_temp_repos"
DOCS_DIR = PROJECT_ROOT / "docs" / "cann"
TARGET_FILENAME = "aclError.md"


def run_git(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def clone_repo() -> Path | None:
    """Clone the runtime repo. Try HTTPS first, fall back to SSH on failure."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    name = REPO["name"]
    dest = TEMP_DIR / name

    if dest.exists():
        print(f"[INFO] {name} already exists, running git pull ...")
        result = run_git(["pull"], cwd=dest)
        if result.returncode == 0:
            print(f"[OK] {name} updated via pull.")
            return dest
        print(f"[WARN] {name} pull failed: {result.stderr.strip()}")
        print(f"[INFO] Removing old directory and re-cloning {name} ...")
        shutil.rmtree(dest, ignore_errors=True)

    if not dest.exists():
        https_url = REPO["https"]
        ssh_url = REPO["ssh"]

        print(f"[INFO] Cloning {name} via HTTPS ({https_url}) ...")
        result = run_git(["clone", "--depth", "1", https_url, str(dest)])

        if result.returncode != 0:
            print(f"[WARN] HTTPS clone failed: {result.stderr.strip()}")
            print(f"[INFO] Retrying {name} via SSH ({ssh_url}) ...")
            result = run_git(["clone", "--depth", "1", ssh_url, str(dest)])

            if result.returncode != 0:
                print(f"[ERROR] SSH clone also failed for {name}!")
                print(f"  HTTPS URL: {https_url}")
                print(f"  SSH   URL: {ssh_url}")
                print(f"  Reason: {result.stderr.strip()}")
                return None

    print(f"[OK] {name} ready.")
    return dest


def resolve_local_repo(path: str) -> Path | None:
    repo_path = Path(path).resolve()
    if not repo_path.is_dir():
        print(f"[ERROR] Directory does not exist: {repo_path}")
        return None
    print(f"[OK] Using local repo: {repo_path}")
    return repo_path


def get_branch_name(repo_path: Path) -> str:
    result = run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_path)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def get_commit_id(repo_path: Path) -> str:
    result = run_git(["rev-parse", "HEAD"], cwd=repo_path)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def find_and_save_doc(repo_path: Path) -> bool:
    """Search aclError.md in the repo tree and copy it to docs/cann/ with metadata header."""
    matches = sorted(repo_path.rglob(TARGET_FILENAME))
    if not matches:
        print(f"[ERROR] {TARGET_FILENAME} not found in {repo_path}")
        return False

    src = matches[0]
    if len(matches) > 1:
        print(f"[WARN] Found {len(matches)} copies of {TARGET_FILENAME}, using: {src}")

    repo_name = REPO["name"]
    branch = get_branch_name(repo_path)
    commit_id = get_commit_id(repo_path)
    today = datetime.date.today().isoformat()

    original_content = src.read_text(encoding="utf-8")

    metadata_header = (
        f"<!-- Source: {repo_name} | Branch: {branch} "
        f"| Commit: {commit_id} | Last updated: {today} -->\n\n"
    )

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    dest = DOCS_DIR / TARGET_FILENAME
    dest.write_text(metadata_header + original_content, encoding="utf-8")

    print(f"[OK] Saved {TARGET_FILENAME} -> {dest}")
    print(f"     Repo: {repo_name}")
    print(f"     Branch: {branch}")
    print(f"     Commit: {commit_id}")
    print(f"     Date: {today}")
    return True


def _force_remove_readonly(func, path, _exc_info):
    """Handle read-only files (e.g. .git objects on Windows) during rmtree."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def cleanup() -> None:
    if TEMP_DIR.exists():
        print(f"[INFO] Cleaning up temp directory: {TEMP_DIR.name} ...")
        shutil.rmtree(TEMP_DIR, onerror=_force_remove_readonly)
        print("[OK] Temp directory removed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch aclError.md from the CANN runtime repository."
    )
    parser.add_argument(
        "--local-repo",
        metavar="DIR",
        help="Use a local runtime repository directory instead of cloning.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("aclError.md Fetch Tool")
    print("=" * 60)

    use_local = args.local_repo is not None
    need_cleanup = False

    if use_local:
        print("\n--- Step 1: Resolve local repository ---")
        repo_path = resolve_local_repo(args.local_repo)
    else:
        print("\n--- Step 1: Clone runtime repository ---")
        repo_path = clone_repo()
        need_cleanup = True

    if repo_path is None:
        print("[ERROR] No repository available. Exiting.")
        sys.exit(1)

    try:
        print(f"\n--- Step 2: Search {TARGET_FILENAME} and save to docs/cann ---")
        ok = find_and_save_doc(repo_path)
        if not ok:
            sys.exit(1)
    finally:
        if need_cleanup:
            print("\n--- Step 3: Cleanup temp directory ---")
            cleanup()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
