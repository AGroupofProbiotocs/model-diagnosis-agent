"""
Fetch aclnn*.md API docs from CANN ops repositories,
collect them into aclnn_api_docs/ and generate an index file.

Usage:
    python get_aclnn_api_docs.py
    python get_aclnn_api_docs.py --local-repos /path/to/ops-nn /path/to/ops-math ...
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

REPOS = [
    {
        "name": "ops-nn",
        "https": "https://gitcode.com/cann/ops-nn.git",
        "ssh": "git@gitcode.com:cann/ops-nn.git",
    },
    {
        "name": "ops-math",
        "https": "https://gitcode.com/cann/ops-math.git",
        "ssh": "git@gitcode.com:cann/ops-math.git",
    },
    {
        "name": "ops-transformer",
        "https": "https://gitcode.com/cann/ops-transformer.git",
        "ssh": "git@gitcode.com:cann/ops-transformer.git",
    },
    {
        "name": "ops-cv",
        "https": "https://gitcode.com/cann/ops-cv.git",
        "ssh": "git@gitcode.com:cann/ops-cv.git",
    },
]

TEMP_DIR = SCRIPT_DIR / "_temp_repos"
REFERENCE_DIR = PROJECT_ROOT / "docs" / "cann"
API_DOCS_DIR = REFERENCE_DIR / "aclnn_api_docs"
INDEX_FILE = REFERENCE_DIR / "aclnn_api_index.md"


def run_git(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def clone_repos() -> dict[str, Path]:
    """Clone repositories to a temp directory. Try HTTPS first, fall back to SSH on failure."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    cloned: dict[str, Path] = {}

    for repo in REPOS:
        name = repo["name"]
        dest = TEMP_DIR / name

        if dest.exists():
            print(f"[INFO] {name} already exists, running git pull ...")
            result = run_git(["pull"], cwd=dest)
            if result.returncode == 0:
                cloned[name] = dest
                print(f"[OK] {name} updated via pull.")
                continue
            print(f"[WARN] {name} pull failed: {result.stderr.strip()}")
            print(f"[INFO] Removing old directory and re-cloning {name} ...")
            shutil.rmtree(dest, ignore_errors=True)

        if not dest.exists():
            https_url = repo["https"]
            ssh_url = repo["ssh"]

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
                    continue

        cloned[name] = dest
        print(f"[OK] {name} ready.")

    return cloned


def resolve_local_repos(paths: list[str]) -> dict[str, Path]:
    """Build a repo mapping from user-provided local directories."""
    known_names = {r["name"] for r in REPOS}
    resolved: dict[str, Path] = {}

    for p in paths:
        repo_path = Path(p).resolve()
        if not repo_path.is_dir():
            print(f"[WARN] Directory does not exist, skipping: {repo_path}")
            continue
        dir_name = repo_path.name
        if dir_name in known_names:
            resolved[dir_name] = repo_path
            print(f"[OK] Using local repo: {dir_name} -> {repo_path}")
        else:
            resolved[dir_name] = repo_path
            print(f"[OK] Using local repo (unrecognized name): {dir_name} -> {repo_path}")

    return resolved


def get_branch_name(repo_path: Path) -> str:
    result = run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_path)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def get_commit_id(repo_path: Path) -> str:
    result = run_git(["rev-parse", "HEAD"], cwd=repo_path)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_and_update_docs(repos: dict[str, Path]) -> dict[str, list[str]]:
    """Search aclnn*.md in each repo and sync to aclnn_api_docs/ (add/replace only, no delete)."""
    API_DOCS_DIR.mkdir(parents=True, exist_ok=True)

    repo_files_map: dict[str, list[str]] = {}

    for name, repo_path in repos.items():
        md_files = sorted(repo_path.rglob("aclnn*.md"))
        collected: list[str] = []

        for src in md_files:
            dest = API_DOCS_DIR / src.name

            if dest.exists():
                if _file_md5(dest) == _file_md5(src):
                    collected.append(src.name)
                    continue
                tag = "REPLACE"
            else:
                tag = "ADD"

            shutil.copy2(src, dest)
            print(f"  [{tag}] {src.name} (from {name})")
            collected.append(src.name)

        repo_files_map[name] = collected
        print(f"[OK] {name}: found {len(md_files)} aclnn*.md file(s).")

    return repo_files_map


def _parse_prev_index() -> dict[str, str]:
    """Parse the existing index file to recover doc -> source mapping for deleted detection."""
    prev_map: dict[str, str] = {}
    if not INDEX_FILE.exists():
        return prev_map
    pattern = re.compile(r"^\|\s*\S+\s*\|\s*\[([^\]]+)\]\([^)]+\)\s*\|\s*(.+?)\s*\|$")
    for line in INDEX_FILE.read_text(encoding="utf-8").splitlines():
        m = pattern.match(line)
        if m:
            doc_name = m.group(1)
            source = m.group(2).strip()
            if doc_name not in prev_map and not source.startswith("deleted in"):
                prev_map[doc_name] = source
    return prev_map


def build_index(repos: dict[str, Path], repo_files_map: dict[str, list[str]]) -> None:
    """Generate / refresh aclnn_api_index.md."""
    prev_doc_source = _parse_prev_index()

    today = datetime.date.today().isoformat()
    lines: list[str] = []

    lines.append("# aclnn API Index\n")
    lines.append(f"**Last updated**: {today}\n")
    lines.append("")

    lines.append("## Repository Info\n")
    lines.append("| Repository | Branch | Commit ID |")
    lines.append("| --- | --- | --- |")
    for name, repo_path in sorted(repos.items()):
        branch = get_branch_name(repo_path)
        commit_id = get_commit_id(repo_path)
        lines.append(f"| {name} | {branch} | `{commit_id}` |")
    lines.append("")

    lines.append("## API Document List\n")

    all_docs = sorted(f.name for f in API_DOCS_DIR.glob("aclnn*.md"))

    doc_to_repo: dict[str, str] = {}
    for name, files in repo_files_map.items():
        for f in files:
            doc_to_repo[f] = name

    api_rows: list[tuple[str, str, str]] = []
    for doc in all_docs:
        stem = Path(doc).stem
        api_names = re.split(r"[&,]", stem)
        repo_name = doc_to_repo.get(doc, "")
        if repo_name:
            source = repo_name
        else:
            original = prev_doc_source.get(doc, "unknown")
            source = f"deleted in {original}"
        for api in api_names:
            api = api.strip()
            if api:
                api_rows.append((api, doc, source))

    api_rows.sort(key=lambda r: r[0])

    lines.append(f"Total: **{len(api_rows)}** API(s) from **{len(all_docs)}** document(s).\n")
    lines.append("| API | Document | Source |")
    lines.append("| --- | --- | --- |")
    for api, doc, source in api_rows:
        lines.append(f"| {api} | [{doc}](aclnn_api_docs/{doc}) | {source} |")
    lines.append("")

    INDEX_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Index file generated: {INDEX_FILE.name}")


def _force_remove_readonly(func, path, _exc_info):
    """Handle read-only files (e.g. .git objects on Windows) during rmtree."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def cleanup() -> None:
    """Remove the temp directory."""
    if TEMP_DIR.exists():
        print(f"[INFO] Cleaning up temp directory: {TEMP_DIR.name} ...")
        shutil.rmtree(TEMP_DIR, onerror=_force_remove_readonly)
        print("[OK] Temp directory removed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch aclnn API docs from CANN ops repositories."
    )
    parser.add_argument(
        "--local-repos",
        nargs="+",
        metavar="DIR",
        help=(
            "Use local repository directories instead of cloning. "
            "Provide one or more paths to repo root directories "
            "(e.g. /path/to/ops-nn /path/to/ops-math)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("aclnn API Docs Fetch Tool")
    print("=" * 60)

    use_local = args.local_repos is not None
    need_cleanup = False

    if use_local:
        print("\n--- Step 1: Resolve local repositories ---")
        repos = resolve_local_repos(args.local_repos)
    else:
        print("\n--- Step 1: Clone repositories ---")
        repos = clone_repos()
        need_cleanup = True

    if not repos:
        print("[ERROR] No repositories available. Exiting.")
        sys.exit(1)

    try:
        print("\n--- Step 2: Search and sync aclnn*.md docs ---")
        repo_files_map = collect_and_update_docs(repos)

        print("\n--- Step 3: Generate index file ---")
        build_index(repos, repo_files_map)
    finally:
        if need_cleanup:
            print("\n--- Step 4: Cleanup temp directory ---")
            cleanup()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
