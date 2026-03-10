"""
Fetch all CANN documentation by invoking sub-tools:
  - get_acl_error_doc.py   : aclError.md from runtime repo
  - get_aclnn_api_docs.py  : aclnn API docs from ops-* repos

Usage:
    python fetch_cann_docs.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

TOOLS: list[dict[str, str]] = [
    {
        "script": "get_acl_error_doc.py",
        "description": "Fetch aclError.md from runtime repo",
    },
    {
        "script": "get_aclnn_api_docs.py",
        "description": "Fetch aclnn API docs from ops-* repos",
    },
]


def run_tool(script: str, description: str) -> bool:
    script_path = SCRIPT_DIR / script
    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        return False

    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"  -> {script}")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=SCRIPT_DIR,
    )

    if result.returncode != 0:
        print(f"[ERROR] {script} exited with code {result.returncode}")
        return False

    return True


def main() -> None:
    print("=" * 60)
    print("CANN Documentation Fetch Tool")
    print("=" * 60)

    failed: list[str] = []

    for tool in TOOLS:
        ok = run_tool(tool["script"], tool["description"])
        if not ok:
            failed.append(tool["script"])

    print("\n" + "=" * 60)
    if failed:
        print(f"Completed with errors. Failed tools: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("All done! All documentation fetched successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
