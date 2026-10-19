#!/usr/bin/env python3
"""Check Git-selected text files for private provenance and generated artifacts.

This is a guard against accidental additions, not a substitute for reviewing
the actual files. Diagnostics name the file and rule without echoing content.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BLOCKED_DIRS = {"results", "logs", "cluster", "data", ".ssh", ".venv", "__pycache__"}
BLOCKED_SUFFIXES = {
    ".log", ".out", ".err", ".sbatch", ".zip", ".tar", ".gz", ".pyc",
    ".pt", ".safetensors", ".mailbox", ".pem", ".key", ".sqlite", ".duckdb",
}
RULES = {
    "absolute local path": r"/(?:home|Users|scratch|lustre|mnt|nfs|workspace|root)/",
    "network address": r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])",
    "private network domain": r"\.(?:internal|corp|local)(?:\b|/)",
    "hardware model": r"\b(?:[AHBV]\d{3}|GB\d{3})(?:\s+(?:NVL|PCIe|SXM\d?))?\b",
    "scheduler directive": r"(?m)^\s*#SBATCH\b",
    "access token": r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b",
    "private key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "email address": r"[A-Z0-9][A-Z0-9._%+-]*@[A-Z0-9.-]+\.[A-Z]{2,}",
}


def main() -> None:
    output = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
    )
    names = sorted(set(output.decode().rstrip("\0").split("\0")) - {""})
    if not names:
        raise SystemExit("FAIL: no Git-selected files to check")
    failures: list[str] = []
    for name in names:
        path = ROOT / name
        relative = Path(name)
        if path.is_symlink() or not path.is_file():
            failures.append(f"{name}: not a regular file")
            continue
        if set(relative.parts) & BLOCKED_DIRS or path.suffix.lower() in BLOCKED_SUFFIXES:
            failures.append(f"{name}: excluded artifact type")
        if path.name == ".env" or path.name.startswith(".env."):
            failures.append(f"{name}: environment file")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            failures.append(f"{name}: binary file requires separate review")
            continue
        if "\0" in text:
            failures.append(f"{name}: binary content")
        for label, pattern in RULES.items():
            if re.search(pattern, name + "\n" + text, re.IGNORECASE):
                failures.append(f"{name}: {label}")
    if failures:
        raise SystemExit("FAIL:\n" + "\n".join(failures))
    print(f"PASS: {len(names)} public files checked")


if __name__ == "__main__":
    main()
