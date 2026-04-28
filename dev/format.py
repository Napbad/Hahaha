#!/usr/bin/env python3
"""
Cross-platform code formatting helper for this repo.

Replaces the old `format.sh` (bash + find) with a Python implementation that
works on Windows/macOS/Linux.

Usage:
  python3 dev/format.py          # format in-place
  python3 dev/format.py --check  # verify formatting (CI-friendly)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_EXTS = {".cpu", ".hpp", ".h", ".cuh"}
DEFAULT_ROOTS = ["core", "examples", "tests"]
DEFAULT_EXCLUDES = {
    "subprojects",
    "build",
    "builddir",
    ".git",
}


def _is_excluded(path: Path, exclude_names: set[str]) -> bool:
    for part in path.parts:
        if part in exclude_names:
            return True
        # common Meson dirs: builddir, builddir-foo, builddir_foo
        if part.startswith("builddir"):
            return True
    return False


def collect_files(repo_root: Path, roots: list[str], exts: set[str]) -> list[Path]:
    files: list[Path] = []
    for r in roots:
        p = (repo_root / r).resolve()
        if not p.exists():
            continue
        for f in p.rglob("*"):
            if not f.is_file():
                continue
            if f.suffix not in exts:
                continue
            if _is_excluded(f.relative_to(repo_root), DEFAULT_EXCLUDES):
                continue
            files.append(f)
    files.sort()
    return files


def run_clang_format(
    clang_format: str,
    files: list[Path],
    check: bool,
) -> int:
    # Run per-file to avoid command-line length limits (esp. Windows).
    # Also makes it easier to pinpoint the first failing file in --check mode.
    for f in files:
        cmd = [
            clang_format,
            "--style=file",
        ]
        if check:
            cmd += ["--dry-run", "--Werror"]
        else:
            cmd += ["-i"]
        cmd.append(str(f))

        proc = subprocess.run(cmd, cwd=None)
        if proc.returncode != 0:
            return proc.returncode
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run clang-format on the repo.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check formatting (no changes). Exits non-zero if reformat needed.",
    )
    parser.add_argument(
        "--clang-format",
        default=None,
        help="Path to clang-format binary (defaults to auto-detected in PATH).",
    )
    parser.add_argument(
        "--roots",
        nargs="*",
        default=DEFAULT_ROOTS,
        help="Top-level directories to scan (default: core examples tests).",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    clang_format = args.clang_format or shutil.which("clang-format")
    if not clang_format:
        print("ERROR: clang-format not found in PATH.", file=sys.stderr)
        return 2

    files = collect_files(repo_root, args.roots, DEFAULT_EXTS)
    if not files:
        print("No source files found to format.")
        return 0

    mode = "check" if args.check else "format"
    print(f"clang-format ({mode}): {len(files)} file(s)")
    return run_clang_format(clang_format, files, check=args.check)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
