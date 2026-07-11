# Copyright (c) 2026 Contributors of hahaha(https://github.com/jason-is-debugging/Hahaha)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Contributors:
# Napbad (napbad.sen@gmail.com) (https://github.com/Napbad)
# jiansongshen (jason.shen111@outlook.com) (https://github.com/jiansongshen)
#
# Dev environment setup: install gcovr, clone vcpkg, install deps from vcpkg.json,
# and configure CMake presets (VCPKG_ROOT).

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

VcpkgRepoUrl = "https://github.com/Microsoft/vcpkg"
VcpkgRootDirName = "vcpkg_root"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger()

script_path = os.path.dirname(os.path.abspath(__file__))
default_workdir = os.path.join(script_path, "..")


def _git_clone_repo(
    url: str,
    depth: int = 1,
    dest_dir: Path = Path("."),
) -> bool:
    result = True
    command = [
        "git",
        "clone",
        "--depth",
        str(depth),
        url,
        str(dest_dir),
    ]
    command_res = subprocess.run(
        command,
        capture_output=True,
        check=False,
    )
    command_stdout = command_res.stdout.decode("utf-8") if command_res.stdout else ""
    command_stderr = command_res.stderr.decode("utf-8") if command_res.stderr else ""
    if command_res.returncode != 0:
        result = False
        logger.error(
            f"Failed to clone repo: {url}, "
            f"stdout of git clone command: \n\t{command_stdout}, \n"
            f"stderr of git clone command: \n\t{command_stderr}"
        )
    return result


def _install_gcovr() -> bool:
    """Install gcovr (for coverage). Prefer pip so it works on Windows/Linux/macOS."""
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "gcovr"],
            capture_output=True,
            check=True,
        )
        logger.info("gcovr installed via pip.")
        return True
    except subprocess.CalledProcessError as e:
        logger.warning(f"Could not install gcovr via pip: {e}. Install manually (e.g. apt install gcovr).")
        return False


def _download_vcpkg_root_repo(vcpkg_root_path: Path) -> None:
    dest = vcpkg_root_path / VcpkgRootDirName
    if dest.exists():
        logger.info(f"vcpkg root directory already exists: {dest}")
        return

    logger.info(f"Cloning vcpkg into {dest} ...")
    _git_clone_repo(
        VcpkgRepoUrl,
        depth=1,
        dest_dir=dest,
    )


def _check_and_bootstrap_vcpkg(vcpkg_root_path: Path) -> bool:
    """
    Check if vcpkg is ready. If local vcpkg repo exists but binary is missing, bootstrap it.
    """
    local_vcpkg_root = vcpkg_root_path / VcpkgRootDirName

    local_vcpkg_exe = local_vcpkg_root / "vcpkg"
    if os.name == "nt":
        local_vcpkg_exe = local_vcpkg_root / "vcpkg.exe"

    if not local_vcpkg_root.exists():
        logger.error(f"vcpkg not found. Local repo not at {local_vcpkg_root}.")
        return False

    if not local_vcpkg_exe.exists():
        logger.info(
            f"Local vcpkg repo found at {local_vcpkg_root}, but 'vcpkg' binary is missing. Bootstrapping..."
        )
        bootstrap_script = "./bootstrap-vcpkg.sh"
        if os.name == "nt":
            bootstrap_script = ".\\bootstrap-vcpkg.bat"

        try:
            subprocess.run(
                [bootstrap_script, "--vcpkg-root", str(local_vcpkg_root)],
                shell=True,
                check=True,
                cwd=str(local_vcpkg_root),
            )
            logger.info("vcpkg bootstrapped successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to bootstrap vcpkg: {e}")
            return False

    return True


def _get_builtin_baseline(work_dir: Path) -> Optional[str]:
    """Read builtin-baseline from work_dir/vcpkg.json if present."""
    manifest_path = work_dir / "vcpkg.json"
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("builtin-baseline")
    except (json.JSONDecodeError, OSError):
        return None


def _ensure_vcpkg_baseline_fetched(work_dir: Path, vcpkg_root_path: Path) -> bool:
    """
    If vcpkg.json has builtin-baseline, ensure that commit is available in the
    vcpkg repo (shallow clone only has latest; fetch the baseline commit).
    """
    baseline = _get_builtin_baseline(work_dir)
    if not baseline:
        return True
    local_vcpkg_root = vcpkg_root_path / VcpkgRootDirName
    if not (local_vcpkg_root / ".git").exists():
        return True
    logger.info(f"Fetching vcpkg baseline commit {baseline} so manifest can use it...")
    res = subprocess.run(
        ["git", "-C", str(local_vcpkg_root), "fetch", "origin", baseline],
        capture_output=True,
        check=False,
    )
    if res.returncode != 0:
        logger.info("Fetch by commit failed; fetching full history (--unshallow)...")
        res2 = subprocess.run(
            ["git", "-C", str(local_vcpkg_root), "fetch", "--unshallow"],
            capture_output=True,
            check=False,
        )
        if res2.returncode != 0:
            out = (res2.stdout or b"").decode("utf-8", errors="replace")
            err = (res2.stderr or b"").decode("utf-8", errors="replace")
            logger.warning(
                f"Could not fetch vcpkg history; vcpkg install may fail. "
                f"stdout: {out} stderr: {err}"
            )
    return True


def _get_vcpkg_exe_and_root(vcpkg_root_path: Path) -> Tuple[str, Path]:
    """Return (vcpkg executable path or 'vcpkg', vcpkg root path)."""
    local_vcpkg_root = vcpkg_root_path / VcpkgRootDirName
    local_vcpkg_exe = (
        local_vcpkg_root / "vcpkg.exe" if os.name == "nt" else local_vcpkg_root / "vcpkg"
    )
    vcpkg_cmd = str(local_vcpkg_exe) if local_vcpkg_exe.exists() else "vcpkg"
    return vcpkg_cmd, local_vcpkg_root


def _vcpkg_install_manifest_mode(work_dir: Path, vcpkg_root_path: Path) -> bool:
    """
    In manifest mode, vcpkg install must be run without package arguments.
    Run from work_dir so vcpkg finds vcpkg.json and installs its dependencies.
    """
    vcpkg_cmd, local_vcpkg_root = _get_vcpkg_exe_and_root(vcpkg_root_path)
    command = [
        vcpkg_cmd,
        "install",
        "--vcpkg-root",
        str(local_vcpkg_root),
    ]
    command_env = os.environ.copy()
    command_env["VCPKG_ROOT"] = str(local_vcpkg_root)

    logger.info(f"Running: {' '.join(command)} (cwd={work_dir})")
    res = subprocess.run(
        command,
        shell=False,
        check=False,
        env=command_env,
        capture_output=True,
        cwd=str(work_dir),
    )
    if res.returncode == 0:
        logger.info("Successfully installed dependencies from vcpkg.json")
        return True

    res_stdout = res.stdout.decode("utf-8") if res.stdout else ""
    res_stderr = res.stderr.decode("utf-8") if res.stderr else ""
    logger.error(
        "Failed to install dependencies via vcpkg (manifest mode)\n"
        f"stdout of vcpkg command: \n\t{res_stdout}, \n"
        f"stderr of vcpkg command: \n\t{res_stderr}  \n"
    )
    return False


def _download_dependencies_via_vcpkg(work_dir: Path, vcpkg_root_path: Path) -> bool:
    if not _check_and_bootstrap_vcpkg(vcpkg_root_path):
        return False
    manifest_path = work_dir / "vcpkg.json"
    if manifest_path.exists():
        _ensure_vcpkg_baseline_fetched(work_dir, vcpkg_root_path)
        return _vcpkg_install_manifest_mode(work_dir, vcpkg_root_path)
    logger.error("No vcpkg.json found; cannot install dependencies in manifest mode.")
    return False


def _print_preset_instructions(work_dir: Path, vcpkg_root_path: Path) -> None:
    """Print how to set VCPKG_ROOT and use CMake presets."""
    local_vcpkg_root = vcpkg_root_path / VcpkgRootDirName
    vcpkg_root_abs = local_vcpkg_root.resolve()
    work_dir_abs = work_dir.resolve()

    logger.info("")
    logger.info("=" * 60)
    logger.info("CMake presets (CMakePresets.json) are configured.")
    logger.info("Set VCPKG_ROOT and then use a preset:")
    logger.info("")
    if os.name == "nt":
        logger.info(f"  $env:VCPKG_ROOT = \"{vcpkg_root_abs}\"")
        logger.info("  cmake --preset vcpkg-windows")
        logger.info("  cmake --build cmake-build-vcpkg-windows")
    else:
        logger.info(f"  export VCPKG_ROOT=\"{vcpkg_root_abs}\"")
        logger.info("  cmake --preset vcpkg-linux   # or vcpkg-macos on macOS")
        logger.info("  cmake --build cmake-build-vcpkg-linux")
    logger.info("")
    logger.info("For coverage (Linux/macOS):")
    logger.info(f"  export VCPKG_ROOT=\"{vcpkg_root_abs}\"")
    logger.info("  python dev/coverage.py builddir-coverage --no-cuda")
    logger.info("")
    logger.info("=" * 60)


def main() -> None:
    logger.info("Setting up development environment")
    parser = argparse.ArgumentParser(
        description="Setup dev environment: gcovr, vcpkg (clone + manifest install), CMake presets."
    )
    parser.add_argument(
        "--workdir",
        default=default_workdir,
        help="Project root (repo root). Default: directory containing dev/.",
    )
    parser.add_argument(
        "--vcpkg-root",
        default=os.path.join(default_workdir, "vcpkg"),
        help="Parent directory for vcpkg clone (vcpkg_root will be created inside). "
        "Default: <workdir>/vcpkg",
    )
    parser.add_argument(
        "--skip-gcovr",
        action="store_true",
        help="Skip installing gcovr (e.g. if already installed via system).",
    )

    args = parser.parse_args()
    work_dir = Path(args.workdir).resolve()
    vcpkg_root_path = Path(args.vcpkg_root).resolve()

    logger.info(f"Working directory: {work_dir}")
    logger.info(f"Vcpkg root parent: {vcpkg_root_path}")

    if not args.skip_gcovr:
        _install_gcovr()

    _download_vcpkg_root_repo(vcpkg_root_path)
    ok = _download_dependencies_via_vcpkg(work_dir, vcpkg_root_path)
    if not ok:
        sys.exit(1)

    _print_preset_instructions(work_dir, vcpkg_root_path)


if __name__ == "__main__":
    main()
