#!/usr/bin/env python3
"""
Coverage script for Hahaha project.

Usage:
    python dev/coverage.py [builddir] [--clean] [--cuda|--no-cuda]

    Use an existing build directory that already passes tests, e.g.:
    python dev/coverage.py cmake-build-vcpkg-windows

Policy:
- Focus on "core" and exclude display (UI/visualization) code from coverage.
- Use gcovr merge options to avoid double-counting template instantiations
  across multiple translation units.
--clean means clean the builddir before running.
--cuda enables CUDA support for coverage testing.
--no-cuda disables CUDA support for coverage testing.
Default behavior: auto-detect CUDA availability and enable if possible.
"""

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
import xml.etree.ElementTree as ET


def get_root_dir():
    """Get the project root directory."""
    script_dir = Path(__file__).parent
    return script_dir.parent.resolve()


def detect_cuda():
    """Auto-detect CUDA availability (nvcc + toolkit path for CMake/VS)."""
    # On Windows with Visual Studio, CMake needs the CUDA toolkit directory,
    # not just nvcc in PATH. Check environment variables first, then standard paths.
    if platform.system() == "Windows":
        # Check CUDA_PATH environment variable (most reliable)
        cuda_path = os.environ.get("CUDA_PATH")
        if cuda_path:
            cuda_dir = Path(cuda_path)
            if cuda_dir.exists():
                nvcc = cuda_dir / "bin" / "nvcc.exe"
                if nvcc.exists():
                    return True
        
        # Check versioned CUDA_PATH_V* environment variables
        for key, value in os.environ.items():
            if key.startswith("CUDA_PATH_") and value:
                cuda_dir = Path(value)
                if cuda_dir.exists():
                    nvcc = cuda_dir / "bin" / "nvcc.exe"
                    if nvcc.exists():
                        return True
        
        # Check standard CUDA installation paths
        cuda_base_paths = [
            Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA"),
            Path("C:/Program Files (x86)/NVIDIA GPU Computing Toolkit/CUDA"),
        ]
        for base in cuda_base_paths:
            if not base.exists():
                continue
            # Toolkit has versioned subdirs like v12.3
            try:
                subdirs = [d for d in base.iterdir() if d.is_dir() and d.name.startswith("v")]
                if subdirs:
                    # Check nvcc exists in one of them (e.g. bin/nvcc.exe)
                    for ver in subdirs:
                        nvcc = ver / "bin" / "nvcc.exe"
                        if nvcc.exists():
                            return True
            except OSError:
                pass
        
        # Last resort: check if nvcc is in PATH (less reliable for VS)
        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                capture_output=True,
            )
            # If nvcc works, check if we can find its installation path
            # by checking common locations relative to where nvcc might be
            return True  # nvcc found, assume it might work
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        return False

    # Unix-like: nvcc in PATH and /opt/cuda or similar
    try:
        subprocess.run(
            ["nvcc", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        if Path("/opt/cuda").exists():
            return True
        # Check CUDA_PATH on Unix too
        cuda_path = os.environ.get("CUDA_PATH")
        if cuda_path and Path(cuda_path).exists():
            return True
        return True  # nvcc found on Unix, assume toolkit is set up
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return False


def get_cmake_generator(build_dir: Path):
    """Get appropriate CMake generator for the platform."""
    def _cmake_generators() -> set[str]:
        """Return the set of generators supported by the current cmake."""
        try:
            p = subprocess.run(
                ["cmake", "--help"],
                capture_output=True,
                text=True,
                check=True,
            )
            text = (p.stdout or "") + (p.stderr or "")
        except Exception:
            return set()

        # Parse the "Generators" section best-effort.
        gens: set[str] = set()
        in_section = False
        for line in text.splitlines():
            if line.strip() == "Generators":
                in_section = True
                continue
            if not in_section:
                continue
            # Stop when we leave the section (blank line after content)
            if in_section and not line.strip():
                # Keep going; some cmake versions have extra whitespace.
                continue
            # Typical format:
            # * Ninja                       = Generates build.ninja files.
            #   Unix Makefiles              = Generates standard UNIX makefiles.
            m = re.match(r"^\s*(?:\*?\s*)?(.+?)\s*=\s+Generates", line)
            if m:
                gens.add(m.group(1).strip())
        return gens

    def _is_windows() -> bool:
        return platform.system() == "Windows"

    def _default_generator() -> str:
        # Prefer Ninja when available everywhere; otherwise fall back to Unix Makefiles on Unix.
        available = _cmake_generators()
        if "Ninja" in available:
            return "Ninja"
        if not _is_windows() and "Unix Makefiles" in available:
            return "Unix Makefiles"
        # Windows: keep historical default.
        return "Visual Studio 17 2022" if _is_windows() else "Ninja"

    # Check if CMakeCache.txt exists and read the generator
    cmake_cache = build_dir / "CMakeCache.txt"
    if cmake_cache.exists():
        try:
            with open(cmake_cache, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("CMAKE_GENERATOR:"):
                        match = re.search(r"CMAKE_GENERATOR:INTERNAL=(.+)", line)
                        if match:
                            generator = match.group(1).strip()
                            # If the cache came from a different platform/toolchain (e.g. VS on WSL),
                            # don't reuse it.
                            if (not _is_windows()) and generator.startswith("Visual Studio"):
                                fallback = _default_generator()
                                print(
                                    f"Found cached generator '{generator}', but host is not Windows. "
                                    f"Using '{fallback}' instead."
                                )
                                return fallback
                            available = _cmake_generators()
                            if available and generator not in available:
                                fallback = _default_generator()
                                print(
                                    f"Cached generator '{generator}' is not available on this host. "
                                    f"Using '{fallback}' instead."
                                )
                                return fallback
                            print(f"Using existing generator: {generator}")
                            return generator
        except Exception:
            pass  # If we can't read it, fall back to default

    # Choose generator based on platform
    if platform.system() == "Windows":
        # Try to detect Visual Studio version
        # Check for Visual Studio 2025 (VS 18)
        vs2025_paths = [
            Path("C:/Program Files/Microsoft Visual Studio/2025/Community"),
            Path("C:/Program Files/Microsoft Visual Studio/2025/Professional"),
            Path("C:/Program Files/Microsoft Visual Studio/2025/Enterprise"),
            Path("C:/Program Files/Microsoft Visual Studio/2025/Preview"),
        ]
        for vs_path in vs2025_paths:
            if vs_path.exists():
                return "Visual Studio 18 2026"
        
        # Check for Visual Studio 2022 (VS 17)
        vs2022_paths = [
            Path("C:/Program Files/Microsoft Visual Studio/2022/Community"),
            Path("C:/Program Files/Microsoft Visual Studio/2022/Professional"),
            Path("C:/Program Files/Microsoft Visual Studio/2022/Enterprise"),
        ]
        for vs_path in vs2022_paths:
            if vs_path.exists():
                return "Visual Studio 17 2022"
        
        # Check for Visual Studio 2019 (VS 16)
        vs2019_paths = [
            Path("C:/Program Files (x86)/Microsoft Visual Studio/2019/Community"),
            Path("C:/Program Files (x86)/Microsoft Visual Studio/2019/Professional"),
            Path("C:/Program Files (x86)/Microsoft Visual Studio/2019/Enterprise"),
        ]
        for vs_path in vs2019_paths:
            if vs_path.exists():
                return "Visual Studio 16 2019"
        
        # Default to Visual Studio 17 2022
        return "Visual Studio 17 2022"
    else:
        # Unix-like systems: prefer Ninja but allow fallback.
        return _default_generator()


def get_vcpkg_triplet() -> str | None:
    """Return the appropriate vcpkg triplet for this host (or None to let vcpkg pick)."""
    # Respect user override.
    env_triplet = os.environ.get("VCPKG_TARGET_TRIPLET")
    if env_triplet:
        return env_triplet
    if platform.system() == "Windows":
        return "x64-windows-static"
    # On Linux/macOS, do not force a Windows triplet.
    return None


def get_vcpkg_toolchain_file(root_dir: Path) -> str | None:
    """Return the vcpkg toolchain file to use (or None to let CMakeLists decide).

    Priority:
    - environment CMAKE_TOOLCHAIN_FILE (explicit override)
    - environment VCPKG_ROOT (use <VCPKG_ROOT>/scripts/buildsystems/vcpkg.cmake)
    - repo-local vcpkg toolchain if present (<repo>/vcpkg/vcpkg_root/...)
    """
    env_toolchain = os.environ.get("CMAKE_TOOLCHAIN_FILE")
    if env_toolchain:
        return env_toolchain

    env_vcpkg_root = os.environ.get("VCPKG_ROOT")
    if env_vcpkg_root:
        candidate = Path(env_vcpkg_root) / "scripts" / "buildsystems" / "vcpkg.cmake"
        if candidate.exists():
            return str(candidate)

    repo_toolchain = root_dir / "vcpkg" / "vcpkg_root" / "scripts" / "buildsystems" / "vcpkg.cmake"
    if repo_toolchain.exists():
        return str(repo_toolchain)

    return None


def clean_build_dir(build_dir: Path):
    """Clean the build directory if it exists."""
    if build_dir.exists():
        print(f"Cleaning build directory: {build_dir}")
        shutil.rmtree(build_dir)


def configure_cmake(build_dir: Path, enable_cuda: str, capture_output: bool = False):
    """Configure CMake with appropriate settings.
    If capture_output is True, returns (success, stderr_text) for fallback handling.
    """
    def _looks_like_cache_mismatch(text: str) -> bool:
        # Common error strings when reusing a build directory created from a different
        # absolute path (Windows drive vs WSL /mnt path, different source dir, etc.).
        needles = [
            "CMakeCache.txt directory",
            "is different than the directory",
            "does not match the source",
            "used to generate cache",
            "Re-run cmake with a different source directory",
        ]
        t = text or ""
        return any(n in t for n in needles)

    def _wipe_cmake_cache(dir_path: Path) -> None:
        # Keep this conservative: remove the cache and CMakeFiles to force a clean re-configure.
        for p in [dir_path / "CMakeCache.txt", dir_path / "CMakeFiles"]:
            try:
                if p.is_dir():
                    shutil.rmtree(p)
                elif p.exists():
                    p.unlink()
            except Exception:
                pass

    def _cached_toolchain_file(dir_path: Path) -> str | None:
        cache = dir_path / "CMakeCache.txt"
        if not cache.exists():
            return None
        try:
            for line in cache.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("CMAKE_TOOLCHAIN_FILE:"):
                    m = re.search(r"CMAKE_TOOLCHAIN_FILE:(?:FILEPATH|UNINITIALIZED)=(.+)", line)
                    if m:
                        return m.group(1).strip()
        except Exception:
            return None
        return None

    def _cached_vcpkg_root_dir(dir_path: Path) -> str | None:
        cache = dir_path / "CMakeCache.txt"
        if not cache.exists():
            return None
        try:
            for line in cache.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("Z_VCPKG_ROOT_DIR:INTERNAL="):
                    return line.split("=", 1)[1].strip()
        except Exception:
            return None
        return None

    root_dir = get_root_dir()
    generator = get_cmake_generator(build_dir)
    
    cmake_args = [
        "cmake",
        "-S",
        ".",
        "-B",
        str(build_dir),
        "-G",
        generator,
    ]
    vcpkg_triplet = get_vcpkg_triplet()
    vcpkg_toolchain = get_vcpkg_toolchain_file(root_dir)

    # If the build dir was configured with a different toolchain (common when switching
    # between Windows and WSL), wipe cache so vcpkg/toolchain is consistent.
    cached_toolchain = _cached_toolchain_file(build_dir)
    if vcpkg_toolchain and cached_toolchain and Path(cached_toolchain) != Path(vcpkg_toolchain):
        print(
            f"Build dir toolchain mismatch (cached '{cached_toolchain}' vs requested '{vcpkg_toolchain}'). "
            "Wiping CMake cache..."
        )
        _wipe_cmake_cache(build_dir)

    # vcpkg toolchain caches the resolved root dir internally (Z_VCPKG_ROOT_DIR). If that
    # points at a drvfs mount (/mnt/c, /mnt/d), vcpkg/CMake can fail with "Operation not permitted".
    if vcpkg_toolchain:
        expected_root = str(Path(vcpkg_toolchain).resolve().parents[2])  # <root>/scripts/buildsystems/vcpkg.cmake
        cached_root = _cached_vcpkg_root_dir(build_dir)
        if cached_root and Path(cached_root) != Path(expected_root):
            print(
                f"Build dir vcpkg root mismatch (cached '{cached_root}' vs expected '{expected_root}'). "
                "Wiping CMake cache..."
            )
            _wipe_cmake_cache(build_dir)
    
    # Visual Studio generators use --config instead of CMAKE_BUILD_TYPE
    if generator.startswith("Visual Studio"):
        cmake_args.extend([
            "-DHAHAHA_DISPLAY=OFF",
            "-DHAHAHA_BUILD_TESTS=ON",
            "-DHAHAHA_BUILD_EXAMPLES=OFF",
            "-DHAHAHA_ENABLE_COVERAGE=ON",
            "-DVCPKG_INSTALL_OPTIONS=--binarysource=clear",
            "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDebug",
        ])
        if vcpkg_toolchain:
            cmake_args.append(f"-DCMAKE_TOOLCHAIN_FILE={vcpkg_toolchain}")
        if vcpkg_triplet:
            cmake_args.append(f"-DVCPKG_TARGET_TRIPLET={vcpkg_triplet}")
    else:
        cmake_args.extend([
            "-DCMAKE_BUILD_TYPE=Debug",
            "-DHAHAHA_DISPLAY=OFF",
            "-DHAHAHA_BUILD_TESTS=ON",
            "-DHAHAHA_BUILD_EXAMPLES=OFF",
            "-DHAHAHA_ENABLE_COVERAGE=ON",
            "-DVCPKG_INSTALL_OPTIONS=--binarysource=clear",
        ])
        if vcpkg_toolchain:
            cmake_args.append(f"-DCMAKE_TOOLCHAIN_FILE={vcpkg_toolchain}")
        if vcpkg_triplet:
            cmake_args.append(f"-DVCPKG_TARGET_TRIPLET={vcpkg_triplet}")

    if enable_cuda == "on":
        cmake_args.append("-DHAHAHA_USE_CUDA=ON")
        print("Building with CUDA support enabled.")
    else:
        cmake_args.append("-DHAHAHA_USE_CUDA=OFF")
        print("Building without CUDA support.")

    print(f"Running: {' '.join(cmake_args)}")
    if capture_output:
        result = subprocess.run(
            cmake_args,
            capture_output=True,
            text=True,
        )
        out = (result.stderr or "") + (result.stdout or "")
        # If the build dir was generated from a different absolute path (common on WSL),
        # wipe cache and retry once.
        if result.returncode != 0 and _looks_like_cache_mismatch(out):
            print("CMake cache/source dir mismatch detected. Wiping CMake cache and retrying...")
            _wipe_cmake_cache(build_dir)
            result2 = subprocess.run(
                cmake_args,
                capture_output=True,
                text=True,
            )
            out2 = (result2.stderr or "") + (result2.stdout or "")
            return result2.returncode == 0, out + "\n" + out2
        return result.returncode == 0, out
    else:
        # Always capture output to show errors even when not using capture_output mode
        result = subprocess.run(
            cmake_args,
            capture_output=True,
            text=True,
        )
        out = (result.stdout or "") + (result.stderr or "")
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.returncode != 0 and _looks_like_cache_mismatch(out):
            print("CMake cache/source dir mismatch detected. Wiping CMake cache and retrying...")
            _wipe_cmake_cache(build_dir)
            result = subprocess.run(
                cmake_args,
                capture_output=True,
                text=True,
            )
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmake_args, result.stdout, result.stderr)
        return None


def build_project(build_dir: Path, parallel: int = 8):
    """Build the project."""
    print(f"Building project with {parallel} parallel jobs...")
    build_args = ["cmake", "--build", str(build_dir), "--parallel", str(parallel)]
    
    # Visual Studio generators need --config Debug for coverage
    cmake_cache = build_dir / "CMakeCache.txt"
    if cmake_cache.exists():
        try:
            with open(cmake_cache, "r", encoding="utf-8") as f:
                content = f.read()
                if "Visual Studio" in content:
                    build_args.extend(["--config", "Debug"])
        except Exception:
            pass
    
    subprocess.run(build_args, check=True)


def find_test_executable(build_dir: Path):
    """Find the test executable in the build directory."""
    # Common locations for test executables (single-config and multi-config)
    test_paths = [
        build_dir / "tests" / "Debug" / "hahaha_tests.exe",  # VS multi-config Debug
        build_dir / "tests" / "Release" / "hahaha_tests.exe",  # VS multi-config Release
        build_dir / "tests" / "hahaha_tests.exe",  # Single-config (Ninja, etc.) or VS run dir
        build_dir / "tests" / "Debug" / "hahaha_tests",  # Unix Debug
        build_dir / "tests" / "Release" / "hahaha_tests",  # Unix Release
        build_dir / "tests" / "hahaha_tests",  # Unix single-config
    ]
    
    for test_path in test_paths:
        if test_path.exists():
            return test_path
    return None


def is_visual_studio_build(build_dir: Path) -> bool:
    """Return True if the build directory is using a Visual Studio generator."""
    cmake_cache = build_dir / "CMakeCache.txt"
    if not cmake_cache.exists():
        return False
    try:
        text = cmake_cache.read_text(encoding="utf-8", errors="ignore")
        return "Visual Studio" in text
    except Exception:
        return False


def find_opencppcoverage(user_path: str | None) -> Path | None:
    """Find OpenCppCoverage.exe.

    Order:
    - user provided path
    - PATH (OpenCppCoverage.exe)
    - common install locations (including repo's default hint)
    """
    candidates: list[Path] = []
    if user_path:
        candidates.append(Path(user_path))

    # PATH lookup
    which = shutil.which("OpenCppCoverage.exe")
    if which:
        candidates.append(Path(which))

    # Common locations (keep the user's hinted location first)
    candidates.extend([
        Path(r"D:\programs\OpenCppCoverage\OpenCppCoverage.exe"),
        Path(r"C:\Program Files\OpenCppCoverage\OpenCppCoverage.exe"),
        Path(r"C:\Program Files (x86)\OpenCppCoverage\OpenCppCoverage.exe"),
    ])

    for c in candidates:
        try:
            if c.exists():
                return c
        except Exception:
            pass
    return None


def parse_cobertura_summary(cobertura_xml: Path) -> dict:
    """Parse Cobertura XML and return summary rates and counts (best-effort)."""
    tree = ET.parse(cobertura_xml)
    root = tree.getroot()
    # Cobertura uses attributes like line-rate/branch-rate (0..1)
    line_rate = float(root.attrib.get("line-rate", "0") or "0")
    branch_rate = float(root.attrib.get("branch-rate", "0") or "0")

    # Some exporters include totals
    lines_valid = int(root.attrib.get("lines-valid", "0") or "0")
    lines_covered = int(root.attrib.get("lines-covered", "0") or "0")
    branches_valid = int(root.attrib.get("branches-valid", "0") or "0")
    branches_covered = int(root.attrib.get("branches-covered", "0") or "0")

    return {
        "line_rate": line_rate,
        "branch_rate": branch_rate,
        "lines_valid": lines_valid,
        "lines_covered": lines_covered,
        "branches_valid": branches_valid,
        "branches_covered": branches_covered,
    }


def run_coverage_opencppcoverage(root_dir: Path, build_dir: Path, opencppcoverage_path: Path, export_html: bool):
    """Run coverage analysis using OpenCppCoverage (Windows/MSVC)."""
    test_exe = find_test_executable(build_dir)
    if not test_exe:
        raise FileNotFoundError(f"Could not find test executable under build dir: {build_dir}")

    out_dir = root_dir / "coverage_report"
    cobertura_xml = root_dir / "coverage.xml"

    # Always generate Cobertura XML so we can print a CLI summary.
    export_args = ["--export_type", f"cobertura:{cobertura_xml}"]
    if export_html:
        export_args += ["--export_type", f"html:{out_dir}"]

    # Source selection: focus on core, exclude tests/vcpkg/examples/display.
    # Note: OpenCppCoverage's patterns are glob-like.
    args = [
        str(opencppcoverage_path),
        "--sources", str(root_dir / "core"),
        "--excluded_sources", "*tests*",
        "--excluded_sources", "*vcpkg*",
        "--excluded_sources", "*examples*",
        "--excluded_sources", "*core\\src\\display*",
        "--excluded_sources", "*core\\include\\display*",
        *export_args,
        "--",
        str(test_exe),
    ]

    print("== coverage (OpenCppCoverage) ==")
    print("Running:", " ".join(args))
    subprocess.run(args, check=True, cwd=root_dir)

    if cobertura_xml.exists():
        s = parse_cobertura_summary(cobertura_xml)
        line_pct = s["line_rate"] * 100.0
        branch_pct = s["branch_rate"] * 100.0
        print()
        print("== coverage summary (from cobertura xml) ==")
        if s["lines_valid"] > 0:
            print(f"Line   : {line_pct:.2f}% ({s['lines_covered']}/{s['lines_valid']})")
        else:
            print(f"Line   : {line_pct:.2f}%")
        if s["branches_valid"] > 0:
            print(f"Branch : {branch_pct:.2f}% ({s['branches_covered']}/{s['branches_valid']})")
        else:
            print(f"Branch : {branch_pct:.2f}%")
        print(f"Cobertura XML: {cobertura_xml}")
        if export_html:
            print(f"HTML report   : {out_dir / 'index.html'}")
    else:
        print(f"Warning: cobertura xml not found: {cobertura_xml}", file=sys.stderr)


def run_tests(build_dir: Path, enable_cuda: str, continue_on_failure: bool = False):
    """Run tests.
    Returns True if tests passed, False otherwise.
    """
    if enable_cuda == "on":
        print("Running tests with CUDA enabled...")
    else:
        print("Running tests without CUDA...")

    ctest_args = [
        "ctest",
        "--test-dir",
        str(build_dir),
        "--output-on-failure",
        "--verbose",  # Show detailed test output
    ]
    
    # Visual Studio generators need -C Debug for coverage
    cmake_cache = build_dir / "CMakeCache.txt"
    config = "Debug"
    if cmake_cache.exists():
        try:
            with open(cmake_cache, "r", encoding="utf-8") as f:
                content = f.read()
                if "Visual Studio" in content:
                    ctest_args.extend(["-C", "Debug"])
        except Exception:
            pass
    
    try:
        result = subprocess.run(ctest_args, capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return True
    except subprocess.CalledProcessError as e:
        # Print ctest output
        if hasattr(e, 'stdout') and e.stdout:
            print(e.stdout)
        if hasattr(e, 'stderr') and e.stderr:
            print(e.stderr, file=sys.stderr)
        
        # Try to run the test executable directly to get more detailed output
        test_exe = find_test_executable(build_dir)
        if test_exe:
            print("\n" + "=" * 70, file=sys.stderr)
            print("Tests failed. Running test executable directly for detailed output:", file=sys.stderr)
            print("=" * 70 + "\n", file=sys.stderr)
            try:
                # Run the test executable directly with output capture
                result = subprocess.run(
                    [str(test_exe)],
                    cwd=build_dir / "tests",
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                )
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print(result.stderr, file=sys.stderr)
                
                # Check for heap corruption or assertion failures
                output = (result.stdout or "") + (result.stderr or "")
                if "Debug Assertion Failed" in output or "heap" in output.lower() or "__acrt_first_block" in output:
                    print("\n" + "!" * 70, file=sys.stderr)
                    print("HEAP CORRUPTION DETECTED!", file=sys.stderr)
                    print("!" * 70, file=sys.stderr)
                    print(
                        "\nThis indicates a memory management bug (buffer overflow, use-after-free, etc.).\n"
                        "Debugging suggestions:\n"
                        "1. Run tests under a debugger (Visual Studio Debugger or WinDbg)\n"
                        "2. Enable AddressSanitizer (ASan) if available\n"
                        "3. Check for buffer overflows, uninitialized memory, or double-free errors\n"
                        "4. Use --continue-on-test-failure to generate coverage anyway\n",
                        file=sys.stderr,
                    )
            except subprocess.TimeoutExpired:
                print("Test executable timed out after 5 minutes.", file=sys.stderr)
            except Exception as ex:
                print(f"Could not run test executable directly: {ex}", file=sys.stderr)
        
        if continue_on_failure:
            print(
                f"\nWarning: Tests failed (exit code {e.returncode}), but continuing with coverage generation...",
                file=sys.stderr,
            )
            return False
        else:
            print(
                "\nTip: Use --continue-on-test-failure to generate coverage even if tests fail.",
                file=sys.stderr,
            )
            raise


def run_coverage(root_dir: Path, build_dir: Path, enable_cuda: str, opencppcoverage_path: Path | None, export_html: bool):
    """Run coverage analysis using gcovr (gcc/clang) or OpenCppCoverage (Windows/MSVC)."""
    if platform.system() == "Windows" and is_visual_studio_build(build_dir):
        occ = find_opencppcoverage(str(opencppcoverage_path) if opencppcoverage_path else None)
        if not occ:
            raise FileNotFoundError(
                "OpenCppCoverage.exe not found. Provide --opencppcoverage-path or install it "
                "(expected e.g. D:\\programs\\OpenCppCoverage\\OpenCppCoverage.exe)."
            )
        return run_coverage_opencppcoverage(root_dir, build_dir, occ, export_html=export_html)

    # Non-Windows/MSVC: use gcovr
    gcovr_common_args = [
        "gcovr",
        "-r",
        ".",
        "--merge-mode-functions=merge-use-line-min",
        "--exclude-unreachable-branches",
        "--exclude-noncode-lines",
        "--filter",
        "core/.",
        "--exclude",
        "subprojects/.",
        "--exclude",
        "examples/.",
        "--exclude",
        "core/src/display/.",
        "--exclude",
        "core/include/display/.",
    ]

    print("== line coverage (core, exclude display) ==")
    line_args = gcovr_common_args + [
        "--txt-metric",
        "line",
        "--fail-under-line",
        "80",
    ]
    subprocess.run(line_args, check=True, cwd=root_dir)

    print()
    print("== branch coverage (core, exclude display) ==")
    branch_args = gcovr_common_args + [
        "--txt-metric",
        "branch",
        "--fail-under-branch",
        "35",
    ]
    subprocess.run(branch_args, check=True, cwd=root_dir)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate coverage report for Hahaha project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "builddir",
        nargs="?",
        default="builddir",
        help="Build directory (default: builddir)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean the build directory before running",
    )
    parser.add_argument(
        "--cuda",
        action="store_const",
        const="on",
        dest="enable_cuda",
        help="Enable CUDA support for coverage testing",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_const",
        const="off",
        dest="enable_cuda",
        help="Disable CUDA support for coverage testing",
    )
    parser.add_argument(
        "--continue-on-test-failure",
        action="store_true",
        help="Continue with coverage generation even if tests fail",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running tests and generate coverage directly (useful when tests have bugs)",
    )
    parser.add_argument(
        "--opencppcoverage-path",
        default=None,
        help="Path to OpenCppCoverage.exe (Windows/MSVC). If omitted, will search PATH and common locations.",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Do not export HTML coverage report (still prints CLI summary and writes coverage.xml where applicable).",
    )

    args = parser.parse_args()

    root_dir = get_root_dir()
    build_dir = root_dir / args.builddir
    enable_cuda = args.enable_cuda if args.enable_cuda else "auto"

    # Change to root directory
    os.chdir(root_dir)

    # Auto-detect CUDA availability if not specified
    if enable_cuda == "auto":
        if detect_cuda():
            enable_cuda = "on"
            cuda_path = os.environ.get("CUDA_PATH", "standard location")
            print(f"CUDA detected (CUDA_PATH={cuda_path}), enabling CUDA support for coverage testing.")
        else:
            enable_cuda = "off"
            print("CUDA not detected, disabling CUDA support for coverage testing.")

    # Clean the build directory if requested
    if args.clean:
        clean_build_dir(build_dir)

    # Create build directory if it doesn't exist
    build_dir.mkdir(parents=True, exist_ok=True)

    # Configure CMake; if CUDA is on and configure fails with CUDA error, retry without CUDA
    if enable_cuda == "on":
        ok, out = configure_cmake(build_dir, enable_cuda, capture_output=True)
        if not ok:
            # Always show the error output
            print("\n" + "=" * 70, file=sys.stderr)
            print("CMake configuration failed with CUDA enabled:", file=sys.stderr)
            print("=" * 70, file=sys.stderr)
            print(out, file=sys.stderr)
            
            if "CUDA" in out or "cuda" in out or "No CUDA toolset found" in out:
                print(
                    "\nCMake failed with CUDA enabled (e.g. no CUDA toolset for Visual Studio). "
                    "Retrying with CUDA disabled for coverage.",
                    file=sys.stderr,
                )
                enable_cuda = "off"
                configure_cmake(build_dir, enable_cuda)
            else:
                print(
                    "\nCMake configuration failed. Check the error messages above.",
                    file=sys.stderr,
                )
                raise subprocess.CalledProcessError(1, "cmake", stderr=out)
    else:
        configure_cmake(build_dir, enable_cuda)

    # Build project
    build_project(build_dir)

    # Run tests (unless skipped)
    if args.skip_tests:
        print("Skipping tests as requested (--skip-tests).", file=sys.stderr)
        print("Warning: Coverage will only reflect code executed during build, not test execution.", file=sys.stderr)
    else:
        tests_passed = run_tests(build_dir, enable_cuda, args.continue_on_test_failure)
        
        if not tests_passed and not args.continue_on_test_failure:
            print(
                "\nTests failed. Options:\n"
                "  --continue-on-test-failure : Continue with coverage generation anyway\n"
                "  --skip-tests              : Skip tests entirely (faster, but less accurate coverage)",
                file=sys.stderr,
            )
            sys.exit(8)

    # Generate the coverage report
    run_coverage(
        root_dir,
        build_dir,
        enable_cuda,
        Path(args.opencppcoverage_path) if args.opencppcoverage_path else None,
        export_html=(not args.no_html),
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
