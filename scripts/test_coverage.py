#!/usr/bin/env python3
# Copyright (c) 2026 Contributors of hahaha(https://github.com/jason-is-debugging/Hahaha)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Contributors:
# Jason Shen (jason.shen.gm@gmail.com) (https://github.com/jason-is-debugging)
#

"""
Test Coverage Script for Hahaha Project

Automatically builds the project with coverage instrumentation, runs tests,
and reports coverage metrics. Supports line coverage thresholds.

Usage:
    python test_coverage.py                              # Default (min 70% line coverage)
    python test_coverage.py --min-line-cov 80           # Custom threshold
    python test_coverage.py --build-dir build-cov       # Custom build directory
    python test_coverage.py --clean                     # Clean before build
    python test_coverage.py --help                      # Show full help
"""

import argparse
import os
import shutil
import subprocess
import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CoverageResult:
    """Stores coverage analysis results."""
    line_rate: float
    branch_rate: float
    covered_lines: int
    total_lines: int
    covered_branches: int
    total_branches: int
    files_covered: list[str]


class CoverageConfig:
    """Configuration for coverage analysis."""
    
    # Default minimum thresholds
    DEFAULT_MIN_LINE_COVERAGE = 70.0
    DEFAULT_MIN_BRANCH_COVERAGE = 50.0
    
    # Build directory names
    COVERAGE_BUILD_DIR = "cmake-build-coverage"
    COVERAGE_RESULT_DIR = "coverage-results"
    
    # Coverage file names
    COVERAGE_INFO_FILE = "coverage.info"
    COVERAGE_REPORT_HTML = "coverage-html"
    COVERAGE_SUMMARY_JSON = "coverage-summary.json"


def detect_compiler() -> str:
    """Detect the current compiler."""
    if sys.platform == "win32":
        # Check for MSVC via MSVC env var or cl.exe
        if os.environ.get("MSVC") or shutil.which("cl.exe"):
            return "msvc"
        # Check for MinGW/Clang
        if shutil.which("clang-cl.exe") or shutil.which("clang"):
            return "clang-cl"
        return "msvc"  # Default on Windows
    else:
        # Unix-like systems
        result = subprocess.run(
            ["cc", "--version"] if shutil.which("cc") else ["gcc", "--version"],
            capture_output=True,
            text=True
        )
        if "clang" in result.stdout.lower():
            return "clang"
        return "gcc"


def setup_coverage_build(config: CoverageConfig, clean: bool, source_dir: Path) -> bool:
    """
    Configure CMake build with coverage instrumentation.
    
    Args:
        config: Coverage configuration
        clean: Whether to clean build directory first
        source_dir: Project source directory
        
    Returns:
        True if configuration succeeded
    """
    build_dir = source_dir / config.COVERAGE_BUILD_DIR
    
    if clean and build_dir.exists():
        print(f"[INFO] Cleaning existing build directory: {build_dir}")
        shutil.rmtree(build_dir)
    
    build_dir.mkdir(parents=True, exist_ok=True)
    
    compiler = detect_compiler()
    print(f"[INFO] Detected compiler: {compiler}")
    
    # Common coverage flags
    if compiler in ("gcc", "clang"):
        coverage_flags = [
            "-fprofile-arcs",
            "-ftest-coverage",
            "-g",
            "-O0",
            "-fno-inline"
        ]
        cmake_flags = [
            f"-DCMAKE_CXX_FLAGS={' '.join(coverage_flags)}",
            "-DCMAKE_BUILD_TYPE=Debug"
        ]
    elif compiler == "clang-cl":
        coverage_flags = [
            "-fprofile-instr-generate",
            "-fcoverage-mapping",
            "-g",
            "-O0"
        ]
        cmake_flags = [
            f"-DCMAKE_CXX_FLAGS={' '.join(coverage_flags)}",
            "-DCMAKE_BUILD_TYPE=Debug"
        ]
    else:  # MSVC
        cmake_flags = [
            "/Od",           # Disable optimization
            "/Z7",           # Debug info format
            "/FS",           # Force synchronous PDB writes
            "-DCOVERAGE"     # Preprocessor define
        ]
        # MSVC uses different approach - we'll use Visual Studio instrumentation
        cmake_flags = ["-DCMAKE_BUILD_TYPE=Debug"]
    
    # Configure with CMake
    cmake_cmd = [
        "cmake",
        "-S", str(source_dir),
        "-B", str(build_dir),
        "-G", "Ninja",
        *cmake_flags
    ]
    
    # Add vcpkg toolchain if available
    vcpkg_root = os.environ.get("VCPKG_ROOT")
    if vcpkg_root and Path(vcpkg_root).exists():
        cmake_cmd.extend([
            f"-DCMAKE_TOOLCHAIN_FILE={vcpkg_root}/scripts/buildsystems/vcpkg.cmake",
            "-DVCPKG_TARGET_TRIPLET=x64-windows-static"
        ])
    
    print(f"[INFO] Configuring CMake with coverage flags...")
    print(f"       Command: {' '.join(cmake_cmd)}")
    
    try:
        result = subprocess.run(
            cmake_cmd,
            cwd=source_dir,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"[ERROR] CMake configuration failed:")
            print(result.stderr)
            return False
        print("[INFO] CMake configuration successful")
        return True
    except FileNotFoundError:
        print("[ERROR] CMake not found. Please install CMake.")
        return False


def build_project(config: CoverageConfig, source_dir: Path) -> bool:
    """
    Build the project with coverage instrumentation.
    
    Args:
        config: Coverage configuration
        source_dir: Project source directory
        
    Returns:
        True if build succeeded
    """
    build_dir = source_dir / config.COVERAGE_BUILD_DIR
    
    print(f"\n[INFO] Building project with coverage instrumentation...")
    
    try:
        # Build test targets
        result = subprocess.run(
            ["cmake", "--build", str(build_dir), "--target", "hahaha_test_core", "-j"],
            cwd=source_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"[ERROR] Build failed:")
            print(result.stdout)
            print(result.stderr)
            return False
        
        print("[INFO] Build successful")
        return True
        
    except FileNotFoundError:
        print("[ERROR] Build tool not found.")
        return False


def run_tests(config: CoverageConfig, source_dir: Path) -> bool:
    """
    Run the test suite.
    
    Args:
        config: Coverage configuration
        source_dir: Project source directory
        
    Returns:
        True if tests passed
    """
    build_dir = source_dir / config.COVERAGE_BUILD_DIR
    test_exe = build_dir / "tests" / "core" / "hahaha_test_core.exe"
    
    # Try different possible locations
    if not test_exe.exists():
        test_exe = build_dir / "tests" / "core" / "hahaha_test_core"
        if not test_exe.exists():
            test_exe = build_dir / "hahaha_test_core.exe"
            if not test_exe.exists():
                test_exe = build_dir / "hahaha_test_core"
    
    if not test_exe.exists():
        print(f"[ERROR] Test executable not found in {build_dir}")
        return False
    
    print(f"\n[INFO] Running tests: {test_exe}")
    
    try:
        result = subprocess.run(
            [str(test_exe)],
            cwd=build_dir,
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        if result.returncode != 0:
            print(f"[ERROR] Tests failed with return code {result.returncode}")
            return False
        
        print("[INFO] All tests passed")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to run tests: {e}")
        return False


def collect_coverage_gcc(config: CoverageConfig, source_dir: Path) -> Optional[CoverageResult]:
    """
    Collect coverage data using gcov/lcov (for GCC/Clang).
    
    Args:
        config: Coverage configuration
        source_dir: Project source directory
        
    Returns:
        CoverageResult or None on failure
    """
    build_dir = source_dir / config.COVERAGE_BUILD_DIR
    result_dir = source_dir / config.COVERAGE_RESULT_DIR
    result_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[INFO] Collecting coverage data (GCC/Clang)...")
    
    # Find all .gcda and .gcno files
    gcda_files = list(build_dir.rglob("*.gcda"))
    print(f"[INFO] Found {len(gcda_files)} coverage data files")
    
    if not gcda_files:
        print("[WARNING] No coverage data files found. Tests may not have executed.")
        return None
    
    # Generate lcov coverage report
    base_dir = source_dir / "src" / "core"
    
    lcov_cmd = [
        "lcov",
        "--capture",
        "--directory", str(build_dir),
        "--output-file", str(result_dir / config.COVERAGE_INFO_FILE),
        "--base-directory", str(base_dir),
        "--gcov-tool", "gcov",
        "-q"
    ]
    
    try:
        result = subprocess.run(lcov_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[WARNING] lcov capture failed: {result.stderr}")
    except FileNotFoundError:
        print("[INFO] lcov not found, using gcov directly")
        # Fallback to manual gcov processing
        pass
    
    # Generate HTML report
    genhtml_cmd = [
        "genhtml",
        str(result_dir / config.COVERAGE_INFO_FILE),
        "--output-directory", str(result_dir / config.COVERAGE_REPORT_HTML),
        "--branch-coverage",
        "--pretty-print"
    ]
    
    try:
        result = subprocess.run(genhtml_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[INFO] HTML report generated: {result_dir / config.COVERAGE_REPORT_HTML}")
        print(result.stdout)
    except FileNotFoundError:
        print("[WARNING] genhtml not found. Skipping HTML report.")
    
    # Parse coverage.info for summary
    return parse_coverage_info(result_dir / config.COVERAGE_INFO_FILE)


def parse_coverage_info(info_file: Path) -> Optional[CoverageResult]:
    """
    Parse lcov coverage.info file to extract metrics.
    
    Args:
        info_file: Path to coverage.info file
        
    Returns:
        CoverageResult with parsed metrics
    """
    if not info_file.exists():
        return None
    
    total_lines = 0
    covered_lines = 0
    total_branches = 0
    covered_branches = 0
    files_covered = []
    
    current_file = None
    
    with open(info_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("SF:"):
                current_file = line[3:]
            elif line.startswith("LH:"):
                if current_file:
                    covered_lines += int(line[3:])
            elif line.startswith("LF:"):
                if current_file:
                    total_lines += int(line[3:])
                    if int(line[3:]) > 0:
                        files_covered.append(current_file)
            elif line.startswith("BRH:"):
                covered_branches += int(line[4:])
            elif line.startswith("BRF:"):
                total_branches += int(line[4:])
            elif line.startswith("end_of_record"):
                current_file = None
    
    if total_lines == 0:
        print("[WARNING] No source lines found in coverage data")
        return None
    
    return CoverageResult(
        line_rate=(covered_lines / total_lines) * 100,
        branch_rate=(covered_branches / total_branches * 100) if total_branches > 0 else 0.0,
        covered_lines=covered_lines,
        total_lines=total_lines,
        covered_branches=covered_branches,
        total_branches=total_branches,
        files_covered=files_covered
    )


def print_coverage_report(result: CoverageResult, min_line_coverage: float) -> bool:
    """
    Print formatted coverage report and check against threshold.
    
    Args:
        result: Coverage analysis results
        min_line_coverage: Minimum required line coverage percentage
        
    Returns:
        True if coverage meets threshold
    """
    print("\n" + "=" * 60)
    print("                    COVERAGE REPORT")
    print("=" * 60)
    print(f"  Line Coverage:      {result.line_rate:.2f}%  ({result.covered_lines}/{result.total_lines})")
    print(f"  Branch Coverage:     {result.branch_rate:.2f}%  ({result.covered_branches}/{result.total_branches})")
    print(f"  Files with coverage: {len(result.files_covered)}")
    print("-" * 60)
    print(f"  Required Line Cov:   {min_line_coverage:.2f}%")
    print("=" * 60)
    
    if result.line_rate >= min_line_coverage:
        print(f"\n[PASS] Line coverage {result.line_rate:.2f}% >= {min_line_coverage:.2f}%")
        return True
    else:
        print(f"\n[FAIL] Line coverage {result.line_rate:.2f}% < {min_line_coverage:.2f}%")
        print("\n  Files with lowest coverage:")
        # Note: For detailed per-file analysis, check the HTML report
        return False


def save_summary(result: CoverageResult, config: CoverageConfig, source_dir: Path):
    """Save coverage summary as JSON."""
    summary_file = source_dir / config.COVERAGE_RESULT_DIR / config.COVERAGE_SUMMARY_JSON
    
    summary = {
        "line_coverage": result.line_rate,
        "branch_coverage": result.branch_rate,
        "covered_lines": result.covered_lines,
        "total_lines": result.total_lines,
        "covered_branches": result.covered_branches,
        "total_branches": result.total_branches,
        "files_covered": result.files_covered
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"[INFO] Coverage summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run tests with coverage analysis for Hahaha project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run with default 70%% threshold
  %(prog)s --min-line-cov 80                  # Require 80%% line coverage
  %(prog)s --build-dir my-build               # Use custom build directory
  %(prog)s --clean                            # Clean before building
  %(prog)s --min-line-cov 90 --clean          # Combined options

Requirements:
  - CMake 3.20+
  - Ninja build system
  - For GCC/Clang: gcov, lcov, genhtml (lcov package)
  - For MSVC: Visual Studio with coverage tools
        """
    )
    
    parser.add_argument(
        "--min-line-cov",
        type=float,
        default=CoverageConfig.DEFAULT_MIN_LINE_COVERAGE,
        help=f"Minimum line coverage percentage required (default: {CoverageConfig.DEFAULT_MIN_LINE_COVERAGE})"
    )
    
    parser.add_argument(
        "--build-dir",
        type=str,
        default=CoverageConfig.COVERAGE_BUILD_DIR,
        help="Build directory name (default: cmake-build-coverage)"
    )
    
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean build directory before building"
    )
    
    args = parser.parse_args()
    
    # Validate threshold
    if not 0 <= args.min_line_cov <= 100:
        print(f"[ERROR] Minimum line coverage must be between 0 and 100, got {args.min_line_cov}")
        return 1
    
    # Determine source directory
    script_dir = Path(__file__).parent
    source_dir = script_dir.parent  # Go up from scripts/ to project root
    
    print(f"[INFO] Hahaha Test Coverage Script")
    print(f"[INFO] Source directory: {source_dir}")
    print(f"[INFO] Minimum line coverage: {args.min_line_cov}%")
    
    config = CoverageConfig()
    
    # Step 1: Configure build with coverage flags
    if not setup_coverage_build(config, args.clean, source_dir):
        return 1
    
    # Step 2: Build project
    if not build_project(config, source_dir):
        return 1
    
    # Step 3: Run tests
    if not run_tests(config, source_dir):
        return 1
    
    # Step 4: Collect coverage
    result = collect_coverage_gcc(config, source_dir)
    
    if result is None:
        print("[ERROR] Failed to collect coverage data")
        return 1
    
    # Step 5: Print report and check threshold
    save_summary(result, config, source_dir)
    passed = print_coverage_report(result, args.min_line_cov)
    
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
