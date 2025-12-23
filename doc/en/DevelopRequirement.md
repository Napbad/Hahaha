# Project Development Guidelines (DevelopRequirement)

Target audience: new contributors (step-by-step onboarding) and experienced engineers (quick reference).

Purpose
This document provides a practical, repository-aligned development guide for the `hahaha` project. It is tailored to the current layout: core headers in `core/include/`, implementation in `core/src/`, and tests in `tests/`.

1. Repository layout (current)
- `core/include/`: public headers (e.g. `core/include/library.h`).
- `core/src/`: implementation files (e.g. `core/src/library.cpp`).
- `tests/`: test code and `meson.build` (e.g. `tests/main_test.cpp`).
- other folders: `sample/`, `doc/`, `example/`, `format.sh`.

Every module (for example, `core`) should be self-contained with its own `meson.build` and expose a `declare_dependency` or library target to be consumed by top-level builds. The top-level `meson.build` currently builds a static library named `hahaha` using sources under `core/src` and headers under `core/include`.

2. Required tools and environment (recommended)
- Meson + Ninja
- Clang, `clang-format`, `clang-tidy`
- A C++ compiler with C++23 support (GCC or Clang)
- GoogleTest (for unit tests), Google Benchmark (optional)
- Optional: `spdlog` for logging in examples or runtime

3. Common development commands (repo-specific)
Build and test (debug):

```bash
meson setup builddir --buildtype=debug
ninja -C builddir
meson test -C builddir
```

Reconfigure for release:

```bash
meson setup builddir --reconfigure --buildtype=release
ninja -C builddir
```

Formatting and static checks:

```bash
./format.sh
clang-format -i **/*.cpp **/*.h core/include/**/*.h
clang-tidy core/src/*.cpp -- -Icore/include -std=c++23
```

Notes: `meson setup` generates `builddir/compile_commands.json` which `clang-tidy` and many IDEs can use.

4. Naming and style (applied to this repository)
- Types / classes: PascalCase (e.g. `Matrix`, `AutogradNode`).
- Functions / methods: camelCase (e.g. `forward()`, `computeGrad()`).
- Variables / parameters / filenames: snake_case (e.g. `batch_size`, `matrix.cpp`).
- Private members: trailing underscore (e.g. `data_`).
- Header include guards: UPPER_SNAKE (e.g. `HAHAHA_LIBRARY_H`).

Enforce style using `.clang-format` at the project root and `.clang-tidy` (this repo includes a starter `.clang-tidy`).

5. Header / API rules
- Public headers must be self-contained — they should compile by themselves.
- Keep implementation out of public headers unless `inline` or `constexpr` is required.
- Use forward declarations and the PIMPL pattern to reduce header dependencies and speed compilation.

6. Error handling, asserts and logging
- Prefer explicit error returns (for example `std::expected<T, ErrorCode>` or an `ErrorCode` enum) instead of throwing exceptions widely in the library layer.
- Use `assert()` for internal consistency checks in development builds; do not rely on asserts for user-visible error handling.
- If you add logging, keep it optional and lightweight. Integrate `spdlog` through Meson when needed.

7. Testing and coverage (practical guidance)
- Current `tests/main_test.cpp` is a simple entry; expand tests to `tests/test_<module>.cpp` using GoogleTest.
- Define tests in `tests/meson.build` and wire them into `meson test`.
- For coverage run CI builds with `--coverage` and collect data with `lcov`/`genhtml`.

8. Meson specifics
- The top-level `meson.build` collects `core/include` headers and `core/src` sources and builds a static library named `hahaha`.
- New modules should follow the pattern: `module/include`, `module/src`, `module/meson.build` that declares a static library and `declare_dependency` for consumers.

9. Code review checklist (for PRs)
- Mandatory:
  - Compiles (`ninja -C builddir`)
  - Tests pass (`meson test -C builddir`)
  - `clang-format` applied
  - `clang-tidy` run with no errors
  - New features covered by unit tests

10. Commit messages and branching
- Branches: `main` (protected), `develop` (integration), `feature/<name>` (feature work).
- Commit example:

```
feat(core): add fast matrix multiply

Add SIMD-accelerated matmul with benchmark and unit tests.
```

11. Onboarding checklist (precise commands)
1. Clone:

```bash
git clone <repo-url>
cd hahaha
```

2. Install (Ubuntu example):

```bash
sudo apt update
sudo apt install -y build-essential ninja-build meson clang-format clang-tidy libgtest-dev lcov
```

3. Build and test:

```bash
meson setup builddir --buildtype=debug
ninja -C builddir
meson test -C builddir
```

4. Run formatting script:

```bash
./format.sh
```

12. Troubleshooting (repo-based)
- Missing headers: ensure `include_directories('core/include')` is present and target includes it.
- Missing sources: check `meson.build` sources collection (top-level uses `find` for `core/src`).
- Link errors: ensure libraries are exported via `declare_dependency(link_with: ...)` and consumed by targets.

13. Appendix: templates
- PR checklist (example):
  - [ ] Build passes
  - [ ] Tests pass
  - [ ] `./format.sh` run
  - [ ] Unit tests added for new behavior

- Module `meson.build` template:

```meson
project('mymodule', 'cpp')
inc = include_directories('include')
srcs = files(glob('src/*.cpp'))
lib = static_library('mymodule', srcs, include_directories: inc)
dep = declare_dependency(link_with: lib, include_directories: inc)
```

---
Next steps I can take for you:
- add a GitHub Actions workflow that runs `clang-format`, `clang-tidy`, build and tests; or
- create an English-to-Chinese sync script to keep docs in sync.
Pick one and I'll implement it.
