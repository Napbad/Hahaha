# Hahaha Coding Style

This document is the single reference for project coding standards. It is derived from `.clang-format`, `.clang-tidy`, `.pre-commit-config.yaml`, and `doc/en/Develop.md`.

---

## 1. General principles

- **Education first**: Code must be concise and readable; every module/function should have a clear intent. Use modern C++23 features but avoid unnecessary complexity.
- **Balance readability and performance**: Core operations (e.g. matmul) may be optimized but must include comments explaining the algorithm and trade-offs.
- **Tooling**: Formatting and static checks are enforced by tools; code must pass `clang-format` and `clang-tidy` (and `python3 dev/format.py`) before submission.

---

## 2. Formatting (authoritative: .clang-format)

| Item                        | Rule                                | Notes                                           |
|-----------------------------|-------------------------------------|-------------------------------------------------|
| Indent                      | 4 spaces                            | `UseTab: Never`                                 |
| Line length                 | 80 columns                          | `ColumnLimit: 80`                               |
| Braces                      | Attach (opening brace on same line) | `BreakBeforeBraces: Attach` → `void f() {`      |
| Pointer/reference alignment | Left                                | `int* p;`, `int& r;` (`PointerAlignment: Left`) |
| Binary operator line break  | Before non-assignment operators     | `BreakBeforeBinaryOperators: NonAssignment`     |
| Includes                    | Sorted and grouped                  | `SortIncludes: true`, system headers first      |
| Space after C-style cast    | Yes                                 | `SpaceAfterCStyleCast: true`                    |
| Space before assignment     | Yes                                 | `SpaceBeforeAssignmentOperators: true`          |

**Before committing, run**:

```bash
python3 dev/format.py
```

Or use pre-commit if installed: it will run `clang-format -i --style=file` on C/C++/CUDA files on commit.

---

## 3. Naming conventions

| Kind                         | Convention                                                 | Examples                                                                                                                                |
|------------------------------|------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| Classes / structs / types    | PascalCase                                                 | `TensorWrapper`, `ComputeNode`, `DeviceBuffer`                                                                                          |
| Functions / methods          | camelCase                                                  | `forward()`, `getShape()`, `cpuAdd`                                                                                                     |
| Variables / parameters       | camelCase                                                  | `inputData`, `learningRate`, `batchSize`                                                                                                |
| Public members               | camelCase                                                  | `size`                                                                                                                                  |
| Private members              | camelCase + suffix `_`                                     | `data_`, `shape_`, `requiresGrad_`                                                                                                      |
| Constants (constexpr/static) | snake_case or PascalCase                                   | `default_init_value`, `MaxBatchSize`                                                                                                    |
| Enums and enum values        | PascalCase                                                 | `enum class DeviceType { CPU, CUDA };`                                                                                                  |
| Namespaces                   | lowercase or hierarchical                                  | `hahaha::backend`, `hahaha::math`                                                                                                       |
| Header guards                | UPPER_SNAKE, project path or UUID                          | `HAHAHA_BACKEND_DEVICE_H`, `HAHAHA_BACKEND_CPU_CPU_ELEMENTWISE_KERNELS_H`, or `HAHAHA_CUDA_MEMORY_CUH_9F482603BCCC4C15924721026F992DB7` |
| File names                   | PascalCase (single-class) or snake_case (other)            | `TensorWrapper.h`, `cuda_memory.cuh`                                                                                                    |
| Macros                       | PascalCase (Function like) or ALL_UPPER_CASE (declaration) | `ComputeSth()`, `HAHAHA_ARCH_X86_64`                                                                                                           |

**Exception**: Symbols that interface with C/CUDA or third-party code (e.g. backend kernel names) may keep snake_case for consistency, e.g. `cpu_add`, `cuda_add`.

**Discouraged but allowed**:
- Hungarian notation (e.g. `pData`, `m_shape`)

**Forbidden**:
- Over-abbreviation (e.g. `gradAcc` instead of a readable name like `computeGradientAccumulation`)
- `using namespace std;` or other global `using` in headers

---

## 4. Headers and include order

- Headers must be **self-contained** (include their own dependencies; no reliance on include order).
- Include order: system / standard library → third-party → this project.
- `.clang-format` has `SortIncludes: true`; formatting will enforce sorting.

Example:

```cpp
#include <expected>
#include <span>
#include <vector>

#include "backend/Device.h"
#include "math/ds/TensorShape.h"
```

---

## 5. Comments and documentation

- **Single-line**: Explain *why*, not *how*; space after `//`.
- **Public API**: Classes and public functions must have Doxygen-style comments with at least:
  - `@brief` short description
  - `@param` for parameters (if any)
  - `@return` for return value (if any)
- **Performance**: Comment non-obvious optimizations (e.g. shared memory, alignment).
- **TODO**: Use a consistent form, e.g. `// TODO(author): description [optional date]`.

Example:

```cpp
/**
 * @brief Allocates a block of memory on the device.
 * @param size The size of the memory block in bytes.
 * @return A DeviceBuffer object representing the allocated memory.
 */
[[nodiscard]] virtual DeviceBuffer allocate(size_t size) = 0;
```

---

## 6. Error handling and return values

- **Exceptions**: Use `std::invalid_argument` or `std::runtime_error` for fatal errors (e.g. shape mismatch, invalid index).
- **Recoverable errors**: Prefer `std::expected<T, E>` (e.g. with `common::Error`) to return either a value or an error, consistent with existing backend dispatch.
- **Assertions**: Use `assert(condition && "msg")` during development; typically disabled in release.
- **Logging**: Use the project logging facility (e.g. `hahaha::utils::logger`); avoid ad hoc `printf` / `std::cerr`.

---

## 7. Modern C++ (C++23)

- **Recommended**:
  - `std::expected`, `std::span` for return values and views
  - `[[nodiscard]]` for return values that must not be ignored
  - `constexpr` for constants and compile-time logic
  - Smart pointers: `std::unique_ptr` / `std::shared_ptr`; avoid raw `new`/`delete`
- **Templates**: Use for generics (e.g. dtype, device); prefer concepts over SFINAE.
- **Forbidden**: Macros in place of functions (use constexpr/inline); global mutable state (use class static or singletons); C-style arrays (use `std::vector` / `std::span`).

---

## 8. Static analysis (clang-tidy)

The project enables clang-tidy checks including:

- **bugprone-***: Potential bugs (e.g. use-after-move, argument-comment)
- **modernize-***: Modern C++ (e.g. use-override, use-nodiscard, use-nullptr)
- **readability-***: Readability (e.g. redundant-declaration, make-member-function-const)
- **performance-***: Performance (e.g. unnecessary-value-param, move-const-arg)
- **cert-***, **cppcoreguidelines-***: Safety and core guidelines

Run clang-tidy locally or in CI and address fixable warnings.

---

## 9. Directory and repository layout

- `core/include/`: Public headers, by module (e.g. `math/`, `backend/`, `ml/`).
- `core/src/`: Implementation (`.cpp`, `.cu`).
- `tests/`: Unit tests (GoogleTest); test files should end with `Test` (e.g. `TensorWrapperTest.cpp`), with cases like `TEST(SuiteName, TestName)` or `TEST_F(FixtureName, TestName)` as appropriate.
- `examples/`: Examples and demos.
- `doc/`: Documentation (e.g. `doc/zh-cn/`, `doc/en/`).
- `dev/format.py`: Cross-platform formatting script; formats `.cpp`, `.h`, `.hpp`, `.cuh` under `core`, `examples`, `tests`.

---

## 10. File header and copyright

- New source and header files should keep the project **file header**: Copyright notice and Apache 2.0 license reference (see any file under `core/include/` or `core/src/`), then add your own attribution.
- Headers use **include guards** (`#ifndef HAHAHA_...` / `#define` / `#endif`), see §3; use a path-based or UUID-style macro name if needed to avoid clashes.

---

## 11. Pre-commit and PR checklist

- [ ] Run `python3 dev/format.py` or use editor auto-format during development.
- [ ] Build succeeds (e.g. `cmake --build builddir`) with no compile/link errors.
- [ ] Tests pass: `ctest --test-dir builddir --output-on-failure`.
- [ ] New or changed public APIs have Doxygen comments.
- [ ] Non-trivial logic has short comments for teaching and maintenance.

---

## 12. Reference files

| File                      | Purpose                                                           |
|---------------------------|-------------------------------------------------------------------|
| `.clang-format`           | Formatting rules (authoritative)                                  |
| `.clang-tidy`             | Static analysis rules                                             |
| `.pre-commit-config.yaml` | Pre-commit hooks (large files, trailing whitespace, clang-format) |
| `doc/en/Develop.md`       | Development guide (more naming and examples)                      |
| `dev/format.py`           | One-shot formatting script                                        |

In case of conflict with other docs, follow `.clang-format`, `.clang-tidy`, and the current repository layout.
