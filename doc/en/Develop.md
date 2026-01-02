## Core Development Guidelines

Welcome to the Hahaha project! This is an education-first deep learning framework built with C++23 and Meson. These guidelines ensure code consistency, maintainability, and educational value.

### 1. Repository Structure
- `core/include/`: Public headers (organized by module, e.g., `core/include/math/`).
- `core/src/`: Core implementation files.
- `tests/`: Unit tests.
- `sample/`: Example applications (e.g., `sample/mnist/`).
- `doc/`: Documentation.
- `format.sh`: One-click formatting script.

### 2. General Principles
- **Education First**: Code must be concise and readable. Every module/function should have a clear intent. Use modern C++23 features but avoid unnecessary complexity.
- **Balance Performance and Readability**: Core operations (like matmul) should be optimized but must include comments explaining the algorithm and trade-offs.
- **Tooling Enforcement**: Use `clang-format` (formatting), `clang-tidy` (static analysis), Meson (build), and Valgrind/ASan (memory checks).
- **Environment**: GCC/Clang with C++23 support; NVIDIA GPU for later CUDA stages.

### 3. Engineering Commands
```bash
# Initialize and build (Debug)
meson setup builddir --buildtype=debug
ninja -C builddir

# Run tests
meson test -C builddir -v

# Format code
./format.sh
```

### 4. Naming Conventions
Strictly mix CamelCase + snake_case. Avoid `SCREAMING_SNAKE_CASE` for variables/functions/constants, except for header guards.

| Item | Convention | Example | Description & Code Snippet |
|------|------------|---------|----------------------------|
| Classes / Structs / Types | PascalCase | `Matrix`, `AutogradNode`, `LinearLayer` | Core concepts; e.g., `class Matrix { ... };` |
| Functions / Methods | camelCase or lowercase (math) | `forward()`, `add()`, `matmul()` | Action-oriented; basic math operators use lowercase by convention |
| Variables / Parameters | camelCase | `inputData`, `gradOutput`, `learningRate` | Descriptive; e.g., `size_t batchSize = 32;` |
| Private Members | camelCase + suffix `_` | `data_`, `shape_`, `requiresGrad_` | Distinguish private state; e.g., `std::vector<float> data_;` |
| Constants (constexpr/static) | snake_case (pref) or PascalCase | `default_init_value`, `MaxBatchSize` | Avoid all-caps; e.g., `constexpr float default_init_value = 0.01f;` |
| Enums / Enum Values | PascalCase | `enum class InitializationMode { Xavier, He };` | Strongly typed; e.g., `InitializationMode::Xavier` |
| Namespaces | snake_case | `hahaha::core`, `hahaha::math` | Hierarchical; e.g., `namespace hahaha::math { ... }` |
| **Header Guards** | UPPER_SNAKE | `HAHAHA_MATH_MATRIX_H` | **Requirement: All-caps**; e.g., `#ifndef HAHAHA_MATH_MATRIX_H ... #endif` |
| Other Macros | PascalCase + underscore | `Hahaha_Enable_Cuda`, `Hahaha_Debug_Log` | Configuration/Debug only; e.g., `#ifdef Hahaha_Enable_Cuda ... #endif` |
| Filenames | PascalCase or snake_case | `Tensor.h`, `ComputeNode.h` | Core classes use PascalCase; utility files use snake_case |
| Directory Structure | Modules in `core/` | `core/include/`, `core/src/` | Strict separation of headers and implementation |

**Additional Rules**:
- Name Length: Prefer descriptive names (`computeGradientAccumulation` over `gradAcc`).
- Avoid Ambiguity: Use `input_matrix` over `matrix`.
- Prohibited: Over-abbreviation, Hungarian notation (no `pData`, `m_shape`); `using namespace std;` or other global usings.
- Namespace Aliases: Use only locally within `.cpp` files (e.g., `namespace core = hahaha::core;`).

### 5. Code Style and Formatting
All code must pass `clang-format` and `clang-tidy`.

#### 5.1 .clang-format Configuration (Summary)
- Indent Width: 4 spaces.
- Column Limit: 100 characters.
- Brace Breaking: Allman style (new line for braces).
- Pointer/Reference Alignment: Left (`int* p;`, `int& r;`).

#### 5.2 Indentation and Spacing
- Indent: 4 spaces.
- Line Width: Max 100 columns.
- Braces: Allman style (new line); empty bodies use `{}`.
- Spacing:
  - Around operators: `a + b`, `if (condition)`.
  - No space after function names (`func()`) or in template brackets (`std::vector<float>`).
- Blank Lines: 1 blank line between logical blocks; no blank lines at the end of files.
- Headers: Self-contained; Order: `<System>`, `<Project>`, `<Third-party>`.

#### 5.3 Documentation
- Single-line: `// Explain WHY, not HOW.`
- Multi-line/Doxygen: `/** @brief Short desc. @param x Input. @return Result. */`
- Location: Every public class/function must have Doxygen comments.
- Performance: Note optimizations (e.g., `// Use shared memory to avoid bank conflicts`).
- TODO: `// TODO(username): description [date]`.

#### 5.4 Error Handling and Logging
- **Exception Handling**: The core library primarily uses `std::invalid_argument` or `std::runtime_error` for fatal errors (e.g., shape mismatch, invalid indices).
- **Assert**: Use `assert(condition && "msg")` or custom assertions during development; disabled in release.
- **Logging**: Use the project's built-in system (`hahaha::utils::log::Logger`).
  Example: `Logger::getInstance().info("Epoch {}: Loss = {}", epoch, loss);`
- **Custom Macros**: `Hahaha_Assert(condition, "msg")` (PascalCase + underscore).

### 6. Modern C++23 Usage
- **Core Features**:
  - Concepts: Constrain templates (e.g., `template <std::floating_point T> class Matrix;`).
  - std::span / mdspan: For views (e.g., `std::span<float> data_view(data_);`).
  - Constexpr: For constant functions/variables.
  - Ranges: For data processing (e.g., `std::ranges::fill(data_, 0.0f);`).
- **Memory Management**: Use `std::unique_ptr`/`std::shared_ptr`. Avoid raw `new`/`delete`.
- **Templates**: Use only for generics (e.g., dtype support). Prefer concepts over SFINAE.
- **Prohibited**: Macro functions (use constexpr); Global variables (use class static); C-style arrays (use std::vector/span).

### 7. Testing and Benchmarking
- **Unit Tests**: GoogleTest (Meson integrated); e.g., `TensorWrapperTest.cpp`.
- **Coverage**: Target 90%; generated using `gcov/lcov`.
- **Benchmarks**: Google Benchmark; e.g., `bench_matrix.cpp`.
- **Integration Tests**: End-to-end (e.g., MLP training loops).
- **Memory Checks**: Run ASan/Valgrind in CI.

### 8. Build and Dependencies
- **Meson Configuration** (`meson.build` in project root):
  ```meson
  project('hahaha', 'cpp', version: '0.1.0', default_options: ['cpp_std=c++23'])

  # Include subdirectories
  subdir('core')
  subdir('tests')
  subdir('sample')
  ```
- **Modular Build**: Each module (e.g., `core`) has its own `meson.build` defining static libraries and exporting dependencies.
- **Dependency Management**: Dependencies are managed via `extern/externlibs` (e.g., ImGui, GLFW) or as Meson dependencies (e.g., GTest).
- **Third-party Integration**:
  - GoogleTest: For unit testing.
  - GLFW/ImGui: Located in `extern/externlibs`, used for visualization.

### 9. Performance Optimization
- **Benchmark-driven**: Run benchmarks before and after every optimization.
- **CPU Optimization**: SIMD; cache-friendly (row-major storage).
- **CUDA Preparation**: Kernel functions use camelCase (`gemmKernel()`).
- **Profiling**: Nsight Compute (CUDA); perf/gprof (CPU).

### 10. Best Practices
- **Small Iterations**: Commit small features daily.
- **Modularity**: One class per `.h`/`.cpp`. Separate core logic (`core`, `ml`, `display`).
- **Educational Comments**: Explain complex logic with pseudo-code (e.g., `// Backprop: gradInput = gradOutput * weight^T`).

### 11. PR Checklist
- [ ] Code compiles via `ninja -C builddir`.
- [ ] All unit tests pass via `meson test -C builddir`.
- [ ] `./format.sh` has been run for consistent styling.
- [ ] Key logic includes educational comments.
- [ ] Doxygen comments added for new public APIs.

