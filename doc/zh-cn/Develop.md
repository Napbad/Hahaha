## 纯开发规范文档（Core Development Guidelines）

欢迎参与 hahaha 项目！这是一个教育优先的深度学习框架，使用 C++23 和 Meson 构建。本规范确保代码一致性、可维护性和教育价值。

### 1. 关键仓库结构
- `core/include/`：公共头文件（按模块划分，如 `core/include/math/`）。
- `core/src/`：核心实现。
- `tests/`：单元测试。
- `sample/`：示例应用（如 `sample/mnist/`）。
- `doc/`：文档目录。
- `format.sh`：一键格式化脚本。

### 2. 总体开发原则
- **教育优先**：代码必须简洁、易读。每个模块/函数应有清晰意图；优先现代 C++23 特性，但避免复杂性。
- **性能与可读平衡**：核心运算（如 matmul）需优化，但必须有注释解释算法/权衡。
- **工具强制执行**：clang-format（格式）、clang-tidy（静态分析）、Meson（构建）、Valgrind/ASan（内存检查）。
- **环境要求**：GCC/Clang 支持 C++23；NVIDIA GPU 用于后期 CUDA。

### 3. 工程化命令
```bash
# 初始化并构建 (Debug)
meson setup builddir --buildtype=debug
ninja -C builddir

# 运行测试
meson test -C builddir -v

# 格式化代码
./format.sh
```

### 4. 命名规范（Naming Conventions）
严格混合 CamelCase + snake_case，避免 SCREAMING_SNAKE_CASE（全大写）用于变量/函数/常量，但头文件防护宏使用全大写。

| 项目                  | 规范                                   | 示例                                      | 说明与示例代码 |
|-----------------------|----------------------------------------|-------------------------------------------|---------------|
| 类 / 结构体 / 类型    | PascalCase（大驼峰）                   | `Matrix`, `AutogradNode`, `LinearLayer`   | 类描述核心概念；示例：`class Matrix { ... };` |
| 函数 / 方法           | camelCase（小驼峰） 或 纯小写（数学运算） | `forward()`, `add()`, `matmul()` | 以动词开头；数学基础算子使用纯小写以符合惯例 |
| 变量 / 参数 / 成员    | camelCase                             | `inputData`, `gradOutput`, `learningRate` | 描述性；示例：`size_t batchSize = 32;` |
| 私有成员变量          | camelCase + 后缀 `_`                  | `data_`, `shape_`, `requiresGrad_`       | 区分私有；示例：`std::vector<float> data_;` |
| 常量（constexpr/static）| snake_case（优先） 或 PascalCase       | `default_init_value`, `MaxBatchSize`      | 避免全大写；示例：`constexpr float default_init_value = 0.01f;` |
| 枚举 / 枚举值         | PascalCase（枚举类） + PascalCase 值   | `enum class InitializationMode { Xavier, He };` | 强类型；示例：`InitializationMode::Xavier` |
| 命名空间              | snake_case                             | `hahaha::core`, `hahaha::math`             | 层次分明；示例：`namespace hahaha::math { ... }` |
| **头文件防护宏**      | 全大写 + 项目前缀 + 路径 + `_H`        | `HAHAHA_CORE_MATRIX_H`                    | **你的要求：全大写**；示例：`#ifndef HAHAHA_CORE_MATRIX_H #define HAHAHA_CORE_MATRIX_H ... #endif` |
| 其他宏（如配置宏）    | PascalCase + 下划线                    | `Hahaha_Enable_Cuda`, `Hahaha_Debug_Log`  | 仅用于配置/调试；示例：`#ifdef Hahaha_Enable_Cuda ... #endif` |
| 文件名                | PascalCase 或 snake_case (.cpp/.h)     | `Tensor.h`, `ComputeNode.h`               | 核心类文件使用 PascalCase，工具/辅助文件使用 snake_case |
| 目录结构              | 核心在 `core/`                         | `core/include/`, `core/src/`              | 严格分离头文件与实现 |

**额外规则**：
- 名称长度：优先描述性（`computeGradientAccumulation` 而非 `gradAcc`）。
- 避免歧义：用 `input_matrix` 而非 `matrix`。
- 禁止：缩写过度、Hungarian 命名（e.g., 无 `pData`、`m_shape`）；`using namespace std;` 或其他全局 using。
- 命名空间别名：仅在 .cpp 中局部使用（e.g., `namespace core = hahaha::core;`）。

### 5. 代码风格与格式（Code Style and Formatting）
所有代码必须通过 clang-format 和 clang-tidy。

#### 5.1 .clang-format 配置（项目根目录文件，完整版）
```yaml
BasedOnStyle: LLVM
IndentWidth: 4
TabWidth: 4
UseTab: Never
ColumnLimit: 100
PointerAlignment: Left                  # int* p; 而非 int *p;
ReferenceAlignment: Left                # int& r;
SpaceAfterCStyleCast: false             # static_cast<int>(x)
AllowShortIfStatementsOnASingleLine: false
AllowShortLoopsOnASingleLine: false
AllowShortFunctionsOnASingleLine: None
AlwaysBreakTemplateDeclarations: Yes
NamespaceIndentation: All               # namespace 内缩进
DerivePointerAlignment: false
Standard: c++23
SortIncludes: CaseSensitive             # 排序 include：系统 < 项目 < 第三方
IncludeBlocks: Preserve                 # 保持 include 组别
BinPackArguments: false                 # 参数不打包在一行
BinPackParameters: false
ContinuationIndentWidth: 4
Cpp11BracedListStyle: true              # {1, 2, 3} 风格
BreakBeforeBraces: Allman               # 新行花括号
IndentCaseLabels: true
IndentPPDirectives: AfterHash           # #define 后缩进
SpaceBeforeParens: ControlStatements    # if (x) 而非 if(x)
SpaceBeforeCpp11BracedList: false
SpaceBeforeInheritanceColon: true
SpaceBeforeRangeBasedForLoopColon: true
```

- **强制执行**：pre-commit hook 或 CI 中运行 `clang-format -i **/*.cpp **/*.h`。
- **静态分析**：.clang-tidy 文件启用：
  ```
  Checks: '-*,modernize-*,performance-*,readability-*,bugprone-*,clang-analyzer-*,cppcoreguidelines-*'
  WarningsAsErrors: '*'
  ```
  运行：`clang-tidy src/**/*.cpp`。

#### 5.2 缩进、空格与行格式
- 缩进：4 空格。
- 行宽：不超过 100 列；多行参数对齐缩进。
- 花括号：新行（Allman 风格）；空函数体用 `{}` 一行。
- 空格规则：
  - 操作符周围空格：`a + b`、`if (condition)`。
  - 无空格：函数名后（`func()`）、模板角括号（`std::vector<float>`）。
- 空行：逻辑块间 1 空行；文件末无空行。
- 头文件：自包含；顺序：`<系统头> <项目头> <第三方头>`。

#### 5.3 注释规范
- 单行：`// 解释为什么，而非怎么做。`
- 多行/Doxygen：`/** @brief 简短描述。 @param x 输入。 @return 结果。 */`
- 位置：每个公共类/函数必须有 Doxygen 注释；私有/实现细节用 `//`。
- 性能注释：优化处标注（e.g., `// 使用 shared memory 避免 bank conflict`）。
- TODO：`// TODO(username): 描述任务 [date]`。

#### 5.4 错误处理与日志
- **异常处理**：核心库目前主要通过抛出 `std::invalid_argument` 或 `std::runtime_error` 处理致命错误（如形状不匹配、非法索引）。
- **Assert**：开发时用 `assert(condition && "消息")` 或自定义断言；发布时禁用。
- **日志**：使用内置日志系统（`hahaha::utils::log::Logger`）；级别：trace/debug/info/warn/error。
  示例：`Logger::getInstance().info("Epoch {}: Loss = {}", epoch, loss);`
- **自定义宏**：`Hahaha_Assert(condition, "消息")`（PascalCase + 下划线）。

### 6. 现代 C++23 使用规范
- **核心特性优先**：
  - Concepts：约束模板（e.g., `template <std::floating_point T> class Matrix;`）。
  - std::span/mdspan：用于视图（e.g., `std::span<float> data_view(data_);`）。
  - Constexpr：常量函数/变量（e.g., `constexpr size_t computeSize(size_t rows, size_t cols) { return rows * cols; }`）。
  - Ranges：数据处理（e.g., `std::ranges::fill(data_, 0.0f);`）。
- **内存管理**：std::unique_ptr/shared_ptr；避免 raw new/delete（用 std::make_unique）。
- **模板规范**：仅用于泛型（如 dtype支持）；优先 concepts 代替 SFINAE；避免模板元编程复杂性。
- **性能特性**：std::execution::par（并行算法）；cache-aligned 分配（std::aligned_alloc）。
- **禁止**：宏函数（用 constexpr）；全局变量（用 class static）；C-style 数组/指针（用 std::vector/span）。
- **兼容性**：代码必须在 GCC/Clang 上编译；无 MSVC 特定。

### 7. 测试与基准规范
- **单元测试**：GoogleTest（Meson 集成）；文件名 `test_matrix.cpp`；测试类 `MatrixTest`。
  示例：`TEST(MatrixTest, MatmulBasic) { ... EXPECT_EQ(result(0,0), expected); }`
- **覆盖率**：目标 90%；用 gcov/lcov 生成报告（CI 运行）。
- **基准测试**：Google Benchmark；文件名 `bench_matrix.cpp`。
  示例：`BENCHMARK(matrixMatmul) -> Arg(1024) -> Unit(benchmark::kMillisecond);`
- **集成测试**：端到端（如 MLP 训练循环）；用随机种子确保可复现。
- **内存检查**：每 CI 运行 Valgrind/ASan（e.g., `valgrind --leak-check=full ./tests`）。
- **可视化测试**：ImGui 相关用 mock 数据测试渲染（e.g., 模拟 loss 曲线）。

### 8. 构建与依赖规范
- **Meson 配置**（项目根目录 `meson.build`）：
  ```meson
  project('hahaha', 'cpp', version: '0.1.0', default_options: ['cpp_std=c++23'])

  # 包含子目录
  subdir('core')
  subdir('tests')
  subdir('sample')
  ```
- **模块化构建**：每个模块（如 `core`）都有自己的 `meson.build`，定义静态库并导出依赖。
- **依赖管理**：目前主要依赖通过 `extern/externlibs` 管理（如 ImGui, GLFW）或作为 Meson 依赖（如 GTest）。
- **第三方集成**：
  - GoogleTest: 用于单元测试。
  - GLFW/ImGui: 位于 `extern/externlibs`，用于可视化。

### 9. 性能优化规范
- **基准驱动**：每优化前/后运行 benchmark；目标：matmul 达 BLAS 水平。
- **CPU 优化**：SIMD（std::execution::par_unseq）；cache-friendly（行优先存储）。
- **CUDA 准备**：内核函数 camelCase（`gemmKernel()`）；用 cooperative groups/warp sync。
- **可视化性能**：ImGui 渲染限 60 FPS；避免主线程阻塞。
- **剖析工具**：Nsight Compute（CUDA）；perf/gprof（CPU）。
- **量化**：后期支持 FP16/INT8；用混合精度注释。

### 10. 常见开发坑与最佳实践
- **坑避免**：
  - 内存泄漏：每函数检查 RAII；运行 ASan。
  - 编译慢：最小化头依赖（pimpl 模式）；用 modules (C++23)。
  - 调试难：用 gdb/lldb；CUDA 用 cuda-gdb/Nsight。
  - 浮点精度：用 epsilon 检查（e.g., `EXPECT_NEAR(result, expected, 1e-5)`）。
- **最佳实践**：
  - 小步迭代：每天 commit 小功能。
  - 模块化：每个 .h/.cpp 一类；核心分离（core/ml/display）。
  - 教育注释：复杂处加伪代码解释（e.g., `// 反向传播：grad_input = grad_output * weight^T`）。
  - 版本兼容：用 #if __cplusplus >= 202300L 检查 C++23。

### 11. PR 提交清单
- [ ] 代码通过 `ninja -C builddir` 编译。
- [ ] 所有单元测试通过 `meson test -C builddir`。
- [ ] 已运行 `./format.sh` 确保风格一致。
- [ ] 关键逻辑附带教育性注释。
- [ ] 新增公开接口已添加 Doxygen 注释。
