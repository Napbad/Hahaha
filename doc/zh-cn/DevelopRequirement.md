# 项目开发注意与规范（DevelopRequirement）

> 目标读者：初学者（逐步上手）与有经验的工程师（快速定位）。

本文件是对项目内部开发规范的全面、可操作指南，适用于小型到大型 C++ 项目（特别是使用 Meson / C++23 的项目）。它在可读性、可维护性与性能之间提供明确权衡，并给出实践模板、检查清单和常见问题解决步骤。

## 一、总体原则
- 教育优先：代码要清晰、可读、有注释，重点解释“为什么”而非“怎么做”。
- 可维护性优先于过早优化：仅在测得瓶颈处优化并标注理由与基准结果。
- 一致性大于个人风格：团队必须遵守统一的格式和检查工具（`clang-format`、`clang-tidy`、`spdlog` 等）。
- 自动化：编译、测试、格式化与静态分析应在本地和 CI 中自动运行。

## 二、仓库布局与模块约定
- 顶层目录建议：
  - `include/`：公共头文件（按命名空间放子目录）。
  - `src/`：实现源文件与私有实现。
  - `tests/`：单元测试与集成测试。
  - `benchmarks/`：基准测试（Google Benchmark）。
  - `examples/`：小型示例程序（教学用途）。
  - `doc/`：文档、开发规范、架构说明。
  - `third_party/` 或 `subprojects/`：第三方依赖（若不使用包管理）。

- 每个库模块（例如 `core`, `nn`, `viz`）都应是自包含的子目录，包含自己的 `meson.build`。

## 三、命名与编码风格（快速参考）
- 类/类型：PascalCase，例如 `Matrix`, `AutogradNode`。
- 函数/方法：camelCase，例如 `forward()`, `computeGrad()`。
- 变量/参数/文件名：snake_case，例如 `batch_size`, `matrix.cpp`。
- 私有成员：以下划线结尾，例如 `data_`。
- 宏: Pascal_Case, 例如 `H3_Linux_Specific`.
- 头文件宏：全大写（项目+路径+`_H`），例如 `HAHAHA_CORE_MATRIX_H`。

（详细的 `.clang-format` 配置和 `clang-tidy` checks` 在项目根的示例文件中定义并在 CI 中强制执行。）

## 四、代码组织与 API 设计
- 头文件（`.h`/`.hpp`）应只包含接口声明与必要的 inline/constexpr；实现放 `.cpp`。
- 头文件自包含：每个头都能单独编译通过。
- 避免在头文件中包含不必要的实现依赖（使用前向声明、PIMPL 如需隐藏实现以加快编译）。
- 公共 API 要保持稳定；向后兼容优先于非破坏性重构。

## 五、现代 C++ 使用准则（C++23）
- 优先使用：`std::span`/`mdspan`（视图）、`constexpr`、`concepts`、`ranges`。
- 内存管理：优先 `std::unique_ptr`/`std::shared_ptr`；避免裸 `new`/`delete`。
- 并发：使用标准库并行算法与线程库；对高性能路径使用显式并行化并记录设计理由。
- 模板：用 `concepts` 替代复杂 SFINAE，提高可读性。

## 六、错误处理与日志
- 返回错误：优先 `std::expected<T, ErrorCode>`（或 `tl::expected`）而非抛异常，函数声明尽量 `noexcept`。
- 断言：开发时使用 `assert()`，生产构建去除，错误用返回值传播。
- 日志：集成 `spdlog` 或等效库，按模块配置日志级别，运行时可调。

## 七、测试、基准与工具链
- 单元测试：GoogleTest（每个模块对应 `test_xxx.cpp`）。
- 基准：Google Benchmark，基准必须附带数据规模参数与运行说明。
- 覆盖率：目标 >= 90%（核心库），使用 `gcov/lcov` 并在 CI 中上传报告。
- 内存安全：CI 运行 ASan/UBSan 与 Valgrind（可选较慢但深度检查）。

## 八、构建与依赖管理（Meson）
- Meson 配置示例应放在项目根的 `meson.build`，并提供 `meson_options.txt`：如 `enable_cuda`。
- 建议使用包管理器（`vcpkg` / system packages）来管理依赖；在 devcontainer 或 CI 中记录安装步骤。

示例（简化）：
```meson
project('hahaha', 'cpp', default_options: ['cpp_std=c++23'])
spdlog_dep = dependency('spdlog', required: true)
inc = include_directories('include')
core = static_library('hahaha_core', ['src/core/matrix.cpp'], include_directories: inc, dependencies: [spdlog_dep])
```

## 九、CI / Git 流程建议
- 分支策略：
  - `main`：随时可发布（受保护分支）。
  - `develop`：日常集成分支（可选）。
  - feature 分支：`feature/<短描述>`。
# 项目开发注意与规范（DevelopRequirement）

> 目标：为 `hahaha` 仓库提供一份与当前代码库结构一致、对初学者友好且对有经验工程师有价值的详尽开发规范。

本文档针对仓库当前结构（核心代码在 `core/`，头文件在 `core/include`，实现位于 `core/src`，测试在 `tests/`）给出具体操作、模板和检查清单。请在变更后同步更新本文件。

## 一、关键仓库结构（基于当前 repo）
- `core/include/`：公共头文件（例如 `core/include/library.h`）。
- `core/src/`：核心实现（例如 `core/src/library.cpp`）。
- `tests/`：测试代码与 `meson.build`（例如 `tests/main_test.cpp`）。
- 其它：`sample/`（例如 `sample/mnist/`）、`doc/`、`example/`、`format.sh`（格式化脚本）等。

基础约定（必须遵守）：每个子模块（例如 `core`）应包含自己的 `meson.build`，并导出 declare_dependency（或 library）以便顶层 `meson.build` 使用。目前顶层静态库名为 `hahaha`（见根 `meson.build`）。

## 二、必备开发工具与环境
- 要求（推荐）：
  - Meson + Ninja（用于构建）
  - Clang/Clang-Format/Clang-Tidy（格式与静态检查）
  - GCC/Clang 支持 C++23
  - GoogleTest（测试）、Google Benchmark（基准，可选）
  - spdlog（运行时日志，可选）

- 本仓库提供 `format.sh`（顶层）来统一运行格式化与基本检查，建议在本地开发时先运行该脚本。

## 三、工程化命令（针对当前仓库）
推荐的本地构建与测试步骤：

```bash
# 第一次：初始化构建目录（Debug）
meson setup builddir --buildtype=debug

# 构建
ninja -C builddir

# 运行测试（meson test 更标准）
meson test -C builddir

# 或直接运行生成的可执行
./builddir/<path-to-exe>
```

如果需要释放构建：

```bash
meson setup builddir --reconfigure --buildtype=release
ninja -C builddir
```

格式化与静态检查（推荐步骤）：

```bash
# 运行仓库内的格式化脚本（项目自带）
./format.sh

# 手动运行 clang-format
clang-format -i **/*.cpp **/*.h core/include/**/*.h

# 运行 clang-tidy（需要先生成 compile_commands.json，meson 会产生）
clang-tidy core/src/*.cpp -- -Icore/include -std=c++23
```

注：`meson setup` 在项目根会产生 `builddir/compile_commands.json`（或在现有 builddir 下），方便 `clang-tidy` 与 IDE 使用。

## 四、命名与代码风格（结合项目现状）
- 类/结构体：PascalCase，例如 `Matrix`、`AutogradNode`。
- 函数/方法：camelCase，例如 `forward()`、`computeGrad()`。
- 变量/参数/文件：snake_case，例如 `batch_size`、`matrix.cpp`。
- 私有成员：末尾 `_`，例如 `data_`。
- 头文件保护宏：全大写（项目+路径），例如 `HAHAHA_LIBRARY_H`（当前 `core/include/library.h` 已采用此风格）。

样式工具（必须在 CI 与本地运行）：
- `.clang-format`：项目根应存放配置（若无，请创建，并在 CI 中运行 `clang-format -i`）。
- `.clang-tidy`：启用 `modernize-*`, `performance-*`, `bugprone-*` 等检查，并将严重问题作为错误。

示例 `.clang-tidy`（建议）:

```yaml
Checks: '-*,modernize-*,performance-*,readability-*,bugprone-*,clang-analyzer-*,cppcoreguidelines-*'
WarningsAsErrors: '*'
```

## 五、接口与头文件管理（仓库实践）
- 头文件自包含：每个头应能独立被编译（使用前向声明代替不必要的包含）。
- 公共头放 `core/include`；实现放 `core/src`。例如 `core/include/library.h`、`core/src/library.cpp`。
- 避免将实现代码放入公共头（除非函数是 `inline` 或 `constexpr`）。

## 六、错误处理、断言与日志（实践建议）
- 函数错误返回：优先使用 `std::expected` 或自定义 `ErrorCode` 类型以显式返回错误，避免在库层大量抛出异常以便嵌入式/轻量使用。
- 断言：`assert()` 用于不应发生的内部错误；公共 API 对错误使用返回值处理。
- 日志：若引入 `spdlog`，请在 `meson.build` 中声明依赖，并按模块初始化 logger（可在 `main` 或测试中设置）。

## 七、测试、基准与覆盖（针对当前仓库）
- 当前 `tests/main_test.cpp` 为项目测试入口，建议将单元测试拆分为 `tests/test_<module>.cpp` 并使用 GoogleTest 框架。
- 在 `meson.build` 的 `tests/` 子目录中声明测试 target（当前仓库已包含 `subdir('tests')`）。
- 覆盖率：在 CI 上用 `--coverage` 编译并用 `lcov` 收集报告；核心库目标覆盖率 >= 80%-90%（根据模块重要性调整）。

示例 Meson 测试段（tests/meson.build）：

```meson
gtest = dependency('gtest', required: true)
test_exe = executable('hahaha_tests', ['tests/main_test.cpp'], include_directories: include_directories('core/include'), dependencies: [hahaha])
test('unit_tests', test_exe)
```

## 八、构建系统细节（Meson）
- 顶层 `meson.build`（当前仓库）会：
  - 收集 `core/include` 的头文件和 `core/src` 的源文件，
  - 构建静态库 `hahaha`，并导出 `hahaha` 作为 `declare_dependency`，供 `tests` 与其他子项目使用。

- 当新增模块（例如 `nn`）时，建议：
  - 在 `nn/` 下创建 `include/` 与 `src/`，并在 `nn/meson.build` 中定义静态库与 `declare_dependency`，顶层 `meson.build` 使用 `subdir('nn')`。

## 九、代码审查（Code Review）清单（适用于 PR）
- 必需项：
  - 编译通过（`ninja -C builddir`）
  - 所有测试通过（`meson test -C builddir`）
  - `clang-format` 已应用（可在 PR 中强制格式化）
  - `clang-tidy` 无错误（警告视情况处理）
  - 新增行为有测试覆盖

- 评审重点：API 设计、内存/线程安全、边界条件测试、兼容性影响与性能说明（若相关）。

## 十、提交规范与分支策略（建议）
- 分支：`main`（受保护）、`develop`（日常合并）、`feature/<name>`（新功能）。
- 提交信息格式（示例）：

```
feat(core): add fast matrix multiply

Add SIMD-accelerated matmul. Includes benchmark and unit tests.
```

## 十一、上手清单（针对新成员，精确命令）
1. 克隆仓库：

```bash
git clone <repo-url>
cd hahaha
```

2. 安装依赖（示例，Ubuntu）：

```bash
sudo apt update
sudo apt install -y build-essential ninja-build meson clang-format clang-tidy libgtest-dev lcov
# 对于 spdlog / gtest 等可使用系统包或 vcpkg
```

3. 本地构建并运行测试：

```bash
meson setup builddir --buildtype=debug
ninja -C builddir
meson test -C builddir
```

4. 运行格式化脚本：

```bash
./format.sh
```

5. 提交与 PR：按提交规范编写消息，发起 `feature/*` 分支的 PR。

## 十二、常见问题与快速排查（基于仓库）
- Meson 找不到头文件：确认 `include_directories('core/include')` 在 `meson.build` 中声明，并在 target 中引用。
- 源文件未编译：检查 `meson.build` 中 `sources` 收集逻辑（当前根 `meson.build` 使用 `find` 搜索 `core/src`）。
- 运行时未链接 symbol：确认在 `meson.build` 中库被正确 link（`declare_dependency(link_with: hahaha_lib)`）。

## 十三、附录：模板片段与脚手架
- PR 检查清单：
  - [ ] 编译通过
  - [ ] 测试通过
  - [ ] 已运行 `./format.sh`
  - [ ] 新增代码附带单元测试

- 新模块 `meson.build` 模板：

```meson
project('mymodule', 'cpp')
inc = include_directories('include')
srcs = files(glob('src/*.cpp'))
lib = static_library('mymodule', srcs, include_directories: inc)
dep = declare_dependency(link_with: lib, include_directories: inc)
```

---
