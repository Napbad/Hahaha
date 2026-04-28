# Hahaha [![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Hahaha is an **education-first** numerical computing and machine learning library written in **C++23**.
It aims to provide a **clear, inspectable implementation** of tensors, automatic differentiation, and training
loops, together with **visualization tooling** so learners can *see* what happens during training.

## What this project is for

- **Learning**: read the code and understand how tensors, broadcasting, and autograd work end-to-end.
- **Teaching / Demos**: run examples that visualize the training process (loss curves, parameter updates, etc.).
- **Small experiments**: simple models and datasets (e.g. MNIST) without depending on a large framework.

## Key features

- **Tensor core**: `Tensor` / `TensorWrapper` with shape (`TensorShape`), stride (`TensorStride`), broadcasting,
  reshape, transpose, and basic math.
- **Autograd + compute graph**: define-by-run graph building via `compute/graph/*` and backward propagation
  with unit tests.
- **Broadcasting as a first-class op**: explicit `Broadcast` node with correct gradient reduction.
- **Visualization + educational UX**:
  - ImGui-based visualizer and demos under `examples/` to show training dynamics.
  - Code is intentionally written to be readable and traceable, with tests as executable documentation.
- **Tooling**: CMake + vcpkg, GoogleTest; format code with `python3 dev/format.py`.

## Build and run

### 1. Prerequisites

- **Compiler**: C++23 (GCC 13+ or Clang 16+; on Windows, MSVC)
- **Build**: [CMake](https://cmake.org/) 3.20+ and [Ninja](https://ninja-build.org/)
- **vcpkg**: Use a **system or repo-local vcpkg**; this repo does not embed or manage vcpkg itself.  
  - Install: <https://vcpkg.io/en/docs/getting-started.html>  
  - Dependencies (`gtest`, `glfw3`, `imgui`) are declared in the root `vcpkg.json` and installed automatically by vcpkg in **manifest mode** when you configure.

### 2. Build (Windows and Linux)

vcpkg recommends **not** setting the toolchain inside `CMakeLists.txt`; pass it via the command line or CMake Presets. This project uses **CMake Presets** for vcpkg.

Set the vcpkg path first, using either:

- **Recommended**: Set the `VCPKG_ROOT` environment variable (e.g. repo-local `<repo>/vcpkg/vcpkg_root` or your system vcpkg path), then use a preset.
- Or pass the toolchain when configuring: `-DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake`

#### Windows (PowerShell):

```powershell
# Using CMake Presets (recommended): set VCPKG_ROOT first
$env:VCPKG_ROOT = "D:\projects\Hahaha\vcpkg\vcpkg_root"   # or your vcpkg path

cmake --preset vcpkg-windows
cmake --build --preset vcpkg-windows
ctest --test-dir cmake-build-vcpkg-windows --output-on-failure
```

Or specify the toolchain directly (without preset):

```powershell
cmake -S . -B builddir -G Ninja -DCMAKE_BUILD_TYPE=Debug `
  -DCMAKE_TOOLCHAIN_FILE=<place to Hahaha>/vcpkg/vcpkg_root/scripts/buildsystems/vcpkg.cmake
cmake --build builddir
```

#### Linux (bash):

On Linux you need a C++23 compiler (GCC 13+ or Clang 16+), Ninja, and vcpkg. To build the ImGui/GLFW visualizer example, add `-DHAHAHA_DISPLAY=ON` when configuring (vcpkg provides GLFW/OpenGL; no extra system packages required if using vcpkg).

```bash
# One-time: clone and bootstrap vcpkg (or use system vcpkg)
python3 dev/dev_env_setup.py

# Set VCPKG_ROOT to your vcpkg root (repo-local or system)
export VCPKG_ROOT="$(pwd)/vcpkg/vcpkg_root"   # or e.g. /opt/vcpkg

cmake --preset vcpkg-linux
cmake --build --preset vcpkg-linux
ctest --test-dir cmake-build-vcpkg-linux --output-on-failure
```

Or without presets:

```bash
cmake -S . -B builddir -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
cmake --build builddir
```

#### Using the provided build scripts:

We've included automated build scripts to simplify the build process:

For PowerShell:
```powershell
.\build.ps1
```

For Command Prompt:
```cmd
build.bat
```

These scripts will:
- Check for required tools (CMake, Ninja)
- Locate your vcpkg installation
- Configure the project with the correct toolchain
- Build the project
- Run tests

### 3. Run examples

After building, run from your build directory (e.g. `builddir`, `cmake-build-vcpkg-windows`, or `cmake-build-vcpkg-linux`):

```bash
# Basic tensor usage
./builddir/examples/hahaha_example_tensor_basic_usage

# Autograd demo
./builddir/examples/hahaha_example_autograd

# ML training demo (CLI)
./builddir/examples/hahaha_example_ml_basic_usage

# Visualization demo (requires GLFW/OpenGL)
./builddir/examples/hahaha_example_ml_visualizer
```

## Troubleshooting

If you encounter build issues:

1. Make sure vcpkg dependencies are installed:
   ```bash
   cd <path-to-vcpkg>
   ./vcpkg install
   ```

2. Ensure you're using the correct triplet (x64-windows-static for the library, x64-windows for the host):
   ```bash
   ./vcpkg install --triplet=x64-windows-static
   ```

3. If you get linker errors, ensure the runtime library settings match:
   - For static linking, use MultiThreaded runtime
   - For dynamic linking, use MultiThreadedDLL runtime

## Where to look (recommended reading order)

- **Tensor & shape/stride**: `core/include/math/TensorWrapper.h`, `core/include/math/ds/*`
- **Compute graph & autograd**: `core/include/compute/graph/*`, especially `ComputeNode` and `compute_funs/*`
- **Visualization**: `core/src/display/*` and `examples/ml_visualizer/*`
- **Tests as documentation**: `tests/core/*` (broadcast + autograd tests show expected behavior)

## Minimal usage example

```cpp
#include "public/Tensor.h"

using h3::Tensor;
using h3::math::NestedData;

int main() {
    Tensor<float> a(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    Tensor<float> b(NestedData<float>{{10.0f, 20.0f}, {30.0f, 40.0f}});
    a.setRequiresGrad(true);
    b.setRequiresGrad(true);

    auto c = a * b + 2.0f;
    c.backward();

    // Inspect gradients
    // a.grad(), b.grad()
    return 0;
}
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a pull request

See [how-to-contribute](https://github.com/JiansongShen/HahahaDevDocument/blob/main/src/en/developers/how-to-contribute.md) for details.

Please ensure your code follows the project's coding standards by running `python3 dev/format.py` before submitting.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.