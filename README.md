# Hahaha[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

A lightweight numerical computing and machine learning library with core components implemented in C++. Designed for learners and small-scale projects, Hahaha provides efficient tensor operations, automatic differentiation foundation, and essential ML components.

## 🚀 Features

- **Tensor Module**: Comprehensive tensor data structure with shape management (`TensorShape`), stride control (`TensorStride`), and nested data handling (`NestedData`) for efficient numerical operations
- **Automatic Differentiation Ready**: Foundation for computational graphs through `ComputeGraph.h`
- **High-Performance Computing**: Optimized for modern C++23 standards with CUDA 13.0 acceleration support
- **Complete Development Environment**: Docker-based development container with `.devcontainer` configuration for consistent setup across platforms
- **Professional Tooling**: Integrated code formatting (`clang-format`), static analysis (`clang-tidy`), and unit testing (Google Test)
- **Example Applications**: Complete MNIST handwritten digit recognition example demonstrating network layers, activation functions, optimizers, and training workflow

## 🛠️ Installation & Setup

### Prerequisites

- Docker (for containerized development)
- Git
- CMake ≥ 3.10
- NVIDIA GPU with CUDA 13.0 support (optional, for GPU acceleration)

### Quick Start with Docker

```bash
# Clone the repository
git clone https://github.com/Napbad/Hahaha.git && cd Hahaha

# Build the development image
docker build -t hahaha-dev .

# Run the development container
docker run --gpus all -it --rm -v $(pwd):/workspace hahaha-dev

# Compile the project
mkdir build && cd build
cmake .. && make
```

### VS Code Development

The project includes full `.devcontainer` support:
1. Open this folder in VS Code
2. Click "Reopen in Container" when prompted
3. The development environment will be automatically configured

## 📚 Usage Examples

### Basic Tensor Operations

```cpp
#include "math/ds/TensorData.h"

// Create tensors with initializer lists
hahaha::math::TensorData<int> tensor1({{1, 2}, {3, 4}});
hahaha::math::TensorData<double> tensor2({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});

// Single value tensor
hahaha::math::TensorData<float> scalar(42.0f);
```

### Running Tests

```bash
# After building
./build/tests/core/math/ds/TensorDataTest
```

## 🧪 Testing

All core components are thoroughly tested with Google Test framework:

```bash
# Run all tests
make test

# Or run specific test executable
cd build/tests/core/math/ds/
./TensorDataTest
```

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a pull request

Please ensure your code follows the project's coding standards by running `format.sh` before submitting.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.