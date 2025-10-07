# Hahaha

A lightweight numerical computing and machine learning basic library with core components implemented in C++. It is designed to provide simple support for tensor operations, loss functions, and optimizers, making it suitable for learning purposes and small-scale project development.

## Core Features

• Tensor: Optimized tensor data structure and basic operation implementation, providing low-level support for numerical computing.

• Machine Learning Components: Includes basic loss functions, optimizers, and some fundamental but flexible implementations of basic models.

• Development Environment Support: Offers .devcontainer, Dockerfile, and prepare_dev.sh script to quickly configure a consistent development environment.

• Code Style Tools: Integrates .clang-format and format.sh for easy unification of code style.

## Environment Preparation

### Dependency Requirements

• C++ compiler (supporting C++23 or higher)

• CMake (version 3.10 or higher)

• Git

### Quick Configuration Steps

1.Clone the repository:
```bash
git clone https://github.com/Napbad/Hahaha.git && cd Hahaha
```
2. Choose one of the following methods to configure the environment:

◦ Script Configuration: Run bash prepare_dev.sh to automatically prepare basic docker development dependencies(container).

◦ Docker Configuration: Run docker build -t hahaha-dev . to build the Docker development image.

◦ IDE Container Configuration: Directly load the project container configuration via IDEs that support .devcontainer (e.g., VS Code or Clion).

## Basic Usage

1. Compile the project:
```bash
mkdir build && cd build
cmake ..
make
```
2. Run tests: After compilation, execute test files (e.g., Linear Regression tests) in the build/tests/ directory.

3. Extended development: Reference modules such as tensors and loss functions in the src/ directory to implement custom numerical computing or machine learning functions.

License

This project is open-source under the Apache License 2.0. See the LICENSE file in the project root directory for details.
