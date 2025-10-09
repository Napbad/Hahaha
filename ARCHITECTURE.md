# Hahaha ML Framework Architecture

This document outlines the architectural principles and structure of the Hahaha ML Framework.

## 1. Project Structure

The project structure will be reorganized for better separation of concerns and scalability.

### Current Structure Issues
- Redundant `src/src` directory.
- Inconsistent naming for test directories (e.g., `hiahiahia`).
- Header and source files are mixed in some places.

### Proposed Structure
```
hahaha-dev/
├── cmake/                # CMake modules
├── docs/                 # Documentation
│   ├── ARCHITECTURE.md
│   └── CODING_STYLE.md
├── examples/             # Example usage of the framework
├── src/                  # Source code
│   ├── hahaha/
│   │   ├── common/         # Core utilities (data structures, error handling)
│   │   ├── math/           # Math operations
│   │   └── ml/             # Machine learning components
│   └── include/            # Public headers (deprecated, to be merged)
├── tests/                # Tests
│   ├── hahaha/
│   │   ├── common/
│   │   └── ml/
│   └── unittest_main.cpp
├── .clang-format
├── CMakeLists.txt
└── ...
```
- The core library code will reside in `src/hahaha`.
- Public headers will be organized within the `src/hahaha` directory structure.
- The redundant `src/include` will be removed.
- Test directory names will be made consistent.

## 2. Namespace Strategy

- All framework code will reside under the `hahaha` namespace.
- Sub-namespaces like `hahaha::ml`, `hahaha::common`, and `hahaha::math` will be used to organize components logically.

## 3. Class Design and Design Patterns

### Core Components
- **Tensor**: This is a fundamental class. Its interface will be reviewed for clarity and completeness.
- **Data Loading (`Dataset`, `DataLoader`)**: A factory pattern will be used to create different types of datasets (`CSVDataset`, `MNIST`, etc.). This will make it easier to add new data sources.
- **Model**: The `Model` base class will be a pure abstract class (interface) defining the contract for all models (e.g., `train`, `predict`).
- **Error Handling**: The `Error` class hierarchy will be used consistently for robust error reporting. Exceptions will be used for unrecoverable errors.
- **Memory Management**: `std::unique_ptr` and `std::shared_ptr` will be used to manage object lifetimes and prevent memory leaks, especially for objects created on the heap.

### Design Goals
- **Modularity**: Components should be loosely coupled.
- **Extensibility**: It should be easy to add new models, layers, optimizers, and datasets.
- **Testability**: All components should be designed to be easily testable in isolation.

## 4. Build System (CMake)

- The `CMakeLists.txt` files will be cleaned up and modernized.
- Clear separation between library and executable targets.
- Test targets will be properly configured.
- Modern CMake practices will be applied.
