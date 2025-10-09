# Hahaha ML Framework Coding Style Guide

This document outlines the coding style and conventions to be used in the hahaha-dev project. Consistency in code style is crucial for readability and maintainability.

## General Principles

*   Clarity and simplicity are paramount.
*   Code should be self-documenting as much as possible.
*   Follow established best practices for C++ and Python.

## Naming Conventions

We use camelCase as the primary naming convention.

### C++

*   **Classes/Structs/Typedefs/Enums:** `UpperCamelCase`
    *   Example: `class Tensor`, `struct TrainStatistics`
*   **Functions/Methods:** `lowerCamelCase`
    *   Example: `void calculateLoss()`, `int getValue()`
*   **Variables (including member variables):** `lowerCamelCase`
    *   Example: `int numEpochs`, `float learningRate`
    *   Private member variables will be suffixed with `_`. Example: `learningRate_`.
*   **Constants and Enum Values:** `UPPER_CASE_WITH_UNDERSCORES`
    *   Example: `const int MAX_ITERATIONS = 100;`, `enum class Status { OK, FAILED };`
*   **Macros:** `UPPER_CASE_WITH_UNDERSCORES` (use sparingly)
    *   Example: `#define MY_MACRO`
*   **Filenames:** `UpperCamelCase.h`, `UpperCamelCase.cpp` for files containing a primary class. `snake_case.h`, `snake_case.cpp` for utility files with free functions.

### Python

*   **Classes:** `UpperCamelCase`
*   **Functions/Methods/Variables:** `lowerCamelCase` for consistency with the C++ codebase.
*   **Constants:** `UPPER_CASE_WITH_UNDERSCORES`
*   **Modules/Packages:** `lowercase_with_underscores`

## Formatting

### C++

We use `clang-format` to enforce a consistent format. The configuration will be in the `.clang-format` file in the root of the project. Key style points include:

*   **Indent:** 4 spaces.
*   **Brace Style:** Allman style (opening braces on new lines).
*   **Column Limit:** 120 characters.
*   **Pointer/Reference Alignment:** Left-aligned (e.g., `int* p`).
*   **Includes:** Sorted and grouped.

### Python

We will use an automated formatter to ensure uniformity for Python code.

## Comments

*   Use `//` for single-line comments.
*   Use `/* ... */` for multi-line comments.
*   Use Doxygen-style comments for documenting interfaces (`.h` files).
