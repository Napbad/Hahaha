#!/bin/bash

# Find and format all C/C++/Header files in src/ and tests/
find src/ tests/ -type f \( -name "*.cpp" -o -name "*.c" -o -name "*.h" -o -name "*.hpp" \) -print0 | xargs -0 clang-format -i
