#!/bin/bash
#!/bin/bash
if [[ "$1" == "--check" ]]; then
  clang-format --style=file --dry-run --Werror $(find core examples tests -name "*.cpp" -o -name "*.hpp" -o -name "*.h" -o -name "*.cuh")
else
  clang-format -i --style=file $(find core examples tests -name "*.cpp" -o -name "*.hpp" -o -name "*.h" -o -name "*.cuh")
fi
