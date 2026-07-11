# Core Examples CMakeLists.txt
# Aggregates core example subdirectories

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../src/core/include)

# Include tensor_usage examples
add_subdirectory(tensor_usage)
