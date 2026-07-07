# Tests Directory CMakeLists.txt
# Aggregates all test subdirectories

# Enable testing
enable_testing()

include_directories(${CMAKE_SOURCE_DIR}/src/core/include)

add_executable(example_src_tensor_usage BasicOperator.cpp)