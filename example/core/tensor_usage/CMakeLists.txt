# Tensor Usage Example CMakeLists.txt

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../src/core/include)

add_executable(example_src_tensor_usage BasicOperator.cpp)
target_link_libraries(example_src_tensor_usage PRIVATE hahaha_core)