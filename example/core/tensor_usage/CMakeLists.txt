# Tensor Usage Example CMakeLists.txt

include_directories(${CMAKE_SOURCE_DIR}/src/core/include)

add_executable(example_core_tensor_usage init_and_assign.cpp)
target_link_libraries(example_core_tensor_usage PRIVATE hahaha_core)