cmake_minimum_required(VERSION 3.22)
project(hahaha_dev_tests_unit)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# cuda config
set(CUDA_PATH "/usr/local/cuda-13.0")
if(NOT EXISTS "${CUDA_PATH}/lib64/libcudadevrt.a")
    message(FATAL_ERROR "not found libcudadevrt.a, please ensure the path exists libcudadevrt : ${CUDA_PATH}/lib64")
endif()
if(NOT EXISTS "${CUDA_PATH}/lib64/libcudart_static.a")
    message(FATAL_ERROR "not found libcudart_static.a, please ensure the path exists libcudart_static : ${CUDA_PATH}/lib64")
endif()
link_directories("${CUDA_PATH}/lib64")
set(CMAKE_CUDA_ARCHITECTURES all)

# test config
enable_testing()
find_package(GTest REQUIRED)
include(GoogleTest)

# file include and collect
file(GLOB_RECURSE TEST_SOURCE_FILES
    archtest_main.cpp
)

include_directories(
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/datasets
    ${CMAKE_SOURCE_DIR}/src/models
    ${CMAKE_SOURCE_DIR}/src/optimizers
    ${CMAKE_SOURCE_DIR}/src/utils
)

# test config
# Create the test executable
add_executable(hahaha_dev_tests_arch_exe
        ${TEST_SOURCE_FILES}
        basic/gpu/BasicGPUTest.cpp
        basic/gpu/testDefine.cu
        basic/gpu/testDefine.cuh
)
# Link libraries
target_link_libraries(hahaha_dev_tests_arch_exe PRIVATE
        hahaha

        "${CUDA_PATH}/lib64/libcudart_static.a"
        "${CUDA_PATH}/lib64/libcudadevrt.a"

        GTest::gtest
        GTest::gtest_main)

target_compile_options(hahaha_dev_tests_arch_exe PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --threads 0
        >
)

set_property(TARGET hahaha_dev_tests_arch_exe PROPERTY CUDA_SEPARABLE_COMPILATION ON)


# Add test
gtest_discover_tests(hahaha_dev_tests_arch_exe)

message(STATUS "Architecture tests configured successfully")