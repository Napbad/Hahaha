cmake_minimum_required(VERSION 3.22)
project(hahaha_dev_tests_unit)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUDA_PATH "/usr/local/cuda-13.0")
link_directories("${CUDA_PATH}/lib64")
# Enable testing
enable_testing()

file(GLOB_RECURSE TEST_SOURCE_FILES
    archtest_main.cpp
)

# Find Google Test package
find_package(GTest REQUIRED)
include(GoogleTest)

# Create the test executable
add_executable(hahaha_dev_tests_arch_exe
    ${TEST_SOURCE_FILES}
        basic/gpu/BasicGPUTest.cpp
        basic/gpu/testDefine.cu
)

target_include_directories(hahaha_dev_tests_arch_exe PUBLIC
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/datasets
    ${CMAKE_SOURCE_DIR}/src/models
    ${CMAKE_SOURCE_DIR}/src/optimizers
    ${CMAKE_SOURCE_DIR}/src/utils
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
        -arch=sm_89
        --threads 0
        >
)

set_property(TARGET hahaha_dev_tests_arch_exe PROPERTY CUDA_SEPARABLE_COMPILATION ON)


# Add test
gtest_discover_tests(hahaha_dev_tests_arch_exe)

message(STATUS "Architecture tests configured successfully")