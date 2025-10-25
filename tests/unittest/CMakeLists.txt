cmake_minimum_required(VERSION 3.22)
project(hahaha_dev_tests_unit)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable testing
enable_testing()

file(GLOB_RECURSE TEST_SOURCE_FILES
    "core/*.cpp"
    "datasets/*.cpp"
    "models/*.cpp"
    "optimizers/*.cpp"
        "hahaha/*.cpp"
)

# Find Google Test package
find_package(GTest REQUIRED)
include(GoogleTest)

# Create the test executable
add_executable(hahaha_dev_tests_unit_exe
    ${TEST_SOURCE_FILES}
)

target_include_directories(hahaha_dev_tests_unit_exe PUBLIC
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/datasets
    ${CMAKE_SOURCE_DIR}/src/models
    ${CMAKE_SOURCE_DIR}/src/optimizers
    ${CMAKE_SOURCE_DIR}/src/utils
)

# Link libraries
target_link_libraries(hahaha_dev_tests_unit_exe PRIVATE hahaha GTest::gtest GTest::gtest_main)

# Add test
gtest_discover_tests(hahaha_dev_tests_unit_exe)

message(STATUS "Unit tests configured successfully")