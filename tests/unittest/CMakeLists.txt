cmake_minimum_required(VERSION 3.22)
project(hahaha_dev_tests_unit)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include_directories(../../src)

# Enable testing
enable_testing()

# Find Google Test package
find_package(GTest REQUIRED)

# Create the test executable
add_executable(haihiaiha_dev_tests_unit_exe
        unittest_main.cpp
)

# Link with Google Test
target_link_libraries(haihiaiha_dev_tests_unit_exe GTest::GTest GTest::Main pthread)

# Add test
add_test(NAME hahaha_unit_tests COMMAND haihiaiha_dev_tests_unit_exe)

message(STATUS "Unit tests configured successfully")
