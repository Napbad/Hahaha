cmake_minimum_required(VERSION 3.22)
project(hiahiahia_dev_tests_unit)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include_directories(../../src)

# Enable testing
enable_testing()

# Try to find GTest, but don't fail if not found
find_package(GTest QUIET)

# Create the test executable
add_executable(haihiaiha_dev_tests_unit_exe
    hiahiahia/common/defines_test.cpp
    unittest_main.cpp
        hiahiahia/common/ds/Tensor_test.cpp
        hiahiahia/common/ds/TensorShape_test.cpp
)

# Link with Google Test
if(GTest_FOUND)
    target_link_libraries(haihiaiha_dev_tests_unit_exe GTest::GTest GTest::Main)
else()
    # Link with the built-from-source libraries
    target_link_libraries(haihiaiha_dev_tests_unit_exe gtest gtest_main)
    
    # Add the Google Test include path
    target_include_directories(haihiaiha_dev_tests_unit_exe PRIVATE 
        /usr/src/googletest/googletest/include
        /usr/src/googletest/googlemock/include
    )
endif()

# Add test
add_test(NAME hiahiahia_unit_tests COMMAND haihiaiha_dev_tests_unit_exe)

message(STATUS "Unit tests configured successfully")
