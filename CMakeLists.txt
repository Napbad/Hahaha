cmake_minimum_required(VERSION 3.22)
project(hahaha)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add include directories
include_directories(${CMAKE_SOURCE_DIR}/src/include)

# Find required packages
find_package(CURL REQUIRED)
include_directories(${CURL_INCLUDE_DIRS})

# Collect source files
file(GLOB_RECURSE HEADER_FILES 
    "${CMAKE_SOURCE_DIR}/src/include/*.h"
)

file(GLOB_RECURSE SOURCE_FILES 
    "${CMAKE_SOURCE_DIR}/src/src/*.cpp"
)

# Print debug information to check if files are found
message(STATUS "Found source files:")
foreach(src ${SOURCE_FILES})
    message(STATUS "  ${src}")
endforeach()

# Main library
add_library(hahaha STATIC
    ${SOURCE_FILES}
    ${HEADER_FILES}
        src/include/ml/util/dataset/MNIST.h
        src/include/ml/common/Array.h
        src/src/common/error.cpp
        src/src/common/ds/Str.cpp
        src/include/ml/common/trainStatistics.h
)

# Set C++ as the link language
set_target_properties(hahaha PROPERTIES
    LINKER_LANGUAGE CXX
)

# Link dependencies
target_link_libraries(hahaha
    ${CURL_LIBRARIES}
)

# Add tests
add_subdirectory(tests)