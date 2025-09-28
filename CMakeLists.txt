cmake_minimum_required(VERSION 3.22)
project(hahaha)

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

# Main library
add_library(hahaha STATIC
    ${SOURCE_FILES}
    ${HEADER_FILES}
        tests/unittest/hiahiahia/ml/util/dataset/DatasetDownloaderTest.cpp
        src/include/ml/util/compute/ComputeGraphNode.h
        src/include/ml/util/compute/ComputeFn.h
        src/include/common/util/vectorize/archDefines.h
        src/include/common/util/vectorize/Vectorize.h
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

# Print debug information
message(STATUS "Include directories:")
get_property(dirs DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
    message(STATUS "  ${dir}")
endforeach()

message(STATUS "Source files:")
foreach(src ${SOURCE_FILES})
    message(STATUS "  ${src}")
endforeach()

message(STATUS "Header files:")
foreach(hdr ${HEADER_FILES})
    message(STATUS "  ${hdr}")
endforeach()
