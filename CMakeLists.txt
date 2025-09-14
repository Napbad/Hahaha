cmake_minimum_required(VERSION 3.22)
project(hiahiahia)

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
add_library(hiahiahia STATIC
    ${SOURCE_FILES}
    ${HEADER_FILES}
)

# Set C++ as the link language
set_target_properties(hiahiahia PROPERTIES
    LINKER_LANGUAGE CXX
)

# Link dependencies
target_link_libraries(hiahiahia
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
