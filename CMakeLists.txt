cmake_minimum_required(VERSION 3.22)
project(hiahiahia)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(src/include)

# Recursively find all header and source files
file(GLOB_RECURSE HEADER_FILES 
    "src/include/*.h"
    "src/include/*/*.h"
    "src/include/*/*/*.h"
    "src/include/*/*/*/*.h"
)

file(GLOB_RECURSE SOURCE_FILES 
    "src/src/*.cpp"
    "src/src/*/*.cpp"
    "src/src/*/*/*.cpp"
    "src/src/*/*/*/*.cpp"
)

# Find libcurl
find_package(CURL REQUIRED)
include_directories(${CURL_INCLUDE_DIRS})

# Main library
add_library(hiahiahia STATIC
    ${HEADER_FILES}
    ${SOURCE_FILES}
)

# Set C++ as the link language
set_target_properties(hiahiahia PROPERTIES
    LINKER_LANGUAGE CXX
)

add_subdirectory(tests)
