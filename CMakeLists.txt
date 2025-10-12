cmake_minimum_required(VERSION 3.22)
project(hahaha)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/src/
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/datasets
    ${CMAKE_SOURCE_DIR}/src/models
    ${CMAKE_SOURCE_DIR}/src/optimizers
    ${CMAKE_SOURCE_DIR}/src/utils
)

# Find required packages
find_package(CURL REQUIRED)
include_directories(${CURL_INCLUDE_DIRS})

# Source files
set(SOURCE_FILES
    src/core/Error.cpp
    src/core/ds/String.cpp
    src/utils/io/file/fileUtil.cpp
    src/utils/io/net/HttpClient.cpp
    src/utils/io/net/HttpRequest.cpp
    src/utils/io/net/HttpResponse.cpp
    src/models/knn/KNN.cpp
    src/models/linear_regression/LinearRegression.cpp
    src/datasets/dataset/DatasetDownloader.cpp
        src/optimizers/AdamOptimizer.h
)

file(GLOB_RECURSE HEADER_FILES
        "${CMAKE_SOURCE_DIR}/src/*.h")

# Main library
add_library(hahaha STATIC
    ${HEADER_FILES}
    ${SOURCE_FILES}
)

# Set C++ as the link language
set_target_properties(hahaha PROPERTIES
    LINKER_LANGUAGE CXX
)

# Link dependencies
target_link_libraries(hahaha
    ${CURL_LIBRARIES}
)

# Add testing support
include(CTest)
enable_testing()

# Add tests
add_subdirectory(tests)

# Add examples
add_subdirectory(examples)