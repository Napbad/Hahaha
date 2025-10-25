cmake_minimum_required(VERSION 3.22)
project(hahaha)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

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

# cuda setup
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)
set(CUDA_PATH "/usr/local/cuda")
link_directories("${CUDA_PATH}/lib64")

file(
        GLOB HEADER_FILES
        src/*.h
)

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
    src/core/gpu/calc.cu
        src/core/gpu/calc.cuh
        src/core/compute/autodiff/ComputeGraph.h
        src/core/compute/autodiff/ComputeNode.h
        src/core/compute/autodiff/AutoDiffOp.h
        src/include/ml/common/TensorVar.h
        src/core/compute/autodiff/derivative_nodes.h
        src/core/TensorPtr.h
        src/include/ml/common/TensorVarOp.h
        src/include/ml/common/TensorVarOp.cpp
        src/include/ml/common/TensorVar.cpp
        src/core/compute/autodiff/ComputeGraph.cpp
        examples/basic_usage/basic_tensor_usage.cpp
)

file(GLOB_RECURSE HEADER_FILES
        "${CMAKE_SOURCE_DIR}/src/*.h")

# Main library
add_library(hahaha STATIC
    ${SOURCE_FILES}
    ${HEADER_FILES}
)

# Set C++ as the link language
set_target_properties(hahaha PROPERTIES
    LINKER_LANGUAGE CXX
)
# Link dependencies
target_link_libraries(hahaha
    ${CURL_LIBRARIES}
    CUDA::cudart
    CUDA::cuda_driver
)
set_property(TARGET hahaha PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Add testing support
include(CTest)
enable_testing()

# Add tests
add_subdirectory(tests)

# Add examples
 add_subdirectory(examples)