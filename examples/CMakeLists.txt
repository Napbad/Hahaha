# Add executable for the simple_nn example
add_executable(simple_nn
        main.cpp)

set(CUDA_PATH "/usr/local/cuda-13.0")
link_directories("${CUDA_PATH}/lib64")
# Link the example against the main library
target_link_libraries(simple_nn PRIVATE hahaha)

# Add include directories to find necessary headers
target_include_directories(simple_nn PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/datasets
    ${CMAKE_SOURCE_DIR}/src/models
    ${CMAKE_SOURCE_DIR}/src/optimizers
    ${CMAKE_SOURCE_DIR}/src/utils
)

message(STATUS "Examples configured successfully")


