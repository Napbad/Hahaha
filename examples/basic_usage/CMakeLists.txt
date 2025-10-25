add_executable(basic_usage
        main.cpp)

target_link_libraries(basic_usage PRIVATE hahaha)

target_include_directories(basic_usage PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/datasets
    ${CMAKE_SOURCE_DIR}/src/models
    ${CMAKE_SOURCE_DIR}/src/optimizers
    ${CMAKE_SOURCE_DIR}/src/utils
)

# Add executable for the simple_nn example
add_executable(basic_example
        main.cpp)

set(CUDA_PATH "/usr/local/cuda-13.0")
link_directories("${CUDA_PATH}/lib64")
# Link the example against the main library
target_link_libraries(basic_example PRIVATE hahaha)

# Add include directories to find necessary headers
target_include_directories(basic_example PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/datasets
    ${CMAKE_SOURCE_DIR}/src/models
    ${CMAKE_SOURCE_DIR}/src/optimizers
    ${CMAKE_SOURCE_DIR}/src/utils
)

message(STATUS "Examples configured successfully")


