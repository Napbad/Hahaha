cmake_minimum_required(VERSION 3.22)
project(hiahiahia)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(src)
add_library(hiahiahia STATIC
        src/include/hiahiahia/common/ds/Tensor.h
        src/src/test.cpp
        src/include/hiahiahia/common/defines.h
        src/include/hiahiahia/common/ds/TensorShape.h
        src/include/hiahiahia/utils/memory.h
        src/include/hiahiahia/common/ds/TensorStride.h
        src/include/hiahiahia/common/Res.h
)

add_subdirectory(tests)