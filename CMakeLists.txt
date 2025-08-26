cmake_minimum_required(VERSION 3.22)
project(hiahiahia)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(src)
add_library(hiahiahia STATIC
        src/include/ml/utils/memory.h
        src/src/ml/model/knn/KNN.cpp
        src/include/ml/model/knn/KNN.h
        src/include/ml/model/Model.h
        src/include/ml/model/linear_regression/LinearRegression.h
        src/include/ml/loss/Loss.h
        src/include/ml/utils/device/Device.h
        src/include/ml/utils/device/GPUDevice.h
        src/include/ml/utils/device/CPUDevice.h
        src/include/common/ds/vec.h
        src/include/common/ds/res.h
        src/include/common/ds/opt.h
        src/include/common/ds/stack.h
        src/include/common/ds/queue.h
        src/include/common/ds/dqueue.h
        src/include/common/ds/map.h
        src/include/common/ds/set.h
        src/include/common/ds/arr.h
        src/include/common/ds/hashset.h
        src/include/common/ds/hashmap.h
        src/include/ml/common/Tensor.h
        src/include/common/ds/str.h
        src/include/common/ds/strbuf.h
        src/include/common/ds/bintree.h
        src/include/common/defines/h3defs.h
        src/include/common/Res.h
        src/include/common/Error.h
        src/include/common/util/Log.h
        src/include/common/algo/StrFinder.h
        src/include/common/util/mem/Allocator.h
        src/include/common/ds/list.h
        src/include/ml/utils/device/Device.h
        src/include/ml/utils/dataset/Dataset.h
        src/include/ml/utils/dataset/DatasetIterator.h
        src/include/ml/utils/dataset/DatasetLoader.h
        src/include/ml/utils/dataset/DatasetType.h
        src/include/ml/utils/dataset/Sample.h
        src/include/common/util/commonUtil.h
        src/include/common/util/StringStream.h
)

add_subdirectory(tests)