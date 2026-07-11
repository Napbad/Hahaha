//  Copyright (c) 2025-2026 Contributors of Hahaha(https://github.com/jason-is-debugging/Hahaha)
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//       https://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//  Contributors:
//  Napbad (napbad.sen@gmail.com) (https://github.com/Napbad)
//

#ifndef HAHAHA_CUDAELEMENTWISEKERNELS_CUH
#define HAHAHA_CUDAELEMENTWISEKERNELS_CUH

#include "compute/operator_executor/impl/detail/ElementwiseCommon.h"
#include "compute/operator_executor/impl/detail/OperatorFunctors.h"
#include "defines.h"

namespace h3::core::compute::detail {

struct CudaTensorView {
    const void* base;
    SizeT byteOffset;
    SizeT rank;
    SizeT shape[kMaxElementwiseRank];
    SizeT stride[kMaxElementwiseRank];
};

__device__ inline SizeT broadcastElementOffset(const CudaTensorView& view,
                                               const SizeT outRank,
                                               const SizeT* outShape,
                                               const SizeT linear) {
    SizeT rem = linear;
    SizeT elemOff = 0;
    for (SizeT k = 0; k < outRank; ++k) {
        const SizeT dimIdx = outRank - 1 - k;
        const SizeT c = rem % outShape[dimIdx];
        rem /= outShape[dimIdx];
        if (k < view.rank) {
            const SizeT tDim = view.rank - 1 - k;
            const SizeT idx = view.shape[tDim] == 1 ? 0 : c;
            elemOff += idx * view.stride[tDim];
        }
    }
    return elemOff;
}

template <typename T, typename Functor>
__global__ void unaryElementwiseKernel(T* output,
                                       const CudaTensorView input,
                                       const CudaTensorView result,
                                       const SizeT* outShape,
                                       const SizeT total) {
    const SizeT idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const SizeT inElem = broadcastElementOffset(input, result.rank, outShape, idx);
    const SizeT outElem = broadcastElementOffset(result, result.rank, outShape, idx);
    const T* inPtr = reinterpret_cast<const T*>(static_cast<const char*>(input.base)
        + input.byteOffset) + inElem;
    T* outPtr = reinterpret_cast<T*>(static_cast<char*>(output) + result.byteOffset) + outElem;
    *outPtr = Functor::template apply<T>(*inPtr);
}

template <typename T, typename Functor>
__global__ void binaryElementwiseKernel(T* output,
                                         const CudaTensorView lhs,
                                         const CudaTensorView rhs,
                                         const CudaTensorView result,
                                         const SizeT* outShape,
                                         const SizeT total) {
    const SizeT idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const SizeT leftElem = broadcastElementOffset(lhs, result.rank, outShape, idx);
    const SizeT rightElem = broadcastElementOffset(rhs, result.rank, outShape, idx);
    const SizeT outElem = broadcastElementOffset(result, result.rank, outShape, idx);
    const T* leftPtr = reinterpret_cast<const T*>(static_cast<const char*>(lhs.base)
        + lhs.byteOffset) + leftElem;
    const T* rightPtr = reinterpret_cast<const T*>(static_cast<const char*>(rhs.base)
        + rhs.byteOffset) + rightElem;
    T* outPtr = reinterpret_cast<T*>(static_cast<char*>(output) + result.byteOffset) + outElem;
    *outPtr = Functor::template apply<T>(*leftPtr, *rightPtr);
}

template <typename T, typename Functor>
__global__ void ternaryElementwiseKernel(T* output,
                                         const CudaTensorView a,
                                         const CudaTensorView b,
                                         const CudaTensorView c,
                                         const CudaTensorView result,
                                         const SizeT* outShape,
                                         const SizeT total) {
    const SizeT idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const SizeT aElem = broadcastElementOffset(a, result.rank, outShape, idx);
    const SizeT bElem = broadcastElementOffset(b, result.rank, outShape, idx);
    const SizeT cElem = broadcastElementOffset(c, result.rank, outShape, idx);
    const SizeT outElem = broadcastElementOffset(result, result.rank, outShape, idx);
    const T* aPtr = reinterpret_cast<const T*>(static_cast<const char*>(a.base) + a.byteOffset) + aElem;
    const T* bPtr = reinterpret_cast<const T*>(static_cast<const char*>(b.base) + b.byteOffset) + bElem;
    const T* cPtr = reinterpret_cast<const T*>(static_cast<const char*>(c.base) + c.byteOffset) + cElem;
    T* outPtr = reinterpret_cast<T*>(static_cast<char*>(output) + result.byteOffset) + outElem;
    *outPtr = Functor::template apply<T>(*aPtr, *bPtr, *cPtr);
}

} // namespace h3::core::compute::detail

#endif // HAHAHA_CUDAELEMENTWISEKERNELS_CUH
