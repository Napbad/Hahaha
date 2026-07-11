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

#ifndef HAHAHA_CUDA_KERNELS_H
#define HAHAHA_CUDA_KERNELS_H

#include "compute/DispatchKey.h"
#include "compute/KernelContext.h"
#include "defines.h"

namespace h3::core::math {
class TensorInner;
}

namespace h3::core::compute {

// Forward declaration
class Dispatcher;

#ifdef HAHAHA_USE_CUDA
namespace cuda {

namespace detail {

template <typename T>
struct AddFunctor {
    __host__ __device__ T apply(T a, T b) { return a + b; }
};

template <typename T>
struct SubFunctor {
    __host__ __device__ T apply(T a, T b) { return a - b; }
};

template <typename T>
struct MulFunctor {
    __host__ __device__ T apply(T a, T b) { return a * b; }
};

template <typename T>
struct DivFunctor {
    __host__ __device__ T apply(T a, T b) { return a / b; }
};

template <typename T>
struct SqrtFunctor {
    __host__ __device__ T apply(T x) { return sqrt(x); }
};

template <typename T>
struct AbsFunctor {
    __host__ __device__ T apply(T x) { return abs(x); }
};

template <typename T>
struct ExpFunctor {
    __host__ __device__ T apply(T x) { return exp(x); }
};

template <typename T>
struct LogFunctor {
    __host__ __device__ T apply(T x) { return log(x); }
};

template <typename T>
struct SinFunctor {
    __host__ __device__ T apply(T x) { return sin(x); }
};

template <typename T>
struct CosFunctor {
    __host__ __device__ T apply(T x) { return cos(x); }
};

template <typename T>
struct ClampFunctor {
    __host__ __device__ T apply(T x, T lo, T hi) {
        T v = x < lo ? lo : x;
        return v > hi ? hi : v;
    }
};

// ============== CUDA Kernels ==============

template <typename T, typename Functor>
__global__ void unaryKernel(const T* __restrict__ src, T* dst, size_t n) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < n; i += stride) {
        dst[i] = Functor::apply(src[i]);
    }
}

template <typename T, typename Functor>
__global__ void binaryKernel(const T* __restrict__ src0, const T* __restrict__ src1, 
                             T* dst, size_t n) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < n; i += stride) {
        dst[i] = Functor::apply(src0[i], src1[i]);
    }
}

template <typename T>
__global__ void clampKernel(const T* __restrict__ src, const T* __restrict__ lo, 
                             const T* __restrict__ hi, T* dst, size_t n) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;
    const T lo_val = *lo;
    const T hi_val = *hi;
    for (size_t i = idx; i < n; i += stride) {
        const T v = src[i];
        dst[i] = v < lo_val ? lo_val : (v > hi_val ? hi_val : v);
    }
}

} // namespace detail

// ============== Kernel Function Declarations ==============

void addKernel(const math::TensorInner& src0, const math::TensorInner& src1,
              math::TensorInner& dst, ComputeContext& ctx);

void subKernel(const math::TensorInner& src0, const math::TensorInner& src1,
              math::TensorInner& dst, ComputeContext& ctx);

void mulKernel(const math::TensorInner& src0, const math::TensorInner& src1,
              math::TensorInner& dst, ComputeContext& ctx);

void divKernel(const math::TensorInner& src0, const math::TensorInner& src1,
              math::TensorInner& dst, ComputeContext& ctx);

void sqrtKernel(const math::TensorInner& src, math::TensorInner& dst, ComputeContext& ctx);

void absKernel(const math::TensorInner& src, math::TensorInner& dst, ComputeContext& ctx);

void expKernel(const math::TensorInner& src, math::TensorInner& dst, ComputeContext& ctx);

void logKernel(const math::TensorInner& src, math::TensorInner& dst, ComputeContext& ctx);

void sinKernel(const math::TensorInner& src, math::TensorInner& dst, ComputeContext& ctx);

void cosKernel(const math::TensorInner& src, math::TensorInner& dst, ComputeContext& ctx);

void clampKernel(const math::TensorInner& src0, const math::TensorInner& src1,
                const math::TensorInner& src2, math::TensorInner& dst, ComputeContext& ctx);

} // namespace cuda

/// Initialize CUDA kernel registrations.
void registerCudaKernels();

#endif // HAHAHA_USE_CUDA

} // namespace h3::core::compute

#endif // HAHAHA_CUDA_KERNELS_H
