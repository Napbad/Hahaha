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

#ifdef HAHAHA_USE_CUDA

#include "compute/kernels/CudaKernels.h"

#include <cuda_runtime.h>

#include "compute/Dispatcher.h"
#include "math/TensorInner.h"

namespace h3::core::compute::cuda {

namespace {

inline cudaError_t checkCuda(cudaError_t err) {
    return err;
}

#define CUDA_CHECK(err) checkCuda(err)

template <typename T>
void launchUnaryKernel(const T* d_src, T* d_dst, size_t n, cudaStream_t stream) {
    const int blockSize = 256;
    const int gridSize = static_cast<int>((n + blockSize - 1) / blockSize);
    unaryKernel<T, detail::SqrtFunctor<T>><<<gridSize, blockSize, 0, stream>>>(d_src, d_dst, n);
}

template <typename T, typename Functor>
void launchBinaryKernel(const T* d_src0, const T* d_src1, T* d_dst, size_t n, cudaStream_t stream) {
    const int blockSize = 256;
    const int gridSize = static_cast<int>((n + blockSize - 1) / blockSize);
    binaryKernel<T, Functor><<<gridSize, blockSize, 0, stream>>>(d_src0, d_src1, d_dst, n);
}

} // anonymous namespace

void addKernel(const math::TensorInner& src0, const math::TensorInner& src1,
              math::TensorInner& dst, KernelContext& ctx) {
    // TODO: Implement actual CUDA kernel launch with memory management
    // This is a placeholder that would need proper device memory handling
    (void)src0; (void)src1; (void)dst; (void)ctx;
}

void subKernel(const math::TensorInner& src0, const math::TensorInner& src1,
              math::TensorInner& dst, KernelContext& ctx) {
    (void)src0; (void)src1; (void)dst; (void)ctx;
}

void mulKernel(const math::TensorInner& src0, const math::TensorInner& src1,
              math::TensorInner& dst, KernelContext& ctx) {
    (void)src0; (void)src1; (void)dst; (void)ctx;
}

void divKernel(const math::TensorInner& src0, const math::TensorInner& src1,
              math::TensorInner& dst, KernelContext& ctx) {
    (void)src0; (void)src1; (void)dst; (void)ctx;
}

void sqrtKernel(const math::TensorInner& src, math::TensorInner& dst, KernelContext& ctx) {
    (void)src; (void)dst; (void)ctx;
}

void absKernel(const math::TensorInner& src, math::TensorInner& dst, KernelContext& ctx) {
    (void)src; (void)dst; (void)ctx;
}

void expKernel(const math::TensorInner& src, math::TensorInner& dst, KernelContext& ctx) {
    (void)src; (void)dst; (void)ctx;
}

void logKernel(const math::TensorInner& src, math::TensorInner& dst, KernelContext& ctx) {
    (void)src; (void)dst; (void)ctx;
}

void sinKernel(const math::TensorInner& src, math::TensorInner& dst, KernelContext& ctx) {
    (void)src; (void)dst; (void)ctx;
}

void cosKernel(const math::TensorInner& src, math::TensorInner& dst, KernelContext& ctx) {
    (void)src; (void)dst; (void)ctx;
}

void clampKernel(const math::TensorInner& src0, const math::TensorInner& src1,
                const math::TensorInner& src2, math::TensorInner& dst, KernelContext& ctx) {
    (void)src0; (void)src1; (void)src2; (void)dst; (void)ctx;
}

// ============== Registration ==============

namespace {
    struct CudaKernelRegistrar {
        CudaKernelRegistrar() {
            Dispatcher& disp = Dispatcher::instance();
            const backend::DeviceType dev = backend::DeviceType::CUDA;
            
            // Unary kernels
            for (int dtype = 1; dtype < static_cast<int>(DataType::Count); ++dtype) {
                DataType dt = static_cast<DataType>(dtype);
                disp.registerUnary(DispatchKey(Operator::Sqrt, dt, dev), sqrtKernel);
                disp.registerUnary(DispatchKey(Operator::Abs, dt, dev), absKernel);
                disp.registerUnary(DispatchKey(Operator::Exp, dt, dev), expKernel);
                disp.registerUnary(DispatchKey(Operator::Log, dt, dev), logKernel);
                disp.registerUnary(DispatchKey(Operator::Sin, dt, dev), sinKernel);
                disp.registerUnary(DispatchKey(Operator::Cos, dt, dev), cosKernel);
            }
            
            // Binary kernels
            for (int dtype = 1; dtype < static_cast<int>(DataType::Count); ++dtype) {
                DataType dt = static_cast<DataType>(dtype);
                disp.registerBinary(DispatchKey(Operator::Add, dt, dev), addKernel);
                disp.registerBinary(DispatchKey(Operator::Sub, dt, dev), subKernel);
                disp.registerBinary(DispatchKey(Operator::Mul, dt, dev), mulKernel);
                disp.registerBinary(DispatchKey(Operator::Div, dt, dev), divKernel);
            }
            
            // Ternary kernels
            for (int dtype = 1; dtype < static_cast<int>(DataType::Count); ++dtype) {
                DataType dt = static_cast<DataType>(dtype);
                disp.registerTernary(DispatchKey(Operator::Clamp, dt, dev), clampKernel);
            }
        }
    };
    
    // Static registration at startup
    static CudaKernelRegistrar registrar;
}

void registerCudaKernels() {
    // No-op: registration happens in static initializer above
}

} // namespace h3::core::compute::cuda

#endif // HAHAHA_USE_CUDA
