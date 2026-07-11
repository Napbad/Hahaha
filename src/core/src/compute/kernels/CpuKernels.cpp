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

#include "compute/kernels/CpuKernels.h"

#include "compute/Dispatcher.h"

namespace h3::core::compute::cpu {

// ============== Kernel Implementations ==============

void addKernel(const math::TensorInner& src0, const math::TensorInner& src1,
              math::TensorInner& dst, ComputeContext& ctx) {
    detail::dispatchBinary<detail::AddFunctor>(src0, src1, dst, ctx.dtype);
}

void subKernel(const math::TensorInner& src0, const math::TensorInner& src1,
              math::TensorInner& dst, ComputeContext& ctx) {
    detail::dispatchBinary<detail::SubFunctor>(src0, src1, dst, ctx.dtype);
}

void mulKernel(const math::TensorInner& src0, const math::TensorInner& src1,
              math::TensorInner& dst, ComputeContext& ctx) {
    detail::dispatchBinary<detail::MulFunctor>(src0, src1, dst, ctx.dtype);
}

void divKernel(const math::TensorInner& src0, const math::TensorInner& src1,
              math::TensorInner& dst, ComputeContext& ctx) {
    detail::dispatchBinary<detail::DivFunctor>(src0, src1, dst, ctx.dtype);
}

void maxKernel(const math::TensorInner& src0, const math::TensorInner& src1,
              math::TensorInner& dst, ComputeContext& ctx) {
    detail::dispatchBinary<detail::MaxFunctor>(src0, src1, dst, ctx.dtype);
}

void minKernel(const math::TensorInner& src0, const math::TensorInner& src1,
              math::TensorInner& dst, ComputeContext& ctx) {
    detail::dispatchBinary<detail::MinFunctor>(src0, src1, dst, ctx.dtype);
}

void modKernel(const math::TensorInner& src0, const math::TensorInner& src1,
              math::TensorInner& dst, ComputeContext& ctx) {
    detail::dispatchBinary<detail::ModFunctor>(src0, src1, dst, ctx.dtype);
}

void powKernel(const math::TensorInner& src0, const math::TensorInner& src1,
              math::TensorInner& dst, ComputeContext& ctx) {
    detail::dispatchBinary<detail::PowFunctor>(src0, src1, dst, ctx.dtype);
}

void sqrtKernel(const math::TensorInner& src, math::TensorInner& dst, ComputeContext& ctx) {
    detail::dispatchUnary<detail::SqrtFunctor>(src, dst, ctx.dtype);
}

void absKernel(const math::TensorInner& src, math::TensorInner& dst, ComputeContext& ctx) {
    detail::dispatchUnary<detail::AbsFunctor>(src, dst, ctx.dtype);
}

void signKernel(const math::TensorInner& src, math::TensorInner& dst, ComputeContext& ctx) {
    detail::dispatchUnary<detail::SignFunctor>(src, dst, ctx.dtype);
}

void expKernel(const math::TensorInner& src, math::TensorInner& dst, ComputeContext& ctx) {
    detail::dispatchUnary<detail::ExpFunctor>(src, dst, ctx.dtype);
}

void logKernel(const math::TensorInner& src, math::TensorInner& dst, ComputeContext& ctx) {
    detail::dispatchUnary<detail::LogFunctor>(src, dst, ctx.dtype);
}

void sinKernel(const math::TensorInner& src, math::TensorInner& dst, ComputeContext& ctx) {
    detail::dispatchUnary<detail::SinFunctor>(src, dst, ctx.dtype);
}

void cosKernel(const math::TensorInner& src, math::TensorInner& dst, ComputeContext& ctx) {
    detail::dispatchUnary<detail::CosFunctor>(src, dst, ctx.dtype);
}

void ceilKernel(const math::TensorInner& src, math::TensorInner& dst, ComputeContext& ctx) {
    detail::dispatchUnary<detail::CeilFunctor>(src, dst, ctx.dtype);
}

void floorKernel(const math::TensorInner& src, math::TensorInner& dst, ComputeContext& ctx) {
    detail::dispatchUnary<detail::FloorFunctor>(src, dst, ctx.dtype);
}

void clampKernel(const math::TensorInner& src0, const math::TensorInner& src1,
                const math::TensorInner& src2, math::TensorInner& dst, ComputeContext& ctx) {
    detail::dispatchTernary<detail::ClampFunctor>(src0, src1, src2, dst, ctx.dtype);
}

// ============== Registration ==============

namespace {
    struct CpuKernelRegistrar {
        CpuKernelRegistrar() {
            Dispatcher& disp = Dispatcher::instance();
            const backend::DeviceType dev = backend::DeviceType::CPU;
            
            // Unary kernels
            for (int dtype = 1; dtype < static_cast<int>(DataType::Count); ++dtype) {
                DataType dt = static_cast<DataType>(dtype);
                disp.registerUnary(DispatchKey(Operator::Sqrt, dt, dev), sqrtKernel);
                disp.registerUnary(DispatchKey(Operator::Abs, dt, dev), absKernel);
                disp.registerUnary(DispatchKey(Operator::Sign, dt, dev), signKernel);
                disp.registerUnary(DispatchKey(Operator::Exp, dt, dev), expKernel);
                disp.registerUnary(DispatchKey(Operator::Log, dt, dev), logKernel);
                disp.registerUnary(DispatchKey(Operator::Sin, dt, dev), sinKernel);
                disp.registerUnary(DispatchKey(Operator::Cos, dt, dev), cosKernel);
                disp.registerUnary(DispatchKey(Operator::Ceil, dt, dev), ceilKernel);
                disp.registerUnary(DispatchKey(Operator::Floor, dt, dev), floorKernel);
            }
            
            // Binary kernels
            for (int dtype = 1; dtype < static_cast<int>(DataType::Count); ++dtype) {
                DataType dt = static_cast<DataType>(dtype);
                disp.registerBinary(DispatchKey(Operator::Add, dt, dev), addKernel);
                disp.registerBinary(DispatchKey(Operator::Sub, dt, dev), subKernel);
                disp.registerBinary(DispatchKey(Operator::Mul, dt, dev), mulKernel);
                disp.registerBinary(DispatchKey(Operator::Div, dt, dev), divKernel);
                disp.registerBinary(DispatchKey(Operator::Max, dt, dev), maxKernel);
                disp.registerBinary(DispatchKey(Operator::Min, dt, dev), minKernel);
                disp.registerBinary(DispatchKey(Operator::Mod, dt, dev), modKernel);
                disp.registerBinary(DispatchKey(Operator::Pow, dt, dev), powKernel);
            }
            
            // Ternary kernels
            for (int dtype = 1; dtype < static_cast<int>(DataType::Count); ++dtype) {
                DataType dt = static_cast<DataType>(dtype);
                disp.registerTernary(DispatchKey(Operator::Clamp, dt, dev), clampKernel);
            }
        }
    };
    
    // Static registration at startup
    static CpuKernelRegistrar registrar;
}

void registerCpuKernels() {
    // No-op: registration happens in static initializer above
}

} // namespace h3::core::compute::cpu
