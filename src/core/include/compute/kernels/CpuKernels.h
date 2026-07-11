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

#ifndef HAHAHA_CPU_KERNELS_H
#define HAHAHA_CPU_KERNELS_H

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "compute/DispatchKey.h"
#include "compute/ComputeContext.h"
#include "defines.h"
#include "math/TensorInner.h"

namespace h3::core::compute {

// Forward declaration for Dispatcher
class Dispatcher;

namespace cpu {

namespace detail {

// ============== Functors ==============

template <typename T>
struct AddFunctor {
    template <typename U>
    HAHAHA_HD static T apply(U a, U b) {
        return static_cast<T>(static_cast<U>(a) + static_cast<U>(b));
    }
};

template <typename T>
struct SubFunctor {
    template <typename U>
    HAHAHA_HD static T apply(U a, U b) {
        return static_cast<T>(static_cast<U>(a) - static_cast<U>(b));
    }
};

template <typename T>
struct MulFunctor {
    template <typename U>
    HAHAHA_HD static T apply(U a, U b) {
        return static_cast<T>(static_cast<U>(a) * static_cast<U>(b));
    }
};

template <typename T>
struct DivFunctor {
    template <typename U>
    HAHAHA_HD static T apply(U a, U b) {
        return static_cast<T>(
            static_cast<double>(static_cast<U>(a)) / static_cast<double>(static_cast<U>(b)));
    }
};

template <typename T>
struct MaxFunctor {
    template <typename U>
    HAHAHA_HD static T apply(U a, U b) {
        return static_cast<T>(a > b ? a : b);
    }
};

template <typename T>
struct MinFunctor {
    template <typename U>
    HAHAHA_HD static T apply(U a, U b) {
        return static_cast<T>(a < b ? a : b);
    }
};

template <typename T>
struct ModFunctor {
    template <typename U>
    HAHAHA_HD static T apply(U a, U b) {
        return static_cast<T>(std::fmod(static_cast<double>(a), static_cast<double>(b)));
    }
};

template <typename T>
struct PowFunctor {
    template <typename U>
    HAHAHA_HD static T apply(U a, U b) {
        return static_cast<T>(std::pow(static_cast<double>(a), static_cast<double>(b)));
    }
};

template <typename T>
struct SqrtFunctor {
    HAHAHA_HD static T apply(T x) {
        return static_cast<T>(std::sqrt(static_cast<double>(x)));
    }
};

template <typename T>
struct AbsFunctor {
    HAHAHA_HD static T apply(T x) {
        if constexpr (std::is_unsigned_v<T>) {
            return x;
        }
        return static_cast<T>(x < T(0) ? -x : x);
    }
};

template <typename T>
struct SignFunctor {
    HAHAHA_HD static T apply(T x) {
        if (x > T(0)) return T(1);
        if (x < T(0)) return T(-1);
        return T(0);
    }
};

template <typename T>
struct ExpFunctor {
    HAHAHA_HD static T apply(T x) {
        return static_cast<T>(std::exp(static_cast<double>(x)));
    }
};

template <typename T>
struct LogFunctor {
    HAHAHA_HD static T apply(T x) {
        return static_cast<T>(std::log(static_cast<double>(x)));
    }
};

template <typename T>
struct SinFunctor {
    HAHAHA_HD static T apply(T x) {
        return static_cast<T>(std::sin(static_cast<double>(x)));
    }
};

template <typename T>
struct CosFunctor {
    HAHAHA_HD static T apply(T x) {
        return static_cast<T>(std::cos(static_cast<double>(x)));
    }
};

template <typename T>
struct CeilFunctor {
    HAHAHA_HD static T apply(T x) {
        return static_cast<T>(std::ceil(static_cast<double>(x)));
    }
};

template <typename T>
struct FloorFunctor {
    HAHAHA_HD static T apply(T x) {
        return static_cast<T>(std::floor(static_cast<double>(x)));
    }
};

template <typename T>
struct ClampFunctor {
    HAHAHA_HD static T apply(T x, T lo, T hi) {
        const T v = x < lo ? lo : x;
        return v > hi ? hi : v;
    }
};

// ============== Helper functions ==============

inline SizeT elementOffset(const math::TensorInner& tensor,
                           const std::vector<SizeT>& coords) {
    const auto& shape = tensor.shapeRef();
    const auto& stride = tensor.strideRef();
    const SizeT outRank = static_cast<SizeT>(coords.size());
    const SizeT tRank = shape.rank();
    SizeT elemOff = 0;
    for (SizeT k = 0; k < outRank; ++k) {
        if (k >= tRank) continue;
        const SizeT tDim = tRank - 1 - k;
        const SizeT c = coords[outRank - 1 - k];
        const SizeT idx = shape[tDim] == 1 ? 0 : c;
        elemOff += idx * stride[tDim];
    }
    return elemOff;
}

inline void* elementPtr(math::TensorInner& tensor, SizeT elemOff) {
    const SizeT byteOff = tensor.offset() + elemOff * sizeOf(tensor.dataType());
    return tensor.storageRef().data().as<char>() + byteOff;
}

inline const void* elementPtr(const math::TensorInner& tensor, SizeT elemOff) {
    const SizeT byteOff = tensor.offset() + elemOff * sizeOf(tensor.dataType());
    return tensor.storageRef().data().as<const char>() + byteOff;
}

// ============== Type-dispatched elementwise runners ==============

template <template <typename> typename Functor, typename T>
void runUnary(const math::TensorInner& src, math::TensorInner& dst) {
    const auto& outShape = dst.shapeRef();
    const SizeT rank = outShape.rank();
    const SizeT total = outShape.getTotalSize();
    std::vector<SizeT> coords(static_cast<std::size_t>(rank));

    for (SizeT linear = 0; linear < total; ++linear) {
        SizeT rem = linear;
        for (SizeT d = rank; d > 0; --d) {
            const SizeT dim = outShape[d - 1];
            coords[static_cast<std::size_t>(d - 1)] = rem % dim;
            rem /= dim;
        }

        const SizeT outIdx = elementOffset(dst, coords);
        T* out = static_cast<T*>(elementPtr(dst, outIdx));
        const T* in = static_cast<const T*>(elementPtr(src, elementOffset(src, coords)));
        *out = Functor<T>::apply(*in);
    }
}

template <template <typename> typename Functor, typename T>
void runBinary(const math::TensorInner& src0, const math::TensorInner& src1, math::TensorInner& dst) {
    const auto& outShape = dst.shapeRef();
    const SizeT rank = outShape.rank();
    const SizeT total = outShape.getTotalSize();
    std::vector<SizeT> coords(static_cast<std::size_t>(rank));

    for (SizeT linear = 0; linear < total; ++linear) {
        SizeT rem = linear;
        for (SizeT d = rank; d > 0; --d) {
            const SizeT dim = outShape[d - 1];
            coords[static_cast<std::size_t>(d - 1)] = rem % dim;
            rem /= dim;
        }

        const SizeT outIdx = elementOffset(dst, coords);
        T* out = static_cast<T*>(elementPtr(dst, outIdx));
        const T* a = static_cast<const T*>(elementPtr(src0, elementOffset(src0, coords)));
        const T* b = static_cast<const T*>(elementPtr(src1, elementOffset(src1, coords)));
        *out = Functor<T>::apply(*a, *b);
    }
}

template <template <typename> typename Functor, typename T>
void runTernary(const math::TensorInner& src0, 
                const math::TensorInner& src1,
                const math::TensorInner& src2, 
                math::TensorInner& dst) {
    const auto& outShape = dst.shapeRef();
    const SizeT rank = outShape.rank();
    const SizeT total = outShape.getTotalSize();
    std::vector<SizeT> coords(static_cast<std::size_t>(rank));

    for (SizeT linear = 0; linear < total; ++linear) {
        SizeT rem = linear;
        for (SizeT d = rank; d > 0; --d) {
            const SizeT dim = outShape[d - 1];
            coords[static_cast<std::size_t>(d - 1)] = rem % dim;
            rem /= dim;
        }

        const SizeT outIdx = elementOffset(dst, coords);
        T* out = static_cast<T*>(elementPtr(dst, outIdx));
        const T* a = static_cast<const T*>(elementPtr(src0, elementOffset(src0, coords)));
        const T* b = static_cast<const T*>(elementPtr(src1, elementOffset(src1, coords)));
        const T* c = static_cast<const T*>(elementPtr(src2, elementOffset(src2, coords)));
        *out = Functor<T>::apply(*a, *b, *c);
    }
}

// Type dispatcher for binary kernels
template <template <typename> typename Functor>
void dispatchBinary(const math::TensorInner& src0,
                    const math::TensorInner& src1,
                    math::TensorInner& dst,
                    DataType dtype) {
    switch (dtype) {
    case DataType::Int8: runBinary<Functor, Int8>(src0, src1, dst); break;
    case DataType::UInt8: runBinary<Functor, UInt8>(src0, src1, dst); break;
    case DataType::Int16: runBinary<Functor, Int16>(src0, src1, dst); break;
    case DataType::UInt16: runBinary<Functor, UInt16>(src0, src1, dst); break;
    case DataType::Int32: runBinary<Functor, Int32>(src0, src1, dst); break;
    case DataType::UInt32: runBinary<Functor, UInt32>(src0, src1, dst); break;
    case DataType::Float32: runBinary<Functor, Float32>(src0, src1, dst); break;
    case DataType::Int64: runBinary<Functor, Int64>(src0, src1, dst); break;
    case DataType::UInt64: runBinary<Functor, UInt64>(src0, src1, dst); break;
    case DataType::Float64: runBinary<Functor, Float64>(src0, src1, dst); break;
    default: break;
    }
}

// Type dispatcher for unary kernels
template <template <typename> typename Functor>
void dispatchUnary(const math::TensorInner& src,
                    math::TensorInner& dst,
                    DataType dtype) {
    switch (dtype) {
    case DataType::Int8: runUnary<Functor, Int8>(src, dst); break;
    case DataType::UInt8: runUnary<Functor, UInt8>(src, dst); break;
    case DataType::Int16: runUnary<Functor, Int16>(src, dst); break;
    case DataType::UInt16: runUnary<Functor, UInt16>(src, dst); break;
    case DataType::Int32: runUnary<Functor, Int32>(src, dst); break;
    case DataType::UInt32: runUnary<Functor, UInt32>(src, dst); break;
    case DataType::Float32: runUnary<Functor, Float32>(src, dst); break;
    case DataType::Int64: runUnary<Functor, Int64>(src, dst); break;
    case DataType::UInt64: runUnary<Functor, UInt64>(src, dst); break;
    case DataType::Float64: runUnary<Functor, Float64>(src, dst); break;
    default: break;
    }
}

// Type dispatcher for ternary kernels
template <template <typename> typename Functor>
void dispatchTernary(const math::TensorInner& src0,
                      const math::TensorInner& src1,
                      const math::TensorInner& src2,
                      math::TensorInner& dst,
                      DataType dtype) {
    switch (dtype) {
    case DataType::Int8: runTernary<Functor, Int8>(src0, src1, src2, dst); break;
    case DataType::UInt8: runTernary<Functor, UInt8>(src0, src1, src2, dst); break;
    case DataType::Int16: runTernary<Functor, Int16>(src0, src1, src2, dst); break;
    case DataType::UInt16: runTernary<Functor, UInt16>(src0, src1, src2, dst); break;
    case DataType::Int32: runTernary<Functor, Int32>(src0, src1, src2, dst); break;
    case DataType::UInt32: runTernary<Functor, UInt32>(src0, src1, src2, dst); break;
    case DataType::Float32: runTernary<Functor, Float32>(src0, src1, src2, dst); break;
    case DataType::Int64: runTernary<Functor, Int64>(src0, src1, src2, dst); break;
    case DataType::UInt64: runTernary<Functor, UInt64>(src0, src1, src2, dst); break;
    case DataType::Float64: runTernary<Functor, Float64>(src0, src1, src2, dst); break;
    default: break;
    }
}

} // namespace detail

// ============== Concrete Kernel Functions (API) ==============

// Binary kernels
void addKernel(const math::TensorInner& src0, const math::TensorInner& src1,
               math::TensorInner& dst, ComputeContext& ctx);

void subKernel(const math::TensorInner& src0, const math::TensorInner& src1,
               math::TensorInner& dst, ComputeContext& ctx);

void mulKernel(const math::TensorInner& src0, const math::TensorInner& src1,
               math::TensorInner& dst, ComputeContext& ctx);

void divKernel(const math::TensorInner& src0, const math::TensorInner& src1,
               math::TensorInner& dst, ComputeContext& ctx);

void maxKernel(const math::TensorInner& src0, const math::TensorInner& src1,
               math::TensorInner& dst, ComputeContext& ctx);

void minKernel(const math::TensorInner& src0, const math::TensorInner& src1,
               math::TensorInner& dst, ComputeContext& ctx);

void modKernel(const math::TensorInner& src0, const math::TensorInner& src1,
               math::TensorInner& dst, ComputeContext& ctx);

void powKernel(const math::TensorInner& src0, const math::TensorInner& src1,
               math::TensorInner& dst, ComputeContext& ctx);

// Unary kernels
void sqrtKernel(const math::TensorInner& src, math::TensorInner& dst, ComputeContext& ctx);
void absKernel(const math::TensorInner& src, math::TensorInner& dst, ComputeContext& ctx);
void signKernel(const math::TensorInner& src, math::TensorInner& dst, ComputeContext& ctx);
void expKernel(const math::TensorInner& src, math::TensorInner& dst, ComputeContext& ctx);
void logKernel(const math::TensorInner& src, math::TensorInner& dst, ComputeContext& ctx);
void sinKernel(const math::TensorInner& src, math::TensorInner& dst, ComputeContext& ctx);
void cosKernel(const math::TensorInner& src, math::TensorInner& dst, ComputeContext& ctx);
void ceilKernel(const math::TensorInner& src, math::TensorInner& dst, ComputeContext& ctx);
void floorKernel(const math::TensorInner& src, math::TensorInner& dst, ComputeContext& ctx);

// Ternary kernels
void clampKernel(const math::TensorInner& src0, const math::TensorInner& src1,
                 const math::TensorInner& src2, math::TensorInner& dst, ComputeContext& ctx);

} // namespace cpu

// ============== Kernel Registration (must be implemented in .cpp) ==============

/// Initialize CPU kernel registrations. Called at program startup.
void registerCpuKernels();

} // namespace h3::core::compute

#endif // HAHAHA_CPU_KERNELS_H
