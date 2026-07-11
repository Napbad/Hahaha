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

#ifndef HAHAHA_CUDAELEMENTWISELAUNCH_CUH
#define HAHAHA_CUDAELEMENTWISELAUNCH_CUH

#include <vector>

#include "compute/operator_executor/impl/detail/CudaElementwise.h"
#include "compute/operator_executor/impl/detail/CudaElementwiseKernels.cuh"
#include "compute/operator_executor/impl/detail/ElementwiseCommon.h"
#include "defines.h"

namespace h3::core::compute::detail {

inline CudaTensorView makeCudaView(const math::TensorInner& tensor) {
    CudaTensorView view{};
    view.base = tensor.storageRef().data().get();
    view.byteOffset = tensor.offset();
    view.rank = tensor.shapeRef().rank();
    for (SizeT i = 0; i < view.rank && i < kMaxElementwiseRank; ++i) {
        view.shape[i] = tensor.shapeRef()[i];
        view.stride[i] = tensor.strideRef()[i];
    }
    return view;
}

inline std::expected<void, Error>
fillOutShape(const math::TensorInner& output, SizeT* deviceShape, SizeT& rank) {
    rank = output.shapeRef().rank();
    if (rank > kMaxElementwiseRank) {
        return std::unexpected(Error(
            "tensor rank exceeds elementwise kernel limit",
            ErrorCode::InvalidArgument));
    }
    std::array<SizeT, kMaxElementwiseRank> hostShape{};
    for (SizeT i = 0; i < rank; ++i) {
        hostShape[static_cast<std::size_t>(i)] = output.shapeRef()[i];
    }
    if (const auto err = cudaMemcpy(deviceShape,
                                    hostShape.data(),
                                    sizeof(SizeT) * static_cast<std::size_t>(rank),
                                    cudaMemcpyHostToDevice);
        err != cudaSuccess) {
        return cudaCheck(err, "cudaMemcpy outShape");
    }
    return {};
}

template <typename T, typename Functor, std::size_t Arity>
std::expected<void, Error>
launchElementwise(ComputeContext& context,
                  std::vector<utils::OwnPointer<math::TensorInner>>& operands) {
    if (const auto countOk = checkOperandCount(Arity + 1, operands); !countOk) {
        return countOk;
    }

    math::TensorInner* output = operands.back().get();
    std::array<math::TensorInner*, Arity> inputs{};
    for (std::size_t i = 0; i < Arity; ++i) {
        inputs[i] = operands[i].get();
    }

    if (const auto ctxOk = checkContextAndTensors(context, inputs, *output); !ctxOk) {
        return ctxOk;
    }

    SizeT* deviceShape = nullptr;
    if (const auto allocErr = cudaMalloc(&deviceShape, sizeof(SizeT) * kMaxElementwiseRank);
        allocErr != cudaSuccess) {
        return cudaCheck(allocErr, "cudaMalloc outShape");
    }

    SizeT rank = 0;
    if (const auto shapeOk = fillOutShape(*output, deviceShape, rank); !shapeOk) {
        cudaFree(deviceShape);
        return shapeOk;
    }

    const SizeT total = output->shapeRef().getTotalSize();
    const CudaTensorView result = makeCudaView(*output);
    auto* outBase = output->storageRef().data().as<char>();

    constexpr int kBlock = 256;
    const int blocks = static_cast<int>((total + kBlock - 1) / kBlock);

    if constexpr (Arity == 1) {
        const CudaTensorView in = makeCudaView(*inputs[0]);
        unaryElementwiseKernel<T, Functor><<<blocks, kBlock>>>(
            reinterpret_cast<T*>(outBase), in, result, deviceShape, total);
    } else if constexpr (Arity == 2) {
        const CudaTensorView lhs = makeCudaView(*inputs[0]);
        const CudaTensorView rhs = makeCudaView(*inputs[1]);
        binaryElementwiseKernel<T, Functor><<<blocks, kBlock>>>(
            reinterpret_cast<T*>(outBase), lhs, rhs, result, deviceShape, total);
    } else {
        const CudaTensorView a = makeCudaView(*inputs[0]);
        const CudaTensorView b = makeCudaView(*inputs[1]);
        const CudaTensorView c = makeCudaView(*inputs[2]);
        ternaryElementwiseKernel<T, Functor><<<blocks, kBlock>>>(
            reinterpret_cast<T*>(outBase), a, b, c, result, deviceShape, total);
    }

    cudaFree(deviceShape);
    return cudaCheck(cudaGetLastError(), "elementwise kernel launch");
}

template <typename Functor>
std::expected<void, Error>
dispatchCudaByDtype(ComputeContext& context,
                    std::vector<utils::OwnPointer<math::TensorInner>>& operands,
                    const std::size_t arity) {
    switch (context.dataType()) {
    case DataType::Int8:
        if (arity == 1) {
            return launchElementwise<Int8, Functor, 1>(context, operands);
        }
        if (arity == 2) {
            return launchElementwise<Int8, Functor, 2>(context, operands);
        }
        return launchElementwise<Int8, Functor, 3>(context, operands);
    case DataType::UInt8:
        if (arity == 1) {
            return launchElementwise<UInt8, Functor, 1>(context, operands);
        }
        if (arity == 2) {
            return launchElementwise<UInt8, Functor, 2>(context, operands);
        }
        return launchElementwise<UInt8, Functor, 3>(context, operands);
    case DataType::Int16:
        if (arity == 1) {
            return launchElementwise<Int16, Functor, 1>(context, operands);
        }
        if (arity == 2) {
            return launchElementwise<Int16, Functor, 2>(context, operands);
        }
        return launchElementwise<Int16, Functor, 3>(context, operands);
    case DataType::UInt16:
        if (arity == 1) {
            return launchElementwise<UInt16, Functor, 1>(context, operands);
        }
        if (arity == 2) {
            return launchElementwise<UInt16, Functor, 2>(context, operands);
        }
        return launchElementwise<UInt16, Functor, 3>(context, operands);
    case DataType::Int32:
        if (arity == 1) {
            return launchElementwise<Int32, Functor, 1>(context, operands);
        }
        if (arity == 2) {
            return launchElementwise<Int32, Functor, 2>(context, operands);
        }
        return launchElementwise<Int32, Functor, 3>(context, operands);
    case DataType::UInt32:
        if (arity == 1) {
            return launchElementwise<UInt32, Functor, 1>(context, operands);
        }
        if (arity == 2) {
            return launchElementwise<UInt32, Functor, 2>(context, operands);
        }
        return launchElementwise<UInt32, Functor, 3>(context, operands);
    case DataType::Float32:
        if (arity == 1) {
            return launchElementwise<Float32, Functor, 1>(context, operands);
        }
        if (arity == 2) {
            return launchElementwise<Float32, Functor, 2>(context, operands);
        }
        return launchElementwise<Float32, Functor, 3>(context, operands);
    case DataType::Int64:
        if (arity == 1) {
            return launchElementwise<Int64, Functor, 1>(context, operands);
        }
        if (arity == 2) {
            return launchElementwise<Int64, Functor, 2>(context, operands);
        }
        return launchElementwise<Int64, Functor, 3>(context, operands);
    case DataType::UInt64:
        if (arity == 1) {
            return launchElementwise<UInt64, Functor, 1>(context, operands);
        }
        if (arity == 2) {
            return launchElementwise<UInt64, Functor, 2>(context, operands);
        }
        return launchElementwise<UInt64, Functor, 3>(context, operands);
    case DataType::Float64:
        if (arity == 1) {
            return launchElementwise<Float64, Functor, 1>(context, operands);
        }
        if (arity == 2) {
            return launchElementwise<Float64, Functor, 2>(context, operands);
        }
        return launchElementwise<Float64, Functor, 3>(context, operands);
    default:
        return std::unexpected(Error(
            "unsupported dtype for cuda elementwise execution",
            ErrorCode::NotImplemented));
    }
}

template <typename Functor>
std::expected<void, Error>
runUnaryCuda(ComputeContext& context,
             std::vector<utils::OwnPointer<math::TensorInner>>& operands) {
    return dispatchCudaByDtype<Functor>(context, operands, 1);
}

template <typename Functor>
std::expected<void, Error>
runBinaryCuda(ComputeContext& context,
              std::vector<utils::OwnPointer<math::TensorInner>>& operands) {
    return dispatchCudaByDtype<Functor>(context, operands, 2);
}

template <typename Functor>
std::expected<void, Error>
runTernaryCuda(ComputeContext& context,
               std::vector<utils::OwnPointer<math::TensorInner>>& operands) {
    return dispatchCudaByDtype<Functor>(context, operands, 3);
}

} // namespace h3::core::compute::detail

#endif // HAHAHA_CUDAELEMENTWISELAUNCH_CUH
