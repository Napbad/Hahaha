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

#ifndef HAHAHA_ELEMENTWISEKERNEL_H
#define HAHAHA_ELEMENTWISEKERNEL_H

#include <expected>
#include <vector>

#include "compute/operator_executor/impl/detail/ElementwiseCommon.h"
#include "defines.h"

namespace h3::core::compute::detail {

template <typename T, typename Functor, std::size_t Arity>
std::expected<void, Error>
runElementwise(ComputeContext& context,
               std::vector<utils::OwnPointer<math::TensorInner>>& operands) {
    if (const auto countOk = checkOperandCount(Arity + 1, operands); !countOk) {
        return countOk;
    }

    const utils::OwnPointer<math::TensorInner> output = operands.back();
    std::array<math::TensorInner*, Arity> inputs{};
    for (std::size_t i = 0; i < Arity; ++i) {
        inputs[i] = operands[i];
    }

    if (const auto ctxOk = checkContextAndTensors(context, inputs, *output); !ctxOk) {
        return ctxOk;
    }

    const auto& outShape = output->shapeRef();
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

        const SizeT outIdx = elementOffset(*output, coords);
        T* out = static_cast<T*>(elementPtr(*output, outIdx));

        if constexpr (Arity == 1) {
            const T* in = static_cast<const T*>(elementPtr(*inputs[0], elementOffset(*inputs[0], coords)));
            *out = Functor::template apply<T>(*in);
        } else if constexpr (Arity == 2) {
            const T* lhs = static_cast<const T*>(elementPtr(*inputs[0], elementOffset(*inputs[0], coords)));
            const T* rhs = static_cast<const T*>(elementPtr(*inputs[1], elementOffset(*inputs[1], coords)));
            *out = Functor::template apply<T>(*lhs, *rhs);
        } else {
            const T* a = static_cast<const T*>(elementPtr(*inputs[0], elementOffset(*inputs[0], coords)));
            const T* b = static_cast<const T*>(elementPtr(*inputs[1], elementOffset(*inputs[1], coords)));
            const T* c = static_cast<const T*>(elementPtr(*inputs[2], elementOffset(*inputs[2], coords)));
            *out = Functor::template apply<T>(*a, *b, *c);
        }
    }

    return {};
}

template <typename Functor>
std::expected<void, Error>
dispatchByDtype(ComputeContext& context,
                std::vector<utils::OwnPointer<math::TensorInner>>& operands,
                const std::size_t arity) {
    switch (context.dataType()) {
    case DataType::Int8:
        if (arity == 1) {
            return runElementwise<Int8, Functor, 1>(context, operands);
        }
        if (arity == 2) {
            return runElementwise<Int8, Functor, 2>(context, operands);
        }
        return runElementwise<Int8, Functor, 3>(context, operands);
    case DataType::UInt8:
        if (arity == 1) {
            return runElementwise<UInt8, Functor, 1>(context, operands);
        }
        if (arity == 2) {
            return runElementwise<UInt8, Functor, 2>(context, operands);
        }
        return runElementwise<UInt8, Functor, 3>(context, operands);
    case DataType::Int16:
        if (arity == 1) {
            return runElementwise<Int16, Functor, 1>(context, operands);
        }
        if (arity == 2) {
            return runElementwise<Int16, Functor, 2>(context, operands);
        }
        return runElementwise<Int16, Functor, 3>(context, operands);
    case DataType::UInt16:
        if (arity == 1) {
            return runElementwise<UInt16, Functor, 1>(context, operands);
        }
        if (arity == 2) {
            return runElementwise<UInt16, Functor, 2>(context, operands);
        }
        return runElementwise<UInt16, Functor, 3>(context, operands);
    case DataType::Int32:
        if (arity == 1) {
            return runElementwise<Int32, Functor, 1>(context, operands);
        }
        if (arity == 2) {
            return runElementwise<Int32, Functor, 2>(context, operands);
        }
        return runElementwise<Int32, Functor, 3>(context, operands);
    case DataType::UInt32:
        if (arity == 1) {
            return runElementwise<UInt32, Functor, 1>(context, operands);
        }
        if (arity == 2) {
            return runElementwise<UInt32, Functor, 2>(context, operands);
        }
        return runElementwise<UInt32, Functor, 3>(context, operands);
    case DataType::Float32:
        if (arity == 1) {
            return runElementwise<Float32, Functor, 1>(context, operands);
        }
        if (arity == 2) {
            return runElementwise<Float32, Functor, 2>(context, operands);
        }
        return runElementwise<Float32, Functor, 3>(context, operands);
    case DataType::Int64:
        if (arity == 1) {
            return runElementwise<Int64, Functor, 1>(context, operands);
        }
        if (arity == 2) {
            return runElementwise<Int64, Functor, 2>(context, operands);
        }
        return runElementwise<Int64, Functor, 3>(context, operands);
    case DataType::UInt64:
        if (arity == 1) {
            return runElementwise<UInt64, Functor, 1>(context, operands);
        }
        if (arity == 2) {
            return runElementwise<UInt64, Functor, 2>(context, operands);
        }
        return runElementwise<UInt64, Functor, 3>(context, operands);
    case DataType::Float64:
        if (arity == 1) {
            return runElementwise<Float64, Functor, 1>(context, operands);
        }
        if (arity == 2) {
            return runElementwise<Float64, Functor, 2>(context, operands);
        }
        return runElementwise<Float64, Functor, 3>(context, operands);
    default:
        return std::unexpected(Error(
            "unsupported dtype for elementwise execution",
            ErrorCode::NotImplemented));
    }
}

template <typename Functor>
std::expected<void, Error>
runUnary(ComputeContext& context,
         std::vector<utils::OwnPointer<math::TensorInner>>& operands) {
    return dispatchByDtype<Functor>(context, operands, 1);
}

template <typename Functor>
std::expected<void, Error>
runBinary(ComputeContext& context,
          std::vector<utils::OwnPointer<math::TensorInner>>& operands) {
    return dispatchByDtype<Functor>(context, operands, 2);
}

template <typename Functor>
std::expected<void, Error>
runTernary(ComputeContext& context,
           std::vector<utils::OwnPointer<math::TensorInner>>& operands) {
    return dispatchByDtype<Functor>(context, operands, 3);
}

} // namespace h3::core::compute::detail

#endif // HAHAHA_ELEMENTWISEKERNEL_H
