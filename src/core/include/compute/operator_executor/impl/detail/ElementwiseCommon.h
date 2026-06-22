//  Copyright (c) 2025-2026 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
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

#ifndef HAHAHA_ELEMENTWISECOMMON_H
#define HAHAHA_ELEMENTWISECOMMON_H

#include <array>
#include <cstddef>
#include <expected>
#include <span>
#include <string>
#include <vector>

#include "Error.h"
#include "compute/ComputeContext.h"
#include "defines.h"
#include "math/TensorInner.h"
#include "utils/OwnPointer.h"

namespace h3::core::compute::detail {

inline constexpr SizeT kMaxElementwiseRank = 8;

inline std::expected<void, Error>
checkOperandCount(const std::size_t expected,
                  const std::vector<utils::OwnPointer<math::TensorInner>>& operands) {
    if (operands.size() != expected) {
        return std::unexpected(Error(
            "expected " + std::to_string(expected) + " tensor operands, got "
                + std::to_string(operands.size()),
            ErrorCode::InvalidArgument));
    }
    return {};
}

inline std::expected<void, Error>
checkContextAndTensors(ComputeContext& context,
                       std::span<math::TensorInner* const> inputs,
                       const math::TensorInner& output) {
    const auto dtype = context.dataType();
    const auto device = context.device();

    for (const auto* tensor : inputs) {
        if (tensor->dataType() != dtype) {
            return std::unexpected(Error(
                "operand dtype mismatch for elementwise execution",
                ErrorCode::InvalidArgument));
        }
        if (tensor->device() != device) {
            return std::unexpected(Error(
                "operand device mismatch for elementwise execution",
                ErrorCode::InvalidArgument));
        }
        if (!tensor->shapeRef().canBroadcastWith(output.shapeRef())) {
            return std::unexpected(Error(
                "operand shape cannot broadcast to output shape",
                ErrorCode::InvalidArgument));
        }
    }

    if (output.dataType() != dtype || output.device() != device) {
        return std::unexpected(Error(
            "output tensor does not match compute context",
            ErrorCode::InvalidArgument));
    }

    return {};
}

inline SizeT elementOffset(const math::TensorInner& tensor,
                           const std::vector<SizeT>& coords) {
    const auto& shape = tensor.shapeRef();
    const auto& stride = tensor.strideRef();
    const SizeT outRank = static_cast<SizeT>(coords.size());
    const SizeT tRank = shape.rank();
    SizeT elemOff = 0;
    for (SizeT k = 0; k < outRank; ++k) {
        if (k >= tRank) {
            continue;
        }
        const SizeT tDim = tRank - 1 - k;
        const SizeT c = coords[outRank - 1 - k];
        const SizeT idx = shape[tDim] == 1 ? 0 : c;
        elemOff += idx * stride[tDim];
    }
    return elemOff;
}

inline void* elementPtr(math::TensorInner& tensor, const SizeT elemOff) {
    const SizeT byteOff = tensor.offset() + elemOff * sizeOf(tensor.dataType());
    return tensor.storageRef().data().as<char>() + byteOff;
}

inline const void* elementPtr(const math::TensorInner& tensor, const SizeT elemOff) {
    const SizeT byteOff = tensor.offset() + elemOff * sizeOf(tensor.dataType());
    return tensor.storageRef().data().as<const char>() + byteOff;
}

} // namespace h3::core::compute::detail

#endif // HAHAHA_ELEMENTWISECOMMON_H
