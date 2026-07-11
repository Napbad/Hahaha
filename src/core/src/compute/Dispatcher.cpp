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

#include "compute/Dispatcher.h"

#include <format>
#include <string>

#include "Error.h"
#include "backend/Device.h"
#include "math/TensorInner.h"
#include "utils/OwnPointer.h"

namespace h3::core::compute {

std::expected<void, Error> Dispatcher::dispatch(
    Operator op,
    backend::DeviceType device,
    DataType dtype,
    std::vector<utils::OwnPointer<math::TensorInner>>& operands) const {
    
    if (operands.empty()) {
        return std::unexpected(Error(
            "dispatch failed: no operands provided",
            ErrorCode::BaseError));
    }

    const size_t arity = operands.size() - 1; // Last operand is output
    DispatchKey key(op, dtype, device);
    
    auto result = resolve(key, arity);
    if (!result) {
        return std::unexpected(Error(
            "no kernel registered for this configuration",
            ErrorCode::BaseError));
    }

    ComputeContext ctx(backend::Device(0, device), dtype);
    
    math::TensorInner& dst = *operands.back();
    
    switch (result.kind) {
    case DispatchResult::Kind::Unary: {
        const math::TensorInner& src = *operands[0];
        result.func.unary(src, dst, ctx);
        break;
    }
    case DispatchResult::Kind::Binary: {
        const math::TensorInner& src0 = *operands[0];
        const math::TensorInner& src1 = *operands[1];
        result.func.binary(src0, src1, dst, ctx);
        break;
    }
    case DispatchResult::Kind::Ternary: {
        const math::TensorInner& src0 = *operands[0];
        const math::TensorInner& src1 = *operands[1];
        const math::TensorInner& src2 = *operands[2];
        result.func.ternary(src0, src1, src2, dst, ctx);
        break;
    }
    case DispatchResult::Kind::None:
        return std::unexpected(Error(
            "invalid dispatch result",
            ErrorCode::BaseError));
    }

    return {};
}

std::string DispatchKey::toString() const {
    return std::format("DispatchKey(op={}, dtype={}, device={})",
                       static_cast<int>(op_),
                       static_cast<int>(dtype_),
                       static_cast<int>(device_));
}

} // namespace h3::core::compute
