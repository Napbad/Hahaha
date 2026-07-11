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

#ifndef HAHAHA_COMPUTE_DISPATCHER_V2_H_290902236E544C94A3504172247EF7D3
#define HAHAHA_COMPUTE_DISPATCHER_V2_H_290902236E544C94A3504172247EF7D3

#include <expected>
#include <vector>

#include "backend/Device.h"
#include "defines.h"
#include "Error.h"
#include "utils/OwnPointer.h"

namespace h3::core::math {
class TensorInner;
}

namespace h3::core::compute {

class Dispatcher;

/// ComputeDispatcher is the high-level dispatch API that replaces the old
/// operator executor factory pattern. It uses the flat Dispatcher registry
/// for O(1) kernel lookup.
class ComputeDispatcher {
public:
    ~ComputeDispatcher();

    /// Dispatch an operation to the appropriate kernel based on operand types.
    /// This is the main entry point for all compute operations.
    static std::expected<void, Error> dispatch(
        Operator op,
        std::vector<utils::OwnPointer<math::TensorInner>>& tensors);

    /// Dispatch with explicit device and dtype override
    static std::expected<void, Error> dispatch(
        Operator op,
        backend::DeviceType device,
        DataType dtype,
        std::vector<utils::OwnPointer<math::TensorInner>>& tensors);
};

} // namespace h3::core::compute

#endif // HAHAHA_COMPUTE_DISPATCHER_V2_H_290902236E544C94A3504172247EF7D3
