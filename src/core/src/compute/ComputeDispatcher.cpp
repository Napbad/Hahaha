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

//
// Created by napbad on 3/26/26.
//

#include "compute/ComputeDispatcher.h"

#include <expected>
#include <format>
#include <stdexcept>

#include "Error.h"
#include "compute/ComputeContext.h"
#include "compute/operator_executor/OperatorExecutorFactory.h"

namespace h3::core::compute {

ComputeDispatcher::~ComputeDispatcher() = default;

std::expected<void, Error> ComputeDispatcher::dispatch(const Operator op,
                                                       std::vector<ComputeNode>& nodes) {

    if (nodes.empty()) {
        throw std::invalid_argument(
            std::format(
                "invalid input while dispatching the operator {}, no tensor is given",
                toString(op))
            );
    }

    const auto type = nodes.front().tensorInner()->dataType();
    const auto device = nodes.front().tensorInner()->device();
    ComputeContext context(device, type);

    static OperatorExecutorFactory factory;
    const auto executor = factory.get(op, context.deviceType());
    if (!executor) {
        return std::unexpected(executor.error());
    }

    return executor.value()->execute(context, nodes);
}

}