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

#include "Error.h"

namespace h3::core::compute {

ComputeDispatcher::~ComputeDispatcher() = default;

std::expected<void, Error> ComputeDispatcher::dispatch(const Operator op,
                                                       const std::vector<ComputeNode>
                                                       & nodes,
                                                       const DataType type,
                                                       const backend::Device
                                                       device) {

    switch (device.type()) {
    case backend::DeviceType::CPU:
        return dispatchOnCPU(op, nodes, type, device);
    case backend::DeviceType::CUDA:
        return dispatchOnCUDA(op, nodes, type, device);
    default: ;
        return std::unexpected(Error(
            "unknown device type",
            ErrorCode::DeviceNotSupportedError));
    }
}

}