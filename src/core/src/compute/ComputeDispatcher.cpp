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

#include "compute/ComputeDispatcher.h"

#include <format>

#include "Error.h"
#include "compute/Dispatcher.h"

// Include kernel implementations to trigger static registration
#include "compute/kernels/CpuKernels.h"
#ifdef HAHAHA_USE_CUDA
#include "compute/kernels/CudaKernels.h"
#endif

namespace h3::core::compute {

ComputeDispatcher::~ComputeDispatcher() = default;

std::expected<void, Error> ComputeDispatcher::dispatch(
    Operator op,
    std::vector<utils::OwnPointer<math::TensorInner>>& tensors) {
    
    if (tensors.empty()) {
        return std::unexpected(Error(
            std::format("dispatch failed: no tensors provided for operator {}",
                       toString(op)),
            ErrorCode::InvalidArgument));
    }

    // Infer device and dtype from input tensors
    const auto& firstTensor = *tensors.front();
    const backend::DeviceType device = firstTensor.device().type();
    const DataType dtype = firstTensor.dataType();

    return dispatch(op, device, dtype, tensors);
}

std::expected<void, Error> ComputeDispatcher::dispatch(
    Operator op,
    backend::DeviceType device,
    DataType dtype,
    std::vector<utils::OwnPointer<math::TensorInner>>& tensors) {
    
    if (tensors.empty()) {
        return std::unexpected(Error(
            std::format("dispatch failed: no tensors provided for operator {}",
                       toString(op)),
            ErrorCode::InvalidArgument));
    }

    // Last tensor is output, rest are inputs
    const size_t arity = tensors.size() - 1;
    if (arity == 0) {
        return std::unexpected(Error(
            "dispatch failed: no input tensors provided",
            ErrorCode::InvalidArgument));
    }

    // Delegate to the flat Dispatcher
    return Dispatcher::instance().dispatch(op, device, dtype, tensors);
}

} // namespace h3::core::compute
