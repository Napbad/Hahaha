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

#ifndef HAHAHA_COMPUTE_CONTEXT_V2_H
#define HAHAHA_COMPUTE_CONTEXT_V2_H

#include <cstddef>
#include <cstdint>

#include "backend/Device.h"
#include "defines.h"

namespace h3::core::compute {

/// ComputeContext carries runtime information needed by kernel execution.
/// This is the kernel-level context passed to all kernel functions.
struct ComputeContext {
    backend::Device device;
    DataType dtype;
    void* stream; // CUDA stream pointer, nullptr for CPU
    size_t threadCount; // Suggested thread/block count for GPU

    ComputeContext(backend::Device dev, DataType dt) noexcept
        : device(dev), dtype(dt), stream(nullptr), threadCount(0) {}

    ComputeContext(backend::Device dev, DataType dt, void* cudaStream) noexcept
        : device(dev), dtype(dt), stream(cudaStream), threadCount(0) {}

    ComputeContext(backend::DeviceType devType, DataType dt) noexcept
        : device(0, devType), dtype(dt), stream(nullptr), threadCount(0) {}
};

// Backward compatibility alias
using KernelContext = ComputeContext;

} // namespace h3::core::compute

#endif // HAHAHA_COMPUTE_CONTEXT_V2_H
