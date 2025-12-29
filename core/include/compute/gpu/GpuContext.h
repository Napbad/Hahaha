// Copyright (c) 2025 Contributors of hahaha(https://github.com/Napbad/Hahaha)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0 
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Contributors:
// Napbad (napbad.sen@gmail.com ) (https://github.com/Napbad )
//

#ifndef HAHAHA_COMPUTE_GPU_GPU_CONTEXT_H
#define HAHAHA_COMPUTE_GPU_GPU_CONTEXT_H

#include <memory>
#include <string>
#include <vector>

namespace hahaha::compute::gpu
{

/**
 * @brief Manages the GPU device context and resources.
 *
 * This class handles initialization, device selection, and management of
 * global resources required for GPU computation.
 */
class GpuContext
{
  public:
    GpuContext() = default;
    virtual ~GpuContext() = default;

    GpuContext(const GpuContext&) = delete;
    GpuContext& operator=(const GpuContext&) = delete;

    /**
     * @brief Initialize the GPU context.
     * @return bool True if initialization was successful.
     */
    virtual bool initialize() = 0;

    /**
     * @brief Select a specific GPU device.
     * @param device_id ID of the device to select.
     */
    virtual void setDevice(int device_id) = 0;

    /**
     * @brief Synchronize the current device.
     */
    virtual void synchronize() = 0;

    /**
     * @brief Get the name of the current device.
     * @return std::string The device name.
     */
    [[nodiscard]] virtual std::string getDeviceName() const = 0;

    /**
     * @brief Get the number of available GPU devices.
     * @return int Number of devices.
     */
    [[nodiscard]] virtual int getDeviceCount() const = 0;
};

} // namespace hahaha::compute::gpu

#endif // HAHAHA_COMPUTE_GPU_GPU_CONTEXT_H

