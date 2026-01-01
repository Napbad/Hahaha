// Copyright (c) 2025 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
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

#ifndef HAHAHA_BACKEND_GPU_GPU_MEMORY_H
#define HAHAHA_BACKEND_GPU_GPU_MEMORY_H

#include <cstddef>

namespace hahaha::backend::gpu {

/**
 * @brief Utilities for GPU memory allocation and data transfer.
 */
class GpuMemory {
  public:
    /**
     * @brief Allocate memory on the GPU.
     * @param size Size of memory to allocate in bytes.
     * @return void* Pointer to the allocated memory.
     */
    static void* allocate(size_t size);

    /**
     * @brief Deallocate memory on the GPU.
     * @param ptr Pointer to the memory to deallocate.
     */
    static void deallocate(void* ptr);

    /**
     * @brief Copy data from host to device.
     * @param device_ptr Destination pointer on the device.
     * @param host_ptr Source pointer on the host.
     * @param size Size of data to copy in bytes.
     */
    static void
    copyToDevice(void* device_ptr, const void* host_ptr, size_t size);

    /**
     * @brief Copy data from device to host.
     * @param host_ptr Destination pointer on the host.
     * @param device_ptr Source pointer on the device.
     * @param size Size of data to copy in bytes.
     */
    static void copyToHost(void* host_ptr, const void* device_ptr, size_t size);

    /**
     * @brief Copy data from device to device.
     * @param dest_ptr Destination pointer on the device.
     * @param src_ptr Source pointer on the device.
     * @param size Size of data to copy in bytes.
     */
    static void
    copyDeviceToDevice(void* dest_ptr, const void* src_ptr, size_t size);
};

} // namespace hahaha::backend::gpu

#endif // HAHAHA_BACKEND_GPU_GPU_MEMORY_H
