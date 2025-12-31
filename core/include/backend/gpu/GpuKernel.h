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

#ifndef HAHAHA_BACKEND_GPU_GPU_KERNEL_H
#define HAHAHA_BACKEND_GPU_GPU_KERNEL_H

#include <memory>
#include <string>
#include <vector>

namespace hahaha::backend::gpu {

/**
 * @brief Represents a GPU kernel that can be executed on a device.
 *
 * This class provides an interface for loading and launching kernels.
 */
class GpuKernel {
  public:
    GpuKernel() = default;
    virtual ~GpuKernel() = default;

    /**
     * @brief Load a kernel from source or binary.
     * @param source The kernel source code or path to binary.
     * @param kernel_name The name of the kernel function.
     * @return bool True if loading was successful.
     */
    virtual bool load(const std::string& source,
                      const std::string& kernel_name) = 0;

    /**
     * @brief Launch the kernel with specified grid and block dimensions.
     * @param grid_dims Dimensions of the grid (number of blocks).
     * @param block_dims Dimensions of the block (number of threads per block).
     * @param args Arguments to pass to the kernel.
     */
    virtual void launch(const std::vector<size_t>& grid_dims,
                        const std::vector<size_t>& block_dims,
                        void** args) = 0;

    /**
     * @brief Get the kernel name.
     * @return std::string The kernel name.
     */
    [[nodiscard]] virtual std::string getName() const = 0;
};

} // namespace hahaha::backend::gpu

#endif // HAHAHA_BACKEND_GPU_GPU_KERNEL_H
