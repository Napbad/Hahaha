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

#ifndef HAHAHA_COMPUTE_DEVICE_COMPUTE_DISPATCHER_H
#define HAHAHA_COMPUTE_DEVICE_COMPUTE_DISPATCHER_H

#include <stdexcept>
#include "compute/Device.h"
#include "compute/compute_graph/Operator.h"

namespace hahaha::math {
template <typename T> class TensorWrapper;
}

namespace hahaha::compute {

/**
 * @brief Top-level dispatcher for device-specific computations.
 *
 * This class serves as a central point to route tensor operations to their
 * respective hardware-optimized implementations (CPU, SIMD, GPU, etc.).
 * It decouples math logic in TensorWrapper from device-specific kernels.
 */
template <typename T>
class DeviceComputeDispatcher {
  public:
    static void dispatchBinary(Operator op,
                               const math::TensorWrapper<T>& lhs,
                               const math::TensorWrapper<T>& rhs,
                               math::TensorWrapper<T>& res) {
        auto device = lhs.getDevice();
        if (device.type == DeviceType::CPU) {
            // Placeholder: Call CPU kernels
            // In a real implementation, this would call specialized functions
            // from hahaha::compute::cpu namespace
            size_t size = lhs.getSize();
            auto* l_ptr = lhs.data_.getData().get();
            auto* r_ptr = rhs.data_.getData().get();
            auto* res_ptr = res.data_.getData().get();

            for (size_t i = 0; i < size; ++i) {
                switch (op) {
                    case Operator::Add: res_ptr[i] = l_ptr[i] + r_ptr[i]; break;
                    case Operator::Sub: res_ptr[i] = l_ptr[i] - r_ptr[i]; break;
                    case Operator::Mul: res_ptr[i] = l_ptr[i] * r_ptr[i]; break;
                    case Operator::Div: res_ptr[i] = l_ptr[i] / r_ptr[i]; break;
                    default: throw std::runtime_error("Unsupported binary op");
                }
            }
        } else if (device.type == DeviceType::GPU) {
            throw std::runtime_error("GPU dispatch not yet implemented");
        } else {
            throw std::runtime_error("Unsupported device type for dispatch");
        }
    }

    static void dispatchMatMul(const math::TensorWrapper<T>& lhs,
                               const math::TensorWrapper<T>& rhs,
                               math::TensorWrapper<T>& res) {
        auto device = lhs.getDevice();
        if (device.type == DeviceType::CPU) {
            // Placeholder for CPU MatMul
        } else {
            throw std::runtime_error("MatMul dispatch not yet implemented");
        }
    }
};

} // namespace hahaha::compute

#endif // HAHAHA_COMPUTE_DEVICE_COMPUTE_DISPATCHER_H
