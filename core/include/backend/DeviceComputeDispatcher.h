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

#ifndef HAHAHA_BACKEND_DEVICE_COMPUTE_DISPATCHER_H
#define HAHAHA_BACKEND_DEVICE_COMPUTE_DISPATCHER_H

#include <stdexcept>

#include "backend/Device.h"
#include "common/Operator.h"

namespace hahaha::math {
template <typename T> class TensorWrapper;
} // namespace hahaha::math

namespace hahaha::backend {

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
    static void dispatchBinary(common::Operator op,
                               const math::TensorWrapper<T>& lhs,
                               const math::TensorWrapper<T>& rhs,
                               math::TensorWrapper<T>& res) {
        auto device = lhs.getDevice();
        if (device.type == backend::DeviceType::CPU) {
            // Placeholder: Call CPU kernels
            // In a real implementation, this would call specialized functions
            // from hahaha::backend::cpu namespace
            size_t size = lhs.getSize();
            auto* lPtr = lhs.data_.getData().get();
            auto* rPtr = rhs.data_.getData().get();
            auto* resPtr = res.data_.getData().get();

            for (size_t i = 0; i < size; ++i) {
                switch (op) {
                    case common::Operator::Add: resPtr[i] = lPtr[i] + rPtr[i]; break;
                    case common::Operator::Sub: resPtr[i] = lPtr[i] - rPtr[i]; break;
                    case common::Operator::Mul: resPtr[i] = lPtr[i] * rPtr[i]; break;
                    case common::Operator::Div:
                        if (rPtr[i] == T(0)) {
                            throw std::runtime_error("Division by zero");
                        }
                        resPtr[i] = lPtr[i] / rPtr[i];
                        break;
                    default: throw std::runtime_error("Unsupported binary op");
                }
            }
        } else if (device.type == backend::DeviceType::GPU) {
            throw std::runtime_error("GPU dispatch not yet implemented");
        } else {
            throw std::runtime_error("Unsupported device type for dispatch");
        }
    }

    static void dispatchMatMul(const math::TensorWrapper<T>& lhs,
                               const math::TensorWrapper<T>& rhs,
                               math::TensorWrapper<T>& res) {
        auto device = lhs.getDevice();
        if (device.type == backend::DeviceType::CPU) {
            const auto& lhsDims = lhs.getShape().getDims();
            const auto& rhsDims = rhs.getShape().getDims();
            
            size_t rows = lhsDims[0];
            size_t cols = rhsDims[1];
            size_t inner = lhsDims[1];

            auto* lPtr = lhs.data_.getData().get();
            auto* rPtr = rhs.data_.getData().get();
            auto* resPtr = res.data_.getData().get();

            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    T sum = T(0);
                    for (size_t k = 0; k < inner; ++k) {
                        sum += lPtr[i * inner + k] * rPtr[k * cols + j];
                    }
                    resPtr[i * cols + j] = sum;
                }
            }
        } else {
            throw std::runtime_error("MatMul dispatch not yet implemented");
        }
    }
};

} // namespace hahaha::backend

#endif // HAHAHA_BACKEND_DEVICE_COMPUTE_DISPATCHER_H
