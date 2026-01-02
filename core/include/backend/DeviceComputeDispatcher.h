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
template <typename T> class DeviceComputeDispatcher {
  public:
    static void dispatchBinary(common::Operator op,
                               const math::TensorWrapper<T>& lhs,
                               const math::TensorWrapper<T>& rhs,
                               math::TensorWrapper<T>& res) {
        auto device = lhs.getDevice();
        if (device.type == backend::DeviceType::CPU) {
            size_t size = lhs.getTotalSize();
            auto* lPtr = lhs.data_.getData().get();
            auto* rPtr = rhs.data_.getData().get();
            auto* resPtr = res.data_.getData().get();

            switch (op) {
            case common::Operator::Add:
                for (size_t i = 0; i < size; ++i)
                    resPtr[i] = lPtr[i] + rPtr[i];
                break;
            case common::Operator::Sub:
                for (size_t i = 0; i < size; ++i)
                    resPtr[i] = lPtr[i] - rPtr[i];
                break;
            case common::Operator::Mul:
                for (size_t i = 0; i < size; ++i)
                    resPtr[i] = lPtr[i] * rPtr[i];
                break;
            case common::Operator::Div:
                for (size_t i = 0; i < size; ++i) {
                    if (rPtr[i] == T(0))
                        throw std::runtime_error("Division by zero");
                    resPtr[i] = lPtr[i] / rPtr[i];
                }
                break;
            default:
                throw std::runtime_error("Unsupported binary op");
            }
        } else if (device.type == backend::DeviceType::GPU) {
            throw std::runtime_error("GPU dispatch not yet implemented");
        } else {
            throw std::runtime_error("Unsupported device type for dispatch");
        }
    }

    static void dispatchScalar(common::Operator op,
                               const math::TensorWrapper<T>& lhs,
                               T rhs,
                               math::TensorWrapper<T>& res) {
        auto device = lhs.getDevice();
        if (device.type == backend::DeviceType::CPU) {
            size_t size = lhs.getTotalSize();
            auto* lPtr = lhs.data_.getData().get();
            auto* resPtr = res.data_.getData().get();

            switch (op) {
            case common::Operator::Add:
                for (size_t i = 0; i < size; ++i)
                    resPtr[i] = lPtr[i] + rhs;
                break;
            case common::Operator::Sub:
                for (size_t i = 0; i < size; ++i)
                    resPtr[i] = lPtr[i] - rhs;
                break;
            case common::Operator::Mul:
                for (size_t i = 0; i < size; ++i)
                    resPtr[i] = lPtr[i] * rhs;
                break;
            case common::Operator::Div:
                if (rhs == T(0))
                    throw std::runtime_error("Division by zero");
                for (size_t i = 0; i < size; ++i)
                    resPtr[i] = lPtr[i] / rhs;
                break;
            default:
                throw std::runtime_error("Unsupported scalar op");
            }
        } else {
            throw std::runtime_error(
                "Scalar dispatch not yet implemented for this device");
        }
    }

    static void dispatchScalar(common::Operator op,
                               T lhs,
                               const math::TensorWrapper<T>& rhs,
                               math::TensorWrapper<T>& res) {
        auto device = rhs.getDevice();
        if (device.type == backend::DeviceType::CPU) {
            size_t size = rhs.getTotalSize();
            auto* rPtr = rhs.data_.getData().get();
            auto* resPtr = res.data_.getData().get();

            switch (op) {
            case common::Operator::Add:
                for (size_t i = 0; i < size; ++i)
                    resPtr[i] = lhs + rPtr[i];
                break;
            case common::Operator::Sub:
                for (size_t i = 0; i < size; ++i)
                    resPtr[i] = lhs - rPtr[i];
                break;
            case common::Operator::Mul:
                for (size_t i = 0; i < size; ++i)
                    resPtr[i] = lhs * rPtr[i];
                break;
            case common::Operator::Div:
                for (size_t i = 0; i < size; ++i) {
                    if (rPtr[i] == T(0))
                        throw std::runtime_error("Division by zero");
                    resPtr[i] = lhs / rPtr[i];
                }
                break;
            default:
                throw std::runtime_error("Unsupported scalar op");
            }
        } else {
            throw std::runtime_error(
                "Scalar dispatch not yet implemented for this device");
        }
    }

    static void dispatchMatMul(const math::TensorWrapper<T>& lhs,
                               const math::TensorWrapper<T>& rhs,
                               math::TensorWrapper<T>& res) {
        auto device = lhs.getDevice();
        if (device.type == backend::DeviceType::CPU) {
            const auto& lhsDims = lhs.getShape();
            const auto& rhsDims = rhs.getShape();

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

    /**
     * @brief Performs res = res + alpha * x in-place.
     * @param alpha Scaling factor.
     * @param x_tensor Input tensor.
     * @param res_tensor Result tensor (updated in-place).
     */
    static void dispatchAxpy(T alpha,
                             const math::TensorWrapper<T>& x_tensor,
                             math::TensorWrapper<T>& res_tensor) {
        auto device = res_tensor.getDevice();
        if (device.type == backend::DeviceType::CPU) {
            size_t size = res_tensor.getTotalSize();
            auto* xPtr = x_tensor.data_.getData().get();
            auto* resPtr = res_tensor.data_.getData().get();

            for (size_t i = 0; i < size; ++i) {
                resPtr[i] += alpha * xPtr[i];
            }
        } else {
            throw std::runtime_error("Axpy dispatch not yet implemented");
        }
    }
};

} // namespace hahaha::backend

#endif // HAHAHA_BACKEND_DEVICE_COMPUTE_DISPATCHER_H
