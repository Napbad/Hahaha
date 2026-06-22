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

#ifndef HAHAHA_COMPUTECONTEXT_H_74F1A022BF2C4C4AB036EA43FE4E8209
#define HAHAHA_COMPUTECONTEXT_H_74F1A022BF2C4C4AB036EA43FE4E8209

#include "backend/Device.h"
#include "defines.h"

namespace h3::core::compute {

/// Runtime execution context passed to every operator executor.
/// Similar to TensorFlow's OpKernelContext, it carries the chosen device,
/// datatype and future execution resources such as streams/allocators.
class ComputeContext {
public:
    ComputeContext(const Operator op, const backend::Device device, const DataType dataType)
        : m_operator(op), m_device(device), m_dataType(dataType) {
    }

    [[nodiscard]] const backend::Device& device() const {
        return m_device;
    }

    [[nodiscard]] backend::DeviceType deviceType() const {
        return m_device.type();
    }

    [[nodiscard]] DataType dataType() const {
        return m_dataType;
    }

    [[nodiscard]] Operator op() const {
        return m_operator;
    }

private:
    Operator m_operator;
    backend::Device m_device;
    DataType m_dataType;
};

} // namespace h3::core::compute

#endif // HAHAHA_COMPUTECONTEXT_H_74F1A022BF2C4C4AB036EA43FE4E8209
