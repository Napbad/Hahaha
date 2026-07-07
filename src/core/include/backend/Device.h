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

#ifndef HAHAHA_DEVICE_H_5F6E415902B445AA92F71929169DEC1E
#define HAHAHA_DEVICE_H_5F6E415902B445AA92F71929169DEC1E

#include <string>

#include "defines.h"

namespace h3::core::backend {

enum class DeviceType { CPU = 0, CUDA, UNKNOWN };

inline std::string deviceTypeToString(const DeviceType type) {
    switch (type) {
    case DeviceType::CPU:
        return "CPU";
    case DeviceType::CUDA:
        return "CUDA";
    default:
        return "UNKNOWN";
    }
}

class Device {
public:
    Device() : m_index(0), m_type(DeviceType::CPU) {
    }

    Device(const SizeT idx, const DeviceType type) : m_index(idx), m_type(type) {
    }

    [[nodiscard]] SizeT index() const {
        return m_index;
    }

    [[nodiscard]] DeviceType type() const {
        return m_type;
    }

    [[nodiscard]] std::string toString() const {
        return "Device { " + deviceTypeToString(m_type) + ": " +
            std::to_string(m_index) + " }";
    }

    bool operator==(const Device& other) const = default;

private:
    SizeT m_index;
    DeviceType m_type;
};

struct DeviceHash {
    std::size_t operator()(const Device& device) const {
        return std::hash<std::size_t>{}(device.index()) ^ 
               std::hash<int>{}(static_cast<int>(device.type())) << 1;
    }
};

static const auto DefaultCPUDevice = Device(0, DeviceType::CPU);

inline Device getDefaultCPUDevice() {
    return DefaultCPUDevice;
}

} // namespace h3::core::backend

#endif // HAHAHA_DEVICE_H_5F6E415902B445AA92F71929169DEC1E