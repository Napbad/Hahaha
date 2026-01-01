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

#ifndef HAHAHA_BACKEND_DEVICE_H
#define HAHAHA_BACKEND_DEVICE_H

#include <cstdint>
#include <string>

namespace hahaha::backend {

/**
 * @brief Types of devices supported for computation.
 */
enum class DeviceType : std::uint8_t {
    CPU, /**< Standard Central Processing Unit. */
    GPU, /**< Graphics Processing Unit. */
    SIMD /**< Single Instruction, Multiple Data (vectorized CPU instructions).
          */
};

/**
 * @brief Represents a compute device where data resides and operations occur.
 */
struct Device {
    DeviceType type = DeviceType::CPU; /**< Type of the device. */
    std::uint8_t id =
        0; /**< Unique identifier for multiple devices of the same type. */

    /** @brief Default constructor (CPU, ID 0). */
    Device() = default;

    /**
     * @brief Construct a Device with type and ID.
     * @param deviceType Device type.
     * @param deviceId Device ID.
     */
    explicit Device(DeviceType deviceType, std::uint8_t deviceId = 0)
        : type(deviceType), id(deviceId) {
    }

    /** @brief Check if two devices are identical. */
    bool operator==(const Device& other) const {
        return type == other.type && id == other.id;
    }

    /** @brief Check if two devices are different. */
    bool operator!=(const Device& other) const {
        return !(*this == other);
    }

    /** @brief Get a string representation of the device. */
    [[nodiscard]] std::string toString() const {
        std::string deviceName;
        switch (type) {
        case DeviceType::CPU:
            deviceName = "CPU";
            break;
        case DeviceType::GPU:
            deviceName = "GPU";
            break;
        case DeviceType::SIMD:
            deviceName = "SIMD";
            break;
        }
        return deviceName + ":" + std::to_string(id);
    }
} __attribute__((aligned(2)));

} // namespace hahaha::backend

#endif // HAHAHA_BACKEND_DEVICE_H
