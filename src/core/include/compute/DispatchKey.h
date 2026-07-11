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

#ifndef HAHAHA_DISPATCH_KEY_H
#define HAHAHA_DISPATCH_KEY_H

#include <cstdint>
#include <string_view>

#include "backend/Device.h"
#include "defines.h"

namespace h3::core::compute {

/// DispatchKey uniquely identifies a kernel implementation by encoding
/// the operation, data type, and device into a compact structure.
///
/// Layout: [op:8][dtype:8][device:8] (as uint64_t for hashing)
class DispatchKey {
public:
    using OpCode = uint8_t;
    using DtypeCode = uint8_t;
    using DeviceCode = uint8_t;

    static constexpr OpCode kMaxOps = 64;
    static constexpr DtypeCode kMaxDtypes = 16;
    static constexpr DeviceCode kMaxDevices = 8;

    constexpr DispatchKey() noexcept
        : op_(0), dtype_(0), device_(0) {}

    constexpr DispatchKey(Operator op, DataType dtype, backend::DeviceType device) noexcept
        : op_(static_cast<OpCode>(op)), 
          dtype_(static_cast<DtypeCode>(dtype)), 
          device_(static_cast<DeviceCode>(device)) {}

    [[nodiscard]] constexpr OpCode op() const noexcept { return op_; }
    [[nodiscard]] constexpr DtypeCode dtype() const noexcept { return dtype_; }
    [[nodiscard]] constexpr DeviceCode device() const noexcept { return device_; }

    [[nodiscard]] Operator opEnum() const noexcept { 
        return static_cast<Operator>(op_); 
    }
    [[nodiscard]] DataType dtypeEnum() const noexcept { 
        return static_cast<DataType>(dtype_); 
    }
    [[nodiscard]] backend::DeviceType deviceEnum() const noexcept { 
        return static_cast<backend::DeviceType>(device_); 
    }

    /// Pack into 64-bit integer for hash map key
    [[nodiscard]] constexpr uint64_t packed() const noexcept {
        return (static_cast<uint64_t>(op_) << 16) | 
               (static_cast<uint64_t>(dtype_) << 8) | 
               static_cast<uint64_t>(device_);
    }

    /// Unpack from 64-bit integer
    [[nodiscard]] static constexpr DispatchKey fromPacked(uint64_t packed) noexcept {
        return DispatchKey(
            static_cast<Operator>((packed >> 16) & 0xFF),
            static_cast<DataType>((packed >> 8) & 0xFF),
            static_cast<backend::DeviceType>(packed & 0xFF)
        );
    }

    constexpr bool operator==(const DispatchKey& other) const noexcept {
        return packed() == other.packed();
    }

    constexpr bool operator!=(const DispatchKey& other) const noexcept {
        return packed() != other.packed();
    }

    [[nodiscard]] std::string toString() const;

private:
    OpCode op_ : 6;      // Supports up to 64 operators (0-63)
    DtypeCode dtype_ : 4; // Supports up to 16 data types (0-15)
    DeviceCode device_ : 4; // Supports up to 8 devices (0-7)
};

// Ensure the struct is compact (packed bit fields may vary by compiler)
// We rely on packed() method for hashing, not sizeof()

/// Hash function specialization for std::unordered_map
struct DispatchKeyHash {
    [[nodiscard]] size_t operator()(const DispatchKey& key) const noexcept {
        return static_cast<size_t>(key.packed());
    }
};

/// Equality comparator for unordered containers
struct DispatchKeyEq {
    [[nodiscard]] bool operator()(const DispatchKey& lhs, 
                                  const DispatchKey& rhs) const noexcept {
        return lhs == rhs;
    }
};

} // namespace h3::core::compute

#endif // HAHAHA_DISPATCH_KEY_H
