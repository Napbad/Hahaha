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
// Created by napbad on 3/29/26.
//

#include "math/Scalar.h"

#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <memory>
#include <ostream>
#include <utility>

#include "backend/MemoryManager.h"
#include "utils/handler/exception_handler.h"

namespace h3::core::math {

namespace {

using backend::CommonPointer;
using backend::Device;
using backend::DeviceType;
using backend::MemoryManager;

[[nodiscard]] bool isFloat(const DataType d) {
    return d == DataType::Float32 || d == DataType::Float64;
}

[[nodiscard]] bool isSignedInt(const DataType d) {
    switch (d) {
    case DataType::Int8:
    case DataType::Int16:
    case DataType::Int32:
    case DataType::Int64:
        return true;
    default:
        return false;
    }
}

[[nodiscard]] bool isUnsignedInt(const DataType d) {
    switch (d) {
    case DataType::UInt8:
    case DataType::UInt16:
    case DataType::UInt32:
    case DataType::UInt64:
        return true;
    default:
        return false;
    }
}

[[nodiscard]] bool isInteger(const DataType d) {
    return isSignedInt(d) || isUnsignedInt(d);
}

/// Resolves a memory manager for \p device (falls back to default CPU manager when unregistered).
[[nodiscard]] std::shared_ptr<MemoryManager> managerFor(const Device& device) {
    if (auto m = backend::getMemoryManagerOn(device)) {
        return m;
    }
    if (device.type() == DeviceType::CPU && device.index() == 0) {
        return backend::getDefaultMemoryManager();
    }
    ThrowRuntime("No MemoryManager registered for device {}", device.toString());
}

[[nodiscard]] CommonPointer allocateOne(const DataType dtype, const Device& device) {
    const SizeT n = sizeOf(dtype);
    const auto mgr = managerFor(device);
    auto p = mgr->allocate(n);
    if (!p) {
        ThrowRuntime("Scalar allocation failed: {}", p.error().message());
    }
    return *p;
}

void copyBytes(const CommonPointer& dst,
               const CommonPointer& src,
               const SizeT n,
               const Device& dev) {
    auto mgr = managerFor(dev);
    if (const auto r = mgr->copy(dst, src, n); !r) {
        ThrowRuntime("Scalar memory copy failed: {}", r.error().message());
    }
}

// --- Load / store as widened types for arithmetic ---

[[nodiscard]] Float64 loadAsFloat64(const Scalar& s) {
    switch (s.dtype()) {
    case DataType::Float32:
        return *s.as<Float32>();
    case DataType::Float64:
        return *s.as<Float64>();
    case DataType::Int8:
        return *s.as<Int8>();
    case DataType::Int16:
        return *s.as<Int16>();
    case DataType::Int32:
        return *s.as<Int32>();
    case DataType::Int64:
        return static_cast<Float64>(*s.as<Int64>());
    case DataType::UInt8:
        return *s.as<UInt8>();
    case DataType::UInt16:
        return *s.as<UInt16>();
    case DataType::UInt32:
        return *s.as<UInt32>();
    case DataType::UInt64:
        return static_cast<Float64>(*s.as<UInt64>());
    default:
        ThrowLogic("Unhandled DataType in loadAsFloat64");
    }
}

void storeFloat64(Scalar& s, const Float64 v) {
    switch (s.dtype()) {
    case DataType::Float32:
        *s.as<Float32>() = static_cast<Float32>(v);
        break;
    case DataType::Float64:
        *s.as<Float64>() = v;
        break;
    case DataType::Int8:
        *s.as<Int8>() = static_cast<Int8>(std::llround(v));
        break;
    case DataType::Int16:
        *s.as<Int16>() = static_cast<Int16>(std::llround(v));
        break;
    case DataType::Int32:
        *s.as<Int32>() = static_cast<Int32>(std::llround(v));
        break;
    case DataType::Int64:
        *s.as<Int64>() = static_cast<Int64>(std::llround(v));
        break;
    case DataType::UInt8:
        *s.as<UInt8>() = static_cast<UInt8>(std::llround(std::max<Float64>(0, v)));
        break;
    case DataType::UInt16:
        *s.as<UInt16>() = static_cast<UInt16>(std::llround(std::max<Float64>(0, v)));
        break;
    case DataType::UInt32:
        *s.as<UInt32>() = static_cast<UInt32>(std::llround(std::max<Float64>(0, v)));
        break;
    case DataType::UInt64:
        *s.as<UInt64>() = static_cast<UInt64>(std::llround(std::max<Float64>(0, v)));
        break;
    default:
        ThrowLogic("Unhandled DataType in storeFloat64");
    }
}

[[nodiscard]] UInt64 loadAsUInt64(const Scalar& s) {
    switch (s.dtype()) {
    case DataType::UInt8:
        return *s.as<UInt8>();
    case DataType::UInt16:
        return *s.as<UInt16>();
    case DataType::UInt32:
        return *s.as<UInt32>();
    case DataType::UInt64:
        return *s.as<UInt64>();
    case DataType::Int8:
        return static_cast<UInt64>(std::max<Int32>(0, *s.as<Int8>()));
    case DataType::Int16:
        return static_cast<UInt64>(std::max<Int32>(0, *s.as<Int16>()));
    case DataType::Int32:
        return static_cast<UInt64>(std::max<Int64>(0, *s.as<Int32>()));
    case DataType::Int64:
        return static_cast<UInt64>(std::max<Int64>(0, *s.as<Int64>()));
    default:
        ThrowInvalid("Bit / shift operations require integer dtypes");
    }
}

[[nodiscard]] Int64 loadAsInt64(const Scalar& s) {
    switch (s.dtype()) {
    case DataType::Int8:
        return *s.as<Int8>();
    case DataType::Int16:
        return *s.as<Int16>();
    case DataType::Int32:
        return *s.as<Int32>();
    case DataType::Int64:
        return *s.as<Int64>();
    case DataType::UInt8:
        return *s.as<UInt8>();
    case DataType::UInt16:
        return *s.as<UInt16>();
    case DataType::UInt32:
        return *s.as<UInt32>();
    case DataType::UInt64: {
        const UInt64 u = *s.as<UInt64>();
        if (u > static_cast<UInt64>(std::numeric_limits<Int64>::max())) {
            return std::numeric_limits<Int64>::max();
        }
        return static_cast<Int64>(u);
    }
    default:
        ThrowInvalid("Expected integer dtype");
    }
}

void storeInt64(Scalar& s, const Int64 v) {
    switch (s.dtype()) {
    case DataType::Int8:
        *s.as<Int8>() = static_cast<Int8>(v);
        break;
    case DataType::Int16:
        *s.as<Int16>() = static_cast<Int16>(v);
        break;
    case DataType::Int32:
        *s.as<Int32>() = static_cast<Int32>(v);
        break;
    case DataType::Int64:
        *s.as<Int64>() = v;
        break;
    case DataType::UInt8:
        *s.as<UInt8>() = static_cast<UInt8>(std::max<Int64>(0, v));
        break;
    case DataType::UInt16:
        *s.as<UInt16>() = static_cast<UInt16>(std::max<Int64>(0, v));
        break;
    case DataType::UInt32:
        *s.as<UInt32>() = static_cast<UInt32>(std::max<Int64>(0, v));
        break;
    case DataType::UInt64:
        *s.as<UInt64>() = static_cast<UInt64>(std::max<Int64>(0, v));
        break;
    default:
        ThrowLogic("storeInt64: not integer");
    }
}

void storeUInt64(Scalar& s, const UInt64 v) {
    switch (s.dtype()) {
    case DataType::UInt8:
        *s.as<UInt8>() = static_cast<UInt8>(v);
        break;
    case DataType::UInt16:
        *s.as<UInt16>() = static_cast<UInt16>(v);
        break;
    case DataType::UInt32:
        *s.as<UInt32>() = static_cast<UInt32>(v);
        break;
    case DataType::UInt64:
        *s.as<UInt64>() = v;
        break;
    case DataType::Int8:
    case DataType::Int16:
    case DataType::Int32:
    case DataType::Int64:
        storeInt64(s, static_cast<Int64>(v));
        break;
    default:
        ThrowLogic("storeUInt64: not integer");
    }
}

/// Result dtype for +, -, *, / (and related) following a practical numeric tower.
[[nodiscard]] DataType promoteArithmetic(const DataType a, const DataType b) {
    if (isFloat(a) || isFloat(b)) {
        if (a == DataType::Float64 || b == DataType::Float64) {
            return DataType::Float64;
        }
        return DataType::Float32;
    }
    if (!isInteger(a) || !isInteger(b)) {
        ThrowInvalid("promoteArithmetic: unsupported DataType combination");
    }
    // Integer-only: prefer signed Int64 if any operand is signed or mixed signedness.
    if (isSignedInt(a) && isSignedInt(b)) {
        return DataType::Int64;
    }
    if (isUnsignedInt(a) && isUnsignedInt(b)) {
        return DataType::UInt64;
    }
    return DataType::Int64;
}

[[nodiscard]] DataType promoteBitwise(const DataType a, const DataType b) {
    if (!isInteger(a) || !isInteger(b)) {
        ThrowInvalid("Bitwise operations require integer dtypes");
    }
    if (isSignedInt(a) && isSignedInt(b)) {
        return DataType::Int64;
    }
    if (isUnsignedInt(a) && isUnsignedInt(b)) {
        return DataType::UInt64;
    }
    return DataType::Int64;
}

void assertSameDevice(const Scalar& x, const Scalar& y) {
    if (x.device() != y.device()) {
        ThrowInvalid(
            "Scalar operands must live on the same device ({} vs {})",
            x.device().toString(),
            y.device().toString());
    }
}

[[nodiscard]] Scalar makeResult(const DataType dtype,
                                const Device& device,
                                const Float64 value) {
    Scalar s(dtype, allocateOne(dtype, device), false);
    storeFloat64(s, value);
    return s;
}

[[nodiscard]] Scalar makeResultInt(const DataType dtype,
                                   const Device& device,
                                   const Int64 value) {
    Scalar s(dtype, allocateOne(dtype, device), false);
    storeInt64(s, value);
    return s;
}

[[nodiscard]] Scalar makeResultUInt(const DataType dtype,
                                    const Device& device,
                                    const UInt64 value) {
    Scalar s(dtype, allocateOne(dtype, device), false);
    storeUInt64(s, value);
    return s;
}

} // namespace

SizeT Scalar::sizeBytes() const {
    return sizeOf(m_dtype);
}

Scalar::Scalar(const DataType dtype,
               const CommonPointer& data,
               const bool isView) : m_dtype(dtype),
                                    m_data(data),
                                    m_isView(isView) {
}

Scalar::Scalar(const Scalar& other) : m_dtype(other.m_dtype) {
    if (other.m_isView) {
        m_data = other.m_data;
        m_isView = true;
    } else {
        m_data = allocateOne(other.m_dtype, other.device());
        copyBytes(m_data, other.m_data, sizeOf(other.m_dtype), other.device());
        m_isView = false;
    }
}

Scalar::Scalar(Scalar&& other) noexcept
    : m_dtype(other.m_dtype),
      m_data(other.m_data),
      m_isView(other.m_isView) {
    other.m_data = CommonPointer();
    other.m_isView = true;
}

Scalar& Scalar::operator=(const Scalar& other) {
    if (this == &other) {
        return *this;
    }
    // View receiving an owning rhs: copy bytes in place (keep aliasing into tensor storage).
    if (m_isView && !other.m_isView && m_dtype == other.m_dtype &&
        device() == other.device()) {
        copyBytes(m_data, other.m_data, sizeOf(m_dtype), device());
        return *this;
    }
    if (!m_isView) {
        m_data.destroy();
    }
    m_dtype = other.m_dtype;
    if (other.m_isView) {
        m_data = other.m_data;
        m_isView = true;
    } else {
        m_data = allocateOne(other.m_dtype, other.device());
        copyBytes(m_data, other.m_data, sizeOf(other.m_dtype), other.device());
        m_isView = false;
    }
    return *this;
}

Scalar& Scalar::operator=(Scalar&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    if (!m_isView) {
        m_data.destroy();
    }
    m_dtype = other.m_dtype;
    m_data = other.m_data;
    m_isView = other.m_isView;
    other.m_data = CommonPointer();
    other.m_isView = true;
    return *this;
}

Scalar::~Scalar() {
    if (!m_isView) {
        m_data.destroy();
    }
}

void Scalar::reinterpret(const DataType dtype) {
    if (sizeOf(dtype) != sizeOf(m_dtype)) {
        ThrowInvalid(
            "reinterpret requires matching sizes ({} vs {} bytes)",
            sizeOf(dtype),
            sizeOf(m_dtype));
    }
    m_dtype = dtype;
}

Scalar Scalar::clone() const {
    const auto p = allocateOne(m_dtype, device());
    Scalar res(m_dtype, p, false);
    copyBytes(res.m_data, m_data, sizeOf(m_dtype), device());
    return res;
}

void Scalar::swap(Scalar& other) noexcept {
    using std::swap;
    swap(m_dtype, other.m_dtype);
    swap(m_data, other.m_data);
    swap(m_isView, other.m_isView);
}

Scalar Scalar::zeros(const DataType dtype, const Device& device) {
    const auto n = sizeOf(dtype);
    auto p = allocateOne(dtype, device);
    std::memset(p.get(), 0, static_cast<std::size_t>(n));
    return Scalar(dtype, p, false);
}

Scalar Scalar::fromHostDouble(const DataType dtype,
                              const Device& device,
                              const Float64 value) {
    Scalar s(dtype, allocateOne(dtype, device), false);
    storeFloat64(s, value);
    return s;
}

Scalar Scalar::operator+() const {
    return clone();
}

Scalar Scalar::operator-() const {
    const DataType out = m_dtype;
    if (isFloat(m_dtype)) {
        return makeResult(out, device(), -loadAsFloat64(*this));
    }
    if (isUnsignedInt(m_dtype)) {
        const UInt64 v = loadAsUInt64(*this);
        return makeResultUInt(out, device(), static_cast<UInt64>(0ull - v));
    }
    if (isSignedInt(m_dtype)) {
        return makeResultInt(out, device(), -loadAsInt64(*this));
    }
    ThrowInvalid("operator-: unsupported dtype");
}

Scalar Scalar::operator+(const Scalar& other) const {
    assertSameDevice(*this, other);
    const DataType out = promoteArithmetic(m_dtype, other.m_dtype);
    if (isFloat(out)) {
        return makeResult(out, device(), loadAsFloat64(*this) + loadAsFloat64(other));
    }
    if (out == DataType::UInt64) {
        return makeResultUInt(out, device(), loadAsUInt64(*this) + loadAsUInt64(other));
    }
    return makeResultInt(out, device(), loadAsInt64(*this) + loadAsInt64(other));
}

Scalar Scalar::operator-(const Scalar& other) const {
    assertSameDevice(*this, other);
    const DataType out = promoteArithmetic(m_dtype, other.m_dtype);
    if (isFloat(out)) {
        return makeResult(out, device(), loadAsFloat64(*this) - loadAsFloat64(other));
    }
    if (out == DataType::UInt64) {
        return makeResultUInt(out, device(), loadAsUInt64(*this) - loadAsUInt64(other));
    }
    return makeResultInt(out, device(), loadAsInt64(*this) - loadAsInt64(other));
}

Scalar Scalar::operator*(const Scalar& other) const {
    assertSameDevice(*this, other);
    const DataType out = promoteArithmetic(m_dtype, other.m_dtype);
    if (isFloat(out)) {
        return makeResult(out, device(), loadAsFloat64(*this) * loadAsFloat64(other));
    }
    if (out == DataType::UInt64) {
        return makeResultUInt(out, device(), loadAsUInt64(*this) * loadAsUInt64(other));
    }
    return makeResultInt(out, device(), loadAsInt64(*this) * loadAsInt64(other));
}

Scalar Scalar::operator/(const Scalar& other) const {
    assertSameDevice(*this, other);
    const DataType out = promoteArithmetic(m_dtype, other.m_dtype);
    if (isFloat(out)) {
        const Float64 denom = loadAsFloat64(other);
        if (denom == 0.0) {
            ThrowInvalid("Scalar division by zero");
        }
        return makeResult(out, device(), loadAsFloat64(*this) / denom);
    }
    if (out == DataType::UInt64) {
        const UInt64 denom = loadAsUInt64(other);
        if (denom == 0) {
            ThrowInvalid("Scalar division by zero");
        }
        return makeResultUInt(out, device(), loadAsUInt64(*this) / denom);
    }
    const Int64 denom = loadAsInt64(other);
    if (denom == 0) {
        ThrowInvalid("Scalar division by zero");
    }
    return makeResultInt(out, device(), loadAsInt64(*this) / denom);
}

Scalar Scalar::operator%(const Scalar& other) const {
    assertSameDevice(*this, other);
    if (isFloat(m_dtype) || isFloat(other.m_dtype)) {
        const DataType out = promoteArithmetic(m_dtype, other.m_dtype);
        const Float64 a = loadAsFloat64(*this);
        const Float64 b = loadAsFloat64(other);
        if (b == 0.0) {
            ThrowInvalid("Scalar fmod: division by zero");
        }
        return makeResult(out, device(), std::fmod(a, b));
    }
    const DataType out = promoteArithmetic(m_dtype, other.m_dtype);
    if (out == DataType::UInt64) {
        const UInt64 b = loadAsUInt64(other);
        if (b == 0) {
            ThrowInvalid("Scalar modulo: division by zero");
        }
        return makeResultUInt(out, device(), loadAsUInt64(*this) % b);
    }
    const Int64 b = loadAsInt64(other);
    if (b == 0) {
        ThrowInvalid("Scalar modulo: division by zero");
    }
    return makeResultInt(out, device(), loadAsInt64(*this) % b);
}

Scalar& Scalar::operator+=(const Scalar& other) {
    const Scalar sum = *this + other;
    if (m_isView) {
        if (sum.dtype() != m_dtype) {
            ThrowInvalid(
                "Compound assignment on a view cannot change element type (got promoted dtype)");
        }
        copyBytes(m_data, sum.m_data, sizeBytes(), device());
        return *this;
    }
    *this = std::move(sum);
    return *this;
}

Scalar& Scalar::operator-=(const Scalar& other) {
    const Scalar diff = *this - other;
    if (m_isView) {
        if (diff.dtype() != m_dtype) {
            ThrowInvalid(
                "Compound assignment on a view cannot change element type (got promoted dtype)");
        }
        copyBytes(m_data, diff.m_data, sizeBytes(), device());
        return *this;
    }
    *this = std::move(diff);
    return *this;
}

Scalar& Scalar::operator*=(const Scalar& other) {
    const Scalar prod = *this * other;
    if (m_isView) {
        if (prod.dtype() != m_dtype) {
            ThrowInvalid(
                "Compound assignment on a view cannot change element type (got promoted dtype)");
        }
        copyBytes(m_data, prod.m_data, sizeBytes(), device());
        return *this;
    }
    *this = std::move(prod);
    return *this;
}

Scalar& Scalar::operator/=(const Scalar& other) {
    const Scalar quot = *this / other;
    if (m_isView) {
        if (quot.dtype() != m_dtype) {
            ThrowInvalid(
                "Compound assignment on a view cannot change element type (got promoted dtype)");
        }
        copyBytes(m_data, quot.m_data, sizeBytes(), device());
        return *this;
    }
    *this = std::move(quot);
    return *this;
}

Scalar& Scalar::operator%=(const Scalar& other) {
    const Scalar rem = *this % other;
    if (m_isView) {
        if (rem.dtype() != m_dtype) {
            ThrowInvalid(
                "Compound assignment on a view cannot change element type (got promoted dtype)");
        }
        copyBytes(m_data, rem.m_data, sizeBytes(), device());
        return *this;
    }
    *this = std::move(rem);
    return *this;
}

Scalar Scalar::operator~() const {
    if (!isInteger(m_dtype)) {
        ThrowInvalid("Bitwise NOT requires an integer dtype");
    }
    const DataType out = isUnsignedInt(m_dtype) ? DataType::UInt64 : DataType::Int64;
    if (out == DataType::UInt64) {
        return makeResultUInt(out, device(), ~loadAsUInt64(*this));
    }
    return makeResultInt(out, device(), ~loadAsInt64(*this));
}

Scalar Scalar::operator&(const Scalar& other) const {
    assertSameDevice(*this, other);
    const DataType out = promoteBitwise(m_dtype, other.m_dtype);
    if (out == DataType::UInt64) {
        return makeResultUInt(out, device(), loadAsUInt64(*this) & loadAsUInt64(other));
    }
    return makeResultInt(out, device(), loadAsInt64(*this) & loadAsInt64(other));
}

Scalar Scalar::operator|(const Scalar& other) const {
    assertSameDevice(*this, other);
    const DataType out = promoteBitwise(m_dtype, other.m_dtype);
    if (out == DataType::UInt64) {
        return makeResultUInt(out, device(), loadAsUInt64(*this) | loadAsUInt64(other));
    }
    return makeResultInt(out, device(), loadAsInt64(*this) | loadAsInt64(other));
}

Scalar Scalar::operator^(const Scalar& other) const {
    assertSameDevice(*this, other);
    const DataType out = promoteBitwise(m_dtype, other.m_dtype);
    if (out == DataType::UInt64) {
        return makeResultUInt(out, device(), loadAsUInt64(*this) ^ loadAsUInt64(other));
    }
    return makeResultInt(out, device(), loadAsInt64(*this) ^ loadAsInt64(other));
}

Scalar& Scalar::operator&=(const Scalar& other) {
    *this = *this & other;
    return *this;
}

Scalar& Scalar::operator|=(const Scalar& other) {
    *this = *this | other;
    return *this;
}

Scalar& Scalar::operator^=(const Scalar& other) {
    *this = *this ^ other;
    return *this;
}

Scalar Scalar::operator<<(const Scalar& other) const {
    assertSameDevice(*this, other);
    if (!isInteger(m_dtype)) {
        ThrowInvalid("Shift left requires integer left-hand dtype");
    }
    const unsigned amount = static_cast<unsigned>(loadAsUInt64(other));
    constexpr unsigned kMax = 64;
    if (amount >= kMax) {
        return zeros(m_dtype, device());
    }
    if (isUnsignedInt(m_dtype)) {
        return makeResultUInt(promoteBitwise(m_dtype, m_dtype),
                              device(),
                              loadAsUInt64(*this) << amount);
    }
    return makeResultInt(promoteBitwise(m_dtype, m_dtype),
                         device(),
                         loadAsInt64(*this) << amount);
}

Scalar Scalar::operator>>(const Scalar& other) const {
    assertSameDevice(*this, other);
    if (!isInteger(m_dtype)) {
        ThrowInvalid("Shift right requires integer left-hand dtype");
    }
    const unsigned amount = static_cast<unsigned>(loadAsUInt64(other));
    constexpr unsigned kMax = 64;
    if (amount >= kMax) {
        return zeros(m_dtype, device());
    }
    if (isUnsignedInt(m_dtype)) {
        return makeResultUInt(promoteBitwise(m_dtype, m_dtype),
                              device(),
                              loadAsUInt64(*this) >> amount);
    }
    return makeResultInt(promoteBitwise(m_dtype, m_dtype),
                         device(),
                         loadAsInt64(*this) >> amount);
}

Scalar& Scalar::operator<<=(const Scalar& other) {
    *this = *this << other;
    return *this;
}

Scalar& Scalar::operator>>=(const Scalar& other) {
    *this = *this >> other;
    return *this;
}

std::strong_ordering Scalar::operator<=>(const Scalar& other) const {
    if (device() != other.device()) {
        return device().index() <=> other.device().index();
    }
    const DataType p = promoteArithmetic(m_dtype, other.m_dtype);
    if (isFloat(p)) {
        const Float64 a = loadAsFloat64(*this);
        const Float64 b = loadAsFloat64(other);
        const bool na = std::isnan(a);
        const bool nb = std::isnan(b);
        if (na != nb) {
            return na ? std::strong_ordering::less : std::strong_ordering::greater;
        }
        if (na && nb) {
            return std::strong_ordering::equal;
        }
        if (a < b) {
            return std::strong_ordering::less;
        }
        if (a > b) {
            return std::strong_ordering::greater;
        }
        return std::strong_ordering::equal;
    }
    if (p == DataType::UInt64) {
        const UInt64 a = loadAsUInt64(*this);
        const UInt64 b = loadAsUInt64(other);
        return a <=> b;
    }
    const Int64 a = loadAsInt64(*this);
    const Int64 b = loadAsInt64(other);
    return a <=> b;
}

std::ostream& operator<<(std::ostream& os, const Scalar& s) {
    os << "Scalar{dtype=" << static_cast<int>(s.dtype())
       << ", device=" << s.device().toString();
    if (isFloat(s.dtype())) {
        os << ", value=" << std::setprecision(17) << loadAsFloat64(s);
    } else if (isUnsignedInt(s.dtype())) {
        os << ", value=" << loadAsUInt64(s);
    } else if (isSignedInt(s.dtype())) {
        os << ", value=" << loadAsInt64(s);
    } else {
        os << ", value=?";
    }
    os << ", view=" << (s.isView() ? "true" : "false") << "}";
    return os;
}

} // namespace h3::core::math
