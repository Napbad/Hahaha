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
//
//  Contributors:
//  Napbad (napbad.sen@gmail.com) (https://github.com/Napbad)
//

//
// Created by napbad on 3/28/26.
//

#ifndef HAHAHA_SCALAR_H_5BA7003C29D9461C92F6F2B9CD8AA86F
#define HAHAHA_SCALAR_H_5BA7003C29D9461C92F6F2B9CD8AA86F

#include <compare>
#include <iosfwd>

#include "defines.h"
#include "backend/CommonPointer.h"

namespace h3::core::math {

/// Single typed element backed by a \ref backend::CommonPointer (owning or view into tensor storage).
/// Views must not outlive the owning storage; owning scalars release memory in the destructor.
class Scalar {
public:
    Scalar(DataType dtype,
           const backend::CommonPointer& data,
           bool isView);

    Scalar(const Scalar& other);
    Scalar(Scalar&& other) noexcept;
    Scalar(DataType dtype, Int32 int32, backend::Device device);
    Scalar(DataType dtype, Int32 int32);

    Scalar& operator=(const Scalar& other);
    Scalar& operator=(Scalar&& other) noexcept;

    ~Scalar();

    [[nodiscard]] DataType dtype() const {
        return m_dtype;
    }

    [[nodiscard]] backend::CommonPointer data() const {
        return m_data;
    }

    [[nodiscard]] backend::Device device() const {
        return m_data.device();

    }

    /// Byte size of the stored element (same as \ref sizeOf(dtype())).
    [[nodiscard]] SizeT sizeBytes() const;

    [[nodiscard]] bool isView() const {
        return m_isView;
    }

    /// Reinterprets the bits under this scalar as \p dtype (size must match \ref sizeOf(dtype)).
    void reinterpret(DataType dtype);

    template <typename T>
    T* as() {
        return m_data.as<T>();
    }

    template <typename T>
    T* as() const {
        return m_data.as<T>();
    }

    /// Deep copy of the value: views copy the pointed-to value into a new owning buffer on the same device.
    [[nodiscard]] Scalar clone() const;

    void swap(Scalar& other) noexcept;

    // --- Factories (owning buffers) ---

    /// One element filled with zero bits.
    [[nodiscard]] static Scalar zeros(DataType dtype, const backend::Device& device);

    /// One element whose value is converted from \p value according to \p dtype.
    [[nodiscard]] static Scalar fromHostDouble(DataType dtype,
                                               const backend::Device& device,
                                               Float64 value);

    // --- Unary ---

    [[nodiscard]] Scalar operator+() const;
    [[nodiscard]] Scalar operator-() const;

    // --- Binary arithmetic (dtype promotion, result on this scalar's device) ---

    [[nodiscard]] Scalar operator+(const Scalar& other) const;
    [[nodiscard]] Scalar operator-(const Scalar& other) const;
    [[nodiscard]] Scalar operator*(const Scalar& other) const;
    [[nodiscard]] Scalar operator/(const Scalar& other) const;
    /// Integer: remainder; floating: \c std::fmod.
    [[nodiscard]] Scalar operator%(const Scalar& other) const;

    Scalar& operator+=(const Scalar& other);
    Scalar& operator-=(const Scalar& other);
    Scalar& operator*=(const Scalar& other);
    Scalar& operator/=(const Scalar& other);
    Scalar& operator%=(const Scalar& other);

    // --- Bitwise (integer dtypes only) ---

    [[nodiscard]] Scalar operator~() const;
    [[nodiscard]] Scalar operator&(const Scalar& other) const;
    [[nodiscard]] Scalar operator|(const Scalar& other) const;
    [[nodiscard]] Scalar operator^(const Scalar& other) const;

    Scalar& operator&=(const Scalar& other);
    Scalar& operator|=(const Scalar& other);
    Scalar& operator^=(const Scalar& other);

    /// Shift amount is taken as an unsigned integer scalar (must fit in \c unsigned for C++ shift rules).
    [[nodiscard]] Scalar operator<<(const Scalar& other) const;
    [[nodiscard]] Scalar operator>>(const Scalar& other) const;

    Scalar& operator<<=(const Scalar& other);
    Scalar& operator>>=(const Scalar& other);

    // --- Comparison (value-based, after dtype promotion) ---

    [[nodiscard]] std::strong_ordering operator<=>(const Scalar& other) const;

private:
    DataType m_dtype;
    backend::CommonPointer m_data;
    bool m_isView;
};

inline void swap(Scalar& a, Scalar& b) noexcept {
    a.swap(b);
}

std::ostream& operator<<(std::ostream& os, const Scalar& s);

} // namespace h3::core::math

#endif // HAHAHA_SCALAR_H_5BA7003C29D9461C92F6F2B9CD8AA86F
