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

#ifndef HAHAHA_BACKEND_VECTORIZE_SIMD_VECTOR_H
#define HAHAHA_BACKEND_VECTORIZE_SIMD_VECTOR_H

#include <cstddef>
#include <type_traits>

namespace hahaha::backend::vectorize {

/**
 * @brief Represents a fixed-size vector for SIMD operations.
 *
 * This class abstracts away architecture-specific SIMD types and operations.
 *
 * @tparam T The numeric type of elements.
 * @tparam Width The number of elements in the SIMD vector.
 */
template <typename T, size_t Width> class SimdVector {
  public:
    static_assert(std::is_arithmetic_v<T>,
                  "SimdVector only supports arithmetic types.");

    SimdVector() = default;

    /**
     * @brief Load data from a memory location into the SIMD vector.
     * @param ptr Pointer to the data to load.
     */
    void load(const T* ptr);

    /**
     * @brief Store the SIMD vector data to a memory location.
     * @param ptr Pointer to the memory location.
     */
    void store(T* ptr) const;

    /**
     * @brief Perform element-wise addition with another SIMD vector.
     * @param other The other SIMD vector.
     * @return SimdVector result of the addition.
     */
    SimdVector add(const SimdVector& other) const;

    /**
     * @brief Perform element-wise multiplication with another SIMD vector.
     * @param other The other SIMD vector.
     * @return SimdVector result of the multiplication.
     */
    SimdVector multiply(const SimdVector& other) const;

    /**
     * @brief Broadcast a single value to all elements of the SIMD vector.
     * @param value The value to broadcast.
     */
    void broadcast(T value);
};

} // namespace hahaha::backend::vectorize

#endif // HAHAHA_BACKEND_VECTORIZE_SIMD_VECTOR_H
