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

#ifndef HAHAHA_BACKEND_VECTORIZE_VECTORIZED_OP_H
#define HAHAHA_BACKEND_VECTORIZE_VECTORIZED_OP_H

#include <cstddef>
#include <vector>

namespace hahaha::backend::vectorize {

/**
 * @brief Base class for vectorized operations on large data sets.
 *
 * This class provides an interface for performing operations that leverage
 * SIMD instructions for performance.
 */
class VectorizedOp {
  public:
    VectorizedOp() = default;
    virtual ~VectorizedOp() = default;

    /**
     * @brief Execute the vectorized operation on the given data.
     * @tparam T The numeric type of the data.
     * @param data_ptr Pointer to the data.
     * @param size Number of elements in the data.
     */
    template <typename T> void execute(T* data_ptr, size_t size);

    /**
     * @brief Check if the operation can be vectorized for the current
     * architecture.
     * @return bool True if vectorization is supported.
     */
    virtual bool isSupported() const = 0;

    /**
     * @brief Get the preferred vector width for the current architecture.
     * @return size_t The preferred vector width.
     */
    virtual size_t getPreferredWidth() const = 0;
};

} // namespace hahaha::backend::vectorize

#endif // HAHAHA_BACKEND_VECTORIZE_VECTORIZED_OP_H
