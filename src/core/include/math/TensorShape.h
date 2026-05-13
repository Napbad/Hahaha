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

#ifndef HAHAHA_TENSORSHAPE_H_D00755481C5B426DA11E0C244ED22F33
#define HAHAHA_TENSORSHAPE_H_D00755481C5B426DA11E0C244ED22F33
#include <algorithm>
#include <expected>
#include <string>
#include <vector>

#include "../Error.h"
#include "../defines.h"

namespace h3::core::math {
// Inner class, which is used to implement base TensorOperations
class TensorShape {
public:
    explicit TensorShape(const std::vector<SizeT>& dims) : m_sizes(dims) {
    }

    explicit TensorShape(const SizeT rank) : m_sizes(rank) {
    }

    std::vector<SizeT>& sizesRef() {
        return m_sizes;
    }

    [[nodiscard]] SizeT getTotalSize() const;

    [[nodiscard]] const std::vector<SizeT>& sizesRef() const {
        return m_sizes;
    }

    [[nodiscard]] SizeT rank() const {
        return static_cast<SizeT>(m_sizes.size());
    }

    const SizeT& operator[](const SizeT index) const {
        if (index < 0) {
            return m_sizes[m_sizes.size() + index];
        }
        return m_sizes[index];
    }

    SizeT& operator[](const SizeT index) {
        if (index < 0) {
            return m_sizes[m_sizes.size() + index];
        }
        return m_sizes[index];
    }

    [[nodiscard]] bool empty() const {
        return m_sizes.empty();
    }

    [[nodiscard]] std::vector<SizeT> sizes() const {
        return m_sizes;
    }

    bool operator==(const TensorShape& other) const {
        return m_sizes == other.m_sizes;
    }

    bool operator!=(const TensorShape& other) const {
        return m_sizes != other.m_sizes;
    }

    [[nodiscard]] bool canBroadcastWith(const TensorShape& other) const {
        if (other.rank() < rank()) {
            return false;
        }

        const SizeT rankOther = other.rank();
        for (auto i = 0; i < rank(); ++i) {
            if (!(other[rankOther - 1 - i] == 1
                || m_sizes[rank() - 1 - i] == other.m_sizes[rank() - 1 - i]
                || m_sizes[rank() - 1 - i] == 1)) {
                return false;
            }
        }

        return true;
    }

    // return an expected Shape or an Err if shapes cannot broadcast
    [[nodiscard]] std::expected<TensorShape, Error>
    broadcastWith(const TensorShape& other) const;

    [[nodiscard]] std::string toString() const;

private:
    std::vector<SizeT> m_sizes;
};
} // namespace h3::core::math

#endif // HAHAHA_TENSORSHAPE_H_D00755481C5B426DA11E0C244ED22F33