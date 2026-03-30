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

#ifndef HAHAHA_TENSORSTRIDE_H_31190F1E4C55477DBC859E84DDCDDDF2
#define HAHAHA_TENSORSTRIDE_H_31190F1E4C55477DBC859E84DDCDDDF2
#include "TensorShape.h"

namespace h3::core::math {
// Inner class, which is used to implement base TensorOperations
class TensorStride {
public:
    explicit TensorStride(const TensorShape& shape) {
        SizeT stride = 1;
        for (auto i = 0; i < shape.rank(); i++) {
            m_strides.push_back(stride);
            stride *= shape.sizesRef()[i];
        }
    }

    explicit TensorStride(const std::vector<SizeT>& strides) : m_strides(strides) {
    }

    [[nodiscard]] SizeT size() const {
        return static_cast<SizeT>(m_strides.size());
    }

    SizeT operator[](const SizeT index) const {
        return m_strides[index];
    }

    [[nodiscard]] std::vector<SizeT> strides() const {
        return m_strides;
    }

    std::vector<SizeT>& stridesRef() {
        return m_strides;
    }

    [[nodiscard]] const std::vector<SizeT>& stridesRef() const {
        return m_strides;
    }

private:
    std::vector<SizeT> m_strides;

};
}

#endif // HAHAHA_TENSORSTRIDE_H_31190F1E4C55477DBC859E84DDCDDDF2