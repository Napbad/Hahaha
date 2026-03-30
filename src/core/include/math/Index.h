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

#ifndef HAHAHA_INDICE_H_699C069F8A6B4822AE5EFB3C6785F732
#define HAHAHA_INDICE_H_699C069F8A6B4822AE5EFB3C6785F732
#include <vector>

#include "defines.h"

namespace h3::core::math {

class Index {
public:
    Index& operator,(const SizeT index) {
        m_indices.push_back(index);
        return *this;
    }

    // NOLINTNEXTLINE
    Index(const SizeT index) {
        m_indices.push_back(index);
    }

    [[nodiscard]] std::vector<SizeT>& indicesRef() {
        return m_indices;
    }

    [[nodiscard]] const std::vector<SizeT>& indicesRef() const {
        return m_indices;
    }

    SizeT operator[](const SizeT index) const {
        return m_indices[index];
    }

    [[nodiscard]] SizeT size() const {
        return static_cast<SizeT>(m_indices.size());
    }

private:
    std::vector<SizeT> m_indices;
};

}

#endif // HAHAHA_INDICE_H_699C069F8A6B4822AE5EFB3C6785F732