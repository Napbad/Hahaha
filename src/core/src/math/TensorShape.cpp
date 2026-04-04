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

#include "math/TensorShape.h"
std::expected<h3::core::math::TensorShape, h3::core::Error>
h3::core::math::TensorShape::broadcastWith(const TensorShape& other) const {
    if (!canBroadcastWith(other)) {
        return std::unexpected(Error("Cannot broadcast shapes " + toString()
                                         + " and " + other.toString(),
                                     ErrorCode::InvalidArgument));
    }
    const SizeT otherRank = other.rank();
    const SizeT resRank = std::max(otherRank, rank());

    TensorShape resShape(resRank);

    for (SizeT i = 0; i < resRank; ++i) {
        const SizeT otherDim = i < otherRank ? other.sizes()[static_cast<std::size_t>(i)] : 1;
        const SizeT thisDim = i < rank() ? sizes()[static_cast<std::size_t>(i)] : 1;
        resShape.sizesRef()[static_cast<std::size_t>(i)] = std::max(otherDim, thisDim);
    }

    return resShape;
}
std::string h3::core::math::TensorShape::toString() const {
    std::string out;
    out.push_back('[');
    for (SizeT i = 0; i < static_cast<SizeT>(m_sizes.size()); ++i) {
        if (i != 0) {
            out.append(", ");
        }
        out.append(std::to_string(sizes()[static_cast<std::size_t>(i)]));
    }
    out.push_back(']');
    return out;
}
h3::core::SizeT h3::core::math::TensorShape::getTotalSize() const {

    SizeT res = 1;
    for (SizeT i = 0; i < rank(); ++i) {
        res *= sizes()[static_cast<std::size_t>(i)];
    }

    return res;
}