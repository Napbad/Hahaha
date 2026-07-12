//  Copyright (c) 2025-2026 Contributors of
//  Hahaha(https://github.com/jason-is-debugging/Hahaha)
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

#include "math/TensorInner.h"

#include "utils/handler/exception_handler.h"

namespace h3::core::math {

Scalar TensorInner::operator()(const Index& index) {
    ensureStorageExists();
    if (index.size() != m_stride.size()) {
        if (index.size() > m_stride.size()) {
            ThrowInvalid("The size of indexes is bigger than stride size, indexes "
                         "size: {}, strides size: {}",
                         index.size(),
                         m_stride.size());
        }
        if (index.size() < m_stride.size()) {
            ThrowInvalid("The size of indexes is smaller than stride, with indexes "
                         "size: {}, strides size: {}, if you want to index a Tensor "
                         "result, use [] instead",
                         index.size(),
                         m_stride.size());
        }
    }

    SizeT offset = m_offset;
    for (auto i = index.size() - 1; i >= 0; --i) {
        offset += index[i] * m_stride[i];
    }

    backend::CommonPointer targetPtr = m_storage.data() + offset;

    return {m_metadata.dataType, targetPtr, true};
}

TensorInner TensorInner::operator[](SizeT index) {
    ensureStorageExists();
    if (m_shape.empty()) {
        ThrowInvalid("Can't index a scalar");
    }

    if (index >= m_shape[0]) {
        ThrowInvalid(
            "Wrong index, the input index is {}, but the shape on this dim is {}",
            index,
            m_shape[0]);
    }

    auto sizes = m_shape.sizes();
    sizes.erase(sizes.begin());
    auto stride = m_stride.strides();
    stride.erase(stride.begin());
    const SizeT offset = sizeOf(dataType()) * index * m_stride[0];
    // share data
    auto res = TensorInner(TensorShape(sizes),
                           TensorStride(stride),
                           m_storage.createView(),
                           m_offset + offset,
                           m_metadata);
    res.m_metadata.isView = true;
    return res;
}

TensorInner
TensorInner::slice(int64_t dim, int64_t start, int64_t end, int64_t step) const {
    // Stub - returns copy of self
    return *this;
}
void TensorInner::setScalarValue(const Scalar& scalar) {
    ensureStorageExists();

    Scalar mutableScalar = scalar;
    mutableScalar.convertToType(m_metadata.dataType);
    if (this->m_shape.rank() == 0 || this->m_shape.getTotalSize() == 1) {
        this->m_storage.copyFrom(m_offset, mutableScalar.data());
        return;
    }

    ThrowInvalid("Can't set scalar value to a non-scalar tensor");
}
void TensorInner::ensureStorageExists() {
    if (m_storage.data() != nullptr) {
        return;
    }
    m_storage.init(m_metadata.dataType, m_shape.getTotalSize());
}

bool TensorInner::isContiguous() const noexcept {
    return m_metadata.isContiguous;
}

TensorShape TensorInner::shape() {
    return m_shape;
}

bool TensorInner::computeAndStoreIsContiguous() {
    SizeT expectedStride = 1;
    if (m_shape.rank() == 0) {
        m_metadata.isContiguous = true;
        return true;
    }
    for (SizeT i = 1; i <= m_stride.size(); ++i) {
        if (m_stride[-i] != expectedStride) {
            m_metadata.isContiguous = false;
            return false;
        }
        expectedStride *= m_shape[-i];
    }
    return m_metadata.isContiguous;
}

Scalar TensorInner::item() const {
    if (m_shape.getTotalSize() != 1) {
        auto sizes = m_shape.sizes();
        ThrowInvalid("Can't call item() on a Tensor with shape: {}",
                     m_shape.toString());
    }

    backend::CommonPointer targetPtr = m_storage.data() + m_offset;
    return {m_metadata.dataType, targetPtr, true};
}
SizeT TensorInner::getTotalSize() const {
    return m_shape.getTotalSize();
}

TensorInner TensorInner::add(const TensorInner& other) const {
    // Stub - returns copy of self
    // Use ComputeNode::add for actual addition
    return *this;
}

} // namespace h3::core::math