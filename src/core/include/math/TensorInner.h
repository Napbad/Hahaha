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

#ifndef HAHAHA_TENSORINNER_H_D0EE919AED8B4D9ABD7CA18281E1928E
#define HAHAHA_TENSORINNER_H_D0EE919AED8B4D9ABD7CA18281E1928E
#include <utility>

#include "Index.h"
#include "Scalar.h"
#include "TensorMetadata.h"
#include "TensorShape.h"
#include "TensorStride.h"
#include "backend/Storage.h"

namespace h3::core::math {
// Inner class, which is used to implement base TensorOperations
class TensorInner {
public:
    explicit TensorInner(const TensorShape& shape) : m_shape(shape),
                                                     m_stride(shape) {

    }

    TensorInner(TensorShape shape,
                TensorStride stride,
                const backend::Storage& storage,
                const TensorMetadata& metadata) : m_shape(std::move(shape)),
                                                  m_stride(std::move(stride)),
                                                  m_storage(storage),
                                                  m_metadata(metadata) {

    }

    // shadow copy, use clone() for deep copy instead.
    TensorInner& operator=(const TensorInner& other) {
        this->m_stride = other.m_stride;
        this->m_shape = other.m_shape;
        this->m_storage = other.m_storage.createView();
        this->m_metadata = other.m_metadata;

        return *this;
    }

    [[nodiscard]] TensorInner clone() const {
        return {
            m_shape,
            m_stride,
            m_storage.clone().value(),
            m_metadata
        };
    }

    [[nodiscard]] const TensorShape& shapeRef() const {
        return m_shape;
    }

    [[nodiscard]] TensorShape& shapeRef() {
        return m_shape;
    }

    [[nodiscard]] TensorStride& strideRef() {
        return m_stride;
    }

    [[nodiscard]] const TensorStride& strideRef() const {
        return m_stride;
    }

    [[nodiscard]] const backend::Storage& storageRef() const {
        return m_storage;
    }

    [[nodiscard]] backend::Storage& storageRef() {
        return m_storage;
    }

    [[nodiscard]] const TensorMetadata& metadataRef() const {
        return m_metadata;
    }

    [[nodiscard]] TensorMetadata& metadataRef() {
        return m_metadata;
    }

    [[nodiscard]] DataType dataType() const {
        return m_metadata.dataType;
    }

    Scalar operator()(const Index& index) const;
    TensorInner operator[](SizeT index) const;

    TensorInner operator +(const TensorInner& other) const;
    TensorInner operator -(const TensorInner& other) const;
    TensorInner operator *(const TensorInner& other) const;
    TensorInner operator /(const TensorInner& other) const;

    TensorInner operator +(const Scalar& other) const;
    TensorInner operator -(const Scalar& other) const;
    TensorInner operator *(const Scalar& other) const;
    TensorInner operator /(const Scalar& other) const;

    [[nodiscard]] TensorInner matmul(const TensorInner& other) const;
private:
    TensorShape m_shape;
    TensorStride m_stride;
    backend::Storage m_storage;
    TensorMetadata m_metadata;
};

}

#endif //HAHAHA_TENSORINNER_H_D0EE919AED8B4D9ABD7CA18281E1928E