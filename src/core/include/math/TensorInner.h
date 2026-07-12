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
// Created by napbad on 3/26/26.
//

#ifndef HAHAHA_TENSORINNER_H_D0EE919AED8B4D9ABD7CA18281E1928E
#define HAHAHA_TENSORINNER_H_D0EE919AED8B4D9ABD7CA18281E1928E
#include <utility>
#include <vector>

#include "Index.h"
#include "Scalar.h"
#include "TensorMetadata.h"
#include "TensorShape.h"
#include "TensorStride.h"
#include "backend/Device.h"
#include "backend/Storage.h"
#include "utils/OwnPointer.h"

namespace h3::core::math {
// Inner class, which is used to implement base TensorOperations
class TensorInner {
public:
    explicit TensorInner(const TensorShape& shape)
        : m_shape(shape), m_stride(shape), m_offset(0) {
    }

    TensorInner(TensorShape shape,
                TensorStride stride,
                const backend::Storage& storage,
                const SizeT offset,
                const TensorMetadata& metadata)
        : m_shape(std::move(shape)), m_stride(std::move(stride)), m_offset(offset),
          m_storage(storage), m_metadata(metadata) {
    }

    TensorInner(const TensorShape& shape, const TensorMetadata& metadata)
        : m_shape(shape), m_stride(shape), m_offset(0), m_metadata(metadata) {
        const SizeT singleElementSize = sizeOf(metadata.dataType);
        m_storage = backend::Storage(shape.getTotalSize() * singleElementSize,
                                     metadata.device);
        m_metadata = metadata;
    }

    // shadow copy, use clone() for deep copy instead.
    TensorInner& operator=(const TensorInner& other) {
        this->m_stride = other.m_stride;
        this->m_shape = other.m_shape;
        this->m_storage = other.m_storage.createView();
        this->m_metadata = other.m_metadata;
        this->m_offset = other.m_offset;

        return *this;
    }

    [[nodiscard]] TensorInner clone() const {
        auto storageClone = m_storage.clone();
        if (!storageClone.has_value()) {
            throw std::invalid_argument(
                "Can not clone storage, Error is " + storageClone.error().message());
        }
        return {m_shape, m_stride, storageClone.value(), m_offset, m_metadata};
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

    [[nodiscard]] backend::Device device() const {
        return m_metadata.device;
    }

    /// Byte offset of this view into \ref storageRef() (always 0 for root tensors).
    [[nodiscard]] SizeT offset() const noexcept {
        return m_offset;
    }

    Scalar operator()(const Index& index);

    TensorInner operator[](SizeT index);

    [[nodiscard]] Scalar item() const;

    [[nodiscard]] SizeT getTotalSize() const;

    // Elementwise (out-of-place)
    [[nodiscard]] TensorInner add(const TensorInner& other) const;

    TensorInner sub(const TensorInner& other) const;

    TensorInner mul(const TensorInner& other) const;

    TensorInner div(const TensorInner& other) const;

    TensorInner add(const Scalar& scalar) const;

    TensorInner sub(const Scalar& scalar) const;

    TensorInner mul(const Scalar& scalar) const;

    TensorInner div(const Scalar& scalar) const;

    // In-place (more memory efficient)
    TensorInner& add_(const TensorInner& other);

    TensorInner& sub_(const TensorInner& other);

    TensorInner& mul_(const TensorInner& other);

    TensorInner& div_(const TensorInner& other);

    TensorInner& add_(const Scalar& scalar);

    TensorInner& sub_(const Scalar& scalar);

    TensorInner& mul_(const Scalar& scalar);

    TensorInner& div_(const Scalar& scalar);

    TensorInner operator +(const TensorInner& other) const;

    TensorInner operator -(const TensorInner& other) const;

    TensorInner operator *(const TensorInner& other) const;

    TensorInner operator /(const TensorInner& other) const;

    [[nodiscard]] bool isContiguous() const noexcept;

    TensorShape shape();

    bool computeAndStoreIsContiguous();

    [[nodiscard]] TensorInner matmul(const TensorInner& other) const;

    [[nodiscard]] TensorInner transpose() const;

    [[nodiscard]] TensorInner reshape(const TensorShape& shape) const;

    [[nodiscard]] TensorInner squeeze() const;

    [[nodiscard]] TensorInner squeeze(SizeT index) const;

    [[nodiscard]] TensorInner unsqueeze() const;


    [[nodiscard]] TensorInner
    slice(int64_t dim, int64_t start, int64_t end, int64_t step = 1) const;

    [[nodiscard]] TensorInner narrow(int64_t dim,
                                     int64_t start,
                                     int64_t length) const;

    [[nodiscard]] TensorInner select(int64_t dim, int64_t index) const;


    [[nodiscard]] TensorInner sum(int64_t dim, bool keepdim = false) const;

    [[nodiscard]] TensorInner sum() const;

    [[nodiscard]] TensorInner mean(int64_t dim, bool keepdim = false) const;

    [[nodiscard]] TensorInner max() const;

    [[nodiscard]] TensorInner min() const;

    [[nodiscard]] Scalar mean() const;

    [[nodiscard]] TensorInner to(const backend::Device& device) const;

    [[nodiscard]] TensorInner to(DataType dtype) const;

    [[nodiscard]] TensorInner to(const backend::Device& device,
                                 DataType dtype) const;

    [[nodiscard]] TensorInner flatten() const;

    [[nodiscard]] TensorInner view(const TensorShape& shape) const;

    [[nodiscard]] TensorInner onesWithSameShape() const;

    [[nodiscard]] TensorInner zerosWithSameShape() const;

    [[nodiscard]] TensorInner randWithSameShape() const;

    [[nodiscard]] TensorInner broadcastTo(TensorShape shape) const;

    void setScalarValue(const Scalar& scalar);

    // Static factory method for creating tensors with initial data
    template<typename T>
    [[nodiscard]] static utils::OwnPointer<TensorInner> create(
        DataType dtype,
        backend::DeviceType deviceType,
        const std::vector<Int64>& shape,
        const std::vector<T>& data = {});

    template<typename T>
    [[nodiscard]] static utils::OwnPointer<TensorInner> create(
        DataType dtype,
        const backend::Device& device,
        const std::vector<Int64>& shape,
        const std::vector<T>& data = {});

  private:
    TensorShape m_shape;
    TensorStride m_stride;
    SizeT m_offset; ///< Byte offset of this view into \ref storageRef() (always 0 for root tensors).
    backend::Storage m_storage;
    TensorMetadata m_metadata;

    // this method is used to ensure that the storage is not empty, because while
    // init a TensorInner with only shape, the storage might not be a valid storage
    void ensureStorageExists();
};

} // namespace h3::core::math

// Template implementations
namespace h3::core::math {

template<typename T>
utils::OwnPointer<TensorInner> TensorInner::create(
    DataType dtype,
    backend::DeviceType deviceType,
    const std::vector<Int64>& shape,
    const std::vector<T>& data) {
    return create(dtype, backend::Device(0, deviceType), shape, data);
}

template<typename T>
utils::OwnPointer<TensorInner> TensorInner::create(
    DataType dtype,
    const backend::Device& device,
    const std::vector<Int64>& shape,
    const std::vector<T>& data) {
    TensorShape tensorShape(shape);
    TensorMetadata meta;
    meta.dataType = dtype;
    meta.device = device;
    meta.isContiguous = true;
    meta.isView = false;
    
    auto tensor = utils::make_own_ptr<TensorInner>(tensorShape, meta);
    
    if (!data.empty()) {
        T* tensorData = tensor->storageRef().data().as<T>();
        for (size_t i = 0; i < data.size() && i < tensorShape.getTotalSize(); ++i) {
            tensorData[i] = data[i];
        }
    }
    
    return tensor;
}

} // namespace h3::core::math

#endif // HAHAHA_TENSORINNER_H_D0EE919AED8B4D9ABD7CA18281E1928E