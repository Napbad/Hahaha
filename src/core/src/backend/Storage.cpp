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

#include "backend/Storage.h"

#include "utils/handler/exception_handler.h"

namespace h3::core::backend {

Storage Storage::createView() const {

    auto res = Storage(
        m_data,
        m_memoryManager,
        m_size,
        true
        );

    return res;
}

std::expected<Storage, Error> Storage::clone() const {
    auto resPtr = m_memoryManager->allocate(m_size);
    if (!resPtr.has_value()) {
        return std::unexpected(resPtr.error());
    }
    return {Storage(resPtr.value(), m_memoryManager, m_size, false)};
}

std::expected<CommonPointer, Error> Storage::resize(const SizeT size) {
    if (size == this->size()) {
        return {data()};
    }

    if (size < this->size()) {
        return std::unexpected(Error(
            "target size is smaller than current size, cannot resize",
            ErrorCode::InvalidArgument));
    }

    std::expected<CommonPointer, Error> expected = memoryManager()->allocate(size);
    if (!expected.has_value()) {
        return std::unexpected(expected.error());
    }

    const auto sourcePtr = data();

    if (std::expected<void, Error> res = memoryManager()->move(
            expected.value(),
            sourcePtr,
            size);
        !res.has_value()) {
        return std::unexpected(res.error());
    }
    memoryManager()->deallocate(sourcePtr);

    m_data = expected.value();

    return expected;
}
void Storage::copyFrom(const SizeT beginPos, const CommonPointer& ptr) const {
    if (ptr.size() > this->m_size) {
        throw std::invalid_argument("target size is bigger than current size,"
                                    " the copy of data will cause unknown behavior");
    }


    if (this->m_data.device() != ptr.device()) {
        throw std::invalid_argument("target device is different from current device");
    }

    if (auto res = this->m_memoryManager->copy(this->m_data + beginPos, ptr, ptr.size());
        !res.has_value()) {
        throw std::invalid_argument(std::string("copy failed, the error is: ") + res.error().message());
        }

}
void Storage::init(const DataType data, const SizeT size) {
    const SizeT singleSize = sizeOf(data);
    auto res = this->m_memoryManager->allocate(singleSize * size);
    if (!res.has_value()) {
        ThrowInvalid("init failed, the error is: %s", res.error().message());
    }
    this->m_data = res.value();
    this->m_size = singleSize * size;
}
} // namespace h3::core::backend