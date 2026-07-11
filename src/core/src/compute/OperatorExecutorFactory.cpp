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

#include "compute/operator_executor/OperatorExecutorFactory.h"

#include <memory>
#include <string>

#include "compute/operator_executor/CachedExecutor.h"

namespace h3::core::compute {

namespace {
constexpr SizeT deviceSlotCount = 2;

std::expected<SizeT, Error> deviceSlot(const backend::DeviceType deviceType) {
    switch (deviceType) {
    case backend::DeviceType::CPU:
        return 0;
    case backend::DeviceType::CUDA:
        return 1;
    default:
        return std::unexpected(Error(
            "unknown device type",
            ErrorCode::DeviceNotSupportedError));
    }
}
} // namespace

OperatorExecutorFactory::OperatorExecutorFactory()
    : m_cachedExecutors(static_cast<SizeT>(Operator::Count) * deviceSlotCount) {
}

SizeT OperatorExecutorFactory::cacheIndex(const Operator op,
                                          const backend::DeviceType deviceType) {
    const SizeT slot = deviceType == backend::DeviceType::CUDA ? 1 : 0;
    return static_cast<SizeT>(op) * deviceSlotCount + slot;
}

std::unique_ptr<OperatorExecutor> OperatorExecutorFactory::create(
    const Operator op,
    const backend::DeviceType deviceType) {
    utils::OwnPointer<OperatorExecutor> owned;
    switch (deviceType) {
    case backend::DeviceType::CPU:
        owned = makeCPUOperatorExecutor(op);
        break;
    case backend::DeviceType::CUDA:
        owned = makeCUDAOperatorExecutor(op);
        break;
    default:
        return nullptr;
    }
    if (!owned) {
        return nullptr;
    }
    return std::unique_ptr<OperatorExecutor>(owned.release());
}

std::expected<OperatorExecutor*, Error> OperatorExecutorFactory::get(
    const Operator op,
    const backend::DeviceType deviceType) {
    if (const auto slot = deviceSlot(deviceType); !slot) {
        return std::unexpected(slot.error());
    }

    const auto index = cacheIndex(op, deviceType);
    if (index < 0 || index >= static_cast<SizeT>(m_cachedExecutors.size())) {
        return std::unexpected(Error(
            "operator is out of executor factory cache range",
            ErrorCode::InvalidArgument));
    }

    if (!m_cachedExecutors[index]) {
        m_cachedExecutors[index] = create(op, deviceType);
    }

    if (!m_cachedExecutors[index]) {
        return std::unexpected(Error(
            std::string("can not create executor for operator ") + toString(op),
            ErrorCode::DeviceNotSupportedError));
    }

    return m_cachedExecutors[index].get();
}

} // namespace h3::core::compute
