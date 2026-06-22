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

#include "compute/ComputeNode.h"

#include <memory>
#include <vector>

#include "compute/ComputeDispatcher.h"
#include "math/TensorInner.h"
#include "math/TensorStride.h"
#include "utils/handler/exception_handler.h"

namespace h3::core::compute {

DataType detectResDataType(const DataType lhs, const DataType rhs) {
    return std::max(lhs, rhs);
}

void checkCanRunBinOper(const utils::OwnPointer<math::TensorInner>& t1,
                        const utils::OwnPointer<math::TensorInner>& t2) {

    if (t1->shapeRef() != t2->shapeRef()
        && !t1->shapeRef().canBroadcastWith(t2->shapeRef())) {
        ThrowInvalid("Tensor have different shapes, and they can not broadcast");
    }

    if (t1->device() != t2->device()) {
        return ThrowInvalid("Tensor have different devices");
    }
}

ComputeNode ComputeNode::add(const ComputeNode& other) const {
    checkCanRunBinOper(tensorInner(), other.tensorInner());
    const DataType resType = detectResDataType(tensorInner()->dataType(),
                                               other.tensorInner()->dataType());
    auto resTensor = utils::make_own_ptr<math::TensorInner>(tensorInner()->shape(),
                                       math::TensorMetadata{.dataType = resType});

    std::vector<utils::OwnPointer<math::TensorInner>> operands{tensorInner(), other.tensorInner(), resTensor};
    ComputeDispatcher::dispatch(Operator::Add, operands);

    return ComputeNode(std::move(resTensor));
}

ComputeNode ComputeNode::operator+(const ComputeNode& other) const {
    return add(other);
}

ComputeNode ComputeNode::view() const {
    return ComputeNode(tensorInner());
}

ComputeNode ComputeNode::broadcastView(const math::TensorShape& newShape) const {
    const auto self = tensorInner();
    const auto& selfShape = self->shapeRef();
    const auto& selfStride = self->strideRef();

    if (!selfShape.canBroadcastWith(newShape) && !newShape.canBroadcastWith(
        selfShape)) {
        throw std::invalid_argument(
            "Tensor with shape: " + selfShape.toString() +
            " cannot broadcast to shape: " + newShape.toString());
    }

    const SizeT newRank = newShape.rank();
    const SizeT selfRank = selfShape.rank();
    std::vector<SizeT> newStridesVec(static_cast<std::size_t>(newRank));

    for (SizeT k = 0; k < newRank; ++k) {
        const SizeT newIdx = newRank - 1 - k;
        const SizeT newDim = newShape[newIdx];
        SizeT selfDim = 1;
        SizeT selfStrideVal = 0;
        if (k < selfRank) {
            const SizeT selfIdx = selfRank - 1 - k;
            selfDim = selfShape[selfIdx];
            selfStrideVal = selfStride[selfIdx];
        }
        if (selfDim == newDim) {
            newStridesVec[static_cast<std::size_t>(newIdx)] = selfStrideVal;
        } else if (selfDim == 1) {
            newStridesVec[static_cast<std::size_t>(newIdx)] = 0;
        } else {
            throw std::invalid_argument(
                "Internal broadcast stride mismatch for shape " + selfShape.
                toString() +
                " -> " + newShape.toString());
        }
    }

    math::TensorMetadata meta = self->metadataRef();
    meta.isView = true;
    const auto view = utils::make_own_ptr<math::TensorInner>(
        newShape,
        math::TensorStride(newStridesVec),
        self->storageRef(),
        self->offset(),
        meta);
    view->computeAndStoreIsContiguous();

    return ComputeNode(view);
}

void ComputeNode::setTensorInner(math::TensorInner&& tensor_inner) {
    this->m_tensor = utils::make_own_ptr<math::TensorInner>(tensor_inner);
}
} // namespace h3::core::compute