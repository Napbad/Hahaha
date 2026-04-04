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

#include "utils/handler/exception_handler.h"

namespace h3::core::compute {

DataType detectResDataType(const DataType lhs, const DataType rhs) {
    return std::max(lhs, rhs);
}
void checkCanRunBinOper(const std::shared_ptr<math::TensorInner>& t1,
                        const std::shared_ptr<math::TensorInner>& t2) {

    if (t1->shapeRef() != t2->shapeRef()
        && !t1->shapeRef().canBroadcastWith(t2->shapeRef())) {
        ThrowInvalid("Tensor have different shapes, and they can not broadcast");
    }
}
ComputeNode ComputeNode::add(const ComputeNode& other) const {
    checkCanRunBinOper(tensorInner(), other.tensorInner());
    DataType resType = detectResDataType(tensorInner()->dataType(),
                                         other.tensorInner()->dataType());
    math::TensorInner(tensorInner()->shape());

    return ComputeNode(tensorInner());
}
} // namespace h3::core::compute