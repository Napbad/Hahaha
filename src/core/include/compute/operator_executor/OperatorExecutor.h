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
// Created by napbad on 5/11/26.
//

#ifndef HAHAHA_OPERATOREXECUTOR_H_C959E590EA4A42BC9E046DF267293849
#define HAHAHA_OPERATOREXECUTOR_H_C959E590EA4A42BC9E046DF267293849
#include <expected>
#include <vector>

#include "Error.h"
#include "compute/ComputeContext.h"
#include "compute/ComputeNode.h"
#include "defines.h"


namespace h3::core::compute {


class OperatorExecutor {
public:
    explicit OperatorExecutor(const Operator op)
        : m_op(op) {
    }

    virtual ~OperatorExecutor() = default;

    [[nodiscard]] Operator op() const {
        return m_op;
    }

    virtual std::expected<void, Error> execute(
        ComputeContext& context,
        std::vector<utils::OwnPointer<math::TensorInner>>& operands) = 0;

private:
    Operator m_op;
};

}


#endif //HAHAHA_OPERATOREXECUTOR_H_C959E590EA4A42BC9E046DF267293849