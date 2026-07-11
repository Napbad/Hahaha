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

#ifndef HAHAHA_MULTIOPERATOREXECUTOR_H
#define HAHAHA_MULTIOPERATOREXECUTOR_H

#include <expected>
#include <vector>

#include "compute/operator_executor/OperatorExecutor.h"
#include "utils/OwnPointer.h"

namespace h3::core::compute {

class MultiOperatorExecutor : public OperatorExecutor {
  public:
    explicit MultiOperatorExecutor(const Operator op) : OperatorExecutor(op) {}
};

#define HAHAHA_DECLARE_MULTI_EXECUTOR(NS, Name)                                          \
    namespace NS {                                                                       \
    class Name##OperatorExecutor final : public MultiOperatorExecutor {                \
      public:                                                                            \
        Name##OperatorExecutor() : MultiOperatorExecutor(Operator::Name) {}              \
        std::expected<void, Error> execute(                                              \
            ComputeContext& context,                                                     \
            std::vector<utils::OwnPointer<math::TensorInner>>& operands) override;       \
    };                                                                                   \
    }

HAHAHA_DECLARE_MULTI_EXECUTOR(cpu, Clamp)

#ifdef HAHAHA_USE_CUDA
HAHAHA_DECLARE_MULTI_EXECUTOR(cuda, Clamp)
#endif

#undef HAHAHA_DECLARE_MULTI_EXECUTOR

} // namespace h3::core::compute

#endif // HAHAHA_MULTIOPERATOREXECUTOR_H
