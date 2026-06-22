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

#ifndef HAHAHA_BINOPERATOREXECUTOR_H
#define HAHAHA_BINOPERATOREXECUTOR_H

#include <expected>
#include <vector>

#include "compute/operator_executor/OperatorExecutor.h"
#include "utils/OwnPointer.h"

namespace h3::core::compute {

class BinOperatorExecutor : public OperatorExecutor {
  public:
    explicit BinOperatorExecutor(const Operator op) : OperatorExecutor(op) {}
};

#define HAHAHA_DECLARE_BIN_EXECUTOR(NS, Name)                                            \
    namespace NS {                                                                       \
    class Name##OperatorExecutor final : public BinOperatorExecutor {                    \
      public:                                                                            \
        Name##OperatorExecutor() : BinOperatorExecutor(Operator::Name) {}                \
        std::expected<void, Error> execute(                                              \
            ComputeContext& context,                                                     \
            std::vector<utils::OwnPointer<math::TensorInner>>& operands) override;       \
    };                                                                                   \
    }

HAHAHA_DECLARE_BIN_EXECUTOR(cpu, Add)
HAHAHA_DECLARE_BIN_EXECUTOR(cpu, Sub)
HAHAHA_DECLARE_BIN_EXECUTOR(cpu, Mul)
HAHAHA_DECLARE_BIN_EXECUTOR(cpu, Div)
HAHAHA_DECLARE_BIN_EXECUTOR(cpu, Mod)
HAHAHA_DECLARE_BIN_EXECUTOR(cpu, Pow)
HAHAHA_DECLARE_BIN_EXECUTOR(cpu, Max)
HAHAHA_DECLARE_BIN_EXECUTOR(cpu, Min)

#ifdef HAHAHA_USE_CUDA
HAHAHA_DECLARE_BIN_EXECUTOR(cuda, Add)
HAHAHA_DECLARE_BIN_EXECUTOR(cuda, Sub)
HAHAHA_DECLARE_BIN_EXECUTOR(cuda, Mul)
HAHAHA_DECLARE_BIN_EXECUTOR(cuda, Div)
HAHAHA_DECLARE_BIN_EXECUTOR(cuda, Mod)
HAHAHA_DECLARE_BIN_EXECUTOR(cuda, Pow)
HAHAHA_DECLARE_BIN_EXECUTOR(cuda, Max)
HAHAHA_DECLARE_BIN_EXECUTOR(cuda, Min)
#endif

#undef HAHAHA_DECLARE_BIN_EXECUTOR

} // namespace h3::core::compute

#endif // HAHAHA_BINOPERATOREXECUTOR_H
