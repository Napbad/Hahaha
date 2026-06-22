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

#ifndef HAHAHA_UNARYOPERATOREXECUTOR_H
#define HAHAHA_UNARYOPERATOREXECUTOR_H

#include <expected>
#include <vector>

#include "compute/operator_executor/OperatorExecutor.h"
#include "utils/OwnPointer.h"

namespace h3::core::compute {

class UnaryOperatorExecutor : public OperatorExecutor {
  public:
    explicit UnaryOperatorExecutor(const Operator op) : OperatorExecutor(op) {}
};

#define HAHAHA_DECLARE_UNARY_EXECUTOR(NS, Name)                                          \
    namespace NS {                                                                       \
    class Name##OperatorExecutor final : public UnaryOperatorExecutor {                  \
      public:                                                                            \
        Name##OperatorExecutor() : UnaryOperatorExecutor(Operator::Name) {}              \
        std::expected<void, Error> execute(                                              \
            ComputeContext& context,                                                     \
            std::vector<utils::OwnPointer<math::TensorInner>>& operands) override;       \
    };                                                                                   \
    }

HAHAHA_DECLARE_UNARY_EXECUTOR(cpu, Sqrt)
HAHAHA_DECLARE_UNARY_EXECUTOR(cpu, Log)
HAHAHA_DECLARE_UNARY_EXECUTOR(cpu, Exp)
HAHAHA_DECLARE_UNARY_EXECUTOR(cpu, Sin)
HAHAHA_DECLARE_UNARY_EXECUTOR(cpu, Cos)
HAHAHA_DECLARE_UNARY_EXECUTOR(cpu, Tan)
HAHAHA_DECLARE_UNARY_EXECUTOR(cpu, Asin)
HAHAHA_DECLARE_UNARY_EXECUTOR(cpu, Acos)
HAHAHA_DECLARE_UNARY_EXECUTOR(cpu, Atan)
HAHAHA_DECLARE_UNARY_EXECUTOR(cpu, Abs)
HAHAHA_DECLARE_UNARY_EXECUTOR(cpu, Sign)
HAHAHA_DECLARE_UNARY_EXECUTOR(cpu, Ceil)
HAHAHA_DECLARE_UNARY_EXECUTOR(cpu, Floor)
HAHAHA_DECLARE_UNARY_EXECUTOR(cpu, Round)
HAHAHA_DECLARE_UNARY_EXECUTOR(cpu, Trunc)
HAHAHA_DECLARE_UNARY_EXECUTOR(cpu, Sinh)
HAHAHA_DECLARE_UNARY_EXECUTOR(cpu, Cosh)
HAHAHA_DECLARE_UNARY_EXECUTOR(cpu, Tanh)
HAHAHA_DECLARE_UNARY_EXECUTOR(cpu, Asinh)
HAHAHA_DECLARE_UNARY_EXECUTOR(cpu, Acosh)
HAHAHA_DECLARE_UNARY_EXECUTOR(cpu, Atanh)
HAHAHA_DECLARE_UNARY_EXECUTOR(cpu, Log10)
HAHAHA_DECLARE_UNARY_EXECUTOR(cpu, Log2)
HAHAHA_DECLARE_UNARY_EXECUTOR(cpu, Log1p)
HAHAHA_DECLARE_UNARY_EXECUTOR(cpu, Exp2)
HAHAHA_DECLARE_UNARY_EXECUTOR(cpu, Expm1)
HAHAHA_DECLARE_UNARY_EXECUTOR(cpu, Cbrt)
HAHAHA_DECLARE_UNARY_EXECUTOR(cpu, Erf)
HAHAHA_DECLARE_UNARY_EXECUTOR(cpu, Erfc)
HAHAHA_DECLARE_UNARY_EXECUTOR(cpu, Tgamma)
HAHAHA_DECLARE_UNARY_EXECUTOR(cpu, Lgamma)

#ifdef HAHAHA_USE_CUDA
HAHAHA_DECLARE_UNARY_EXECUTOR(cuda, Sqrt)
HAHAHA_DECLARE_UNARY_EXECUTOR(cuda, Log)
HAHAHA_DECLARE_UNARY_EXECUTOR(cuda, Exp)
HAHAHA_DECLARE_UNARY_EXECUTOR(cuda, Sin)
HAHAHA_DECLARE_UNARY_EXECUTOR(cuda, Cos)
HAHAHA_DECLARE_UNARY_EXECUTOR(cuda, Tan)
HAHAHA_DECLARE_UNARY_EXECUTOR(cuda, Asin)
HAHAHA_DECLARE_UNARY_EXECUTOR(cuda, Acos)
HAHAHA_DECLARE_UNARY_EXECUTOR(cuda, Atan)
HAHAHA_DECLARE_UNARY_EXECUTOR(cuda, Abs)
HAHAHA_DECLARE_UNARY_EXECUTOR(cuda, Sign)
HAHAHA_DECLARE_UNARY_EXECUTOR(cuda, Ceil)
HAHAHA_DECLARE_UNARY_EXECUTOR(cuda, Floor)
HAHAHA_DECLARE_UNARY_EXECUTOR(cuda, Round)
HAHAHA_DECLARE_UNARY_EXECUTOR(cuda, Trunc)
HAHAHA_DECLARE_UNARY_EXECUTOR(cuda, Sinh)
HAHAHA_DECLARE_UNARY_EXECUTOR(cuda, Cosh)
HAHAHA_DECLARE_UNARY_EXECUTOR(cuda, Tanh)
HAHAHA_DECLARE_UNARY_EXECUTOR(cuda, Asinh)
HAHAHA_DECLARE_UNARY_EXECUTOR(cuda, Acosh)
HAHAHA_DECLARE_UNARY_EXECUTOR(cuda, Atanh)
HAHAHA_DECLARE_UNARY_EXECUTOR(cuda, Log10)
HAHAHA_DECLARE_UNARY_EXECUTOR(cuda, Log2)
HAHAHA_DECLARE_UNARY_EXECUTOR(cuda, Log1p)
HAHAHA_DECLARE_UNARY_EXECUTOR(cuda, Exp2)
HAHAHA_DECLARE_UNARY_EXECUTOR(cuda, Expm1)
HAHAHA_DECLARE_UNARY_EXECUTOR(cuda, Cbrt)
HAHAHA_DECLARE_UNARY_EXECUTOR(cuda, Erf)
HAHAHA_DECLARE_UNARY_EXECUTOR(cuda, Erfc)
HAHAHA_DECLARE_UNARY_EXECUTOR(cuda, Tgamma)
HAHAHA_DECLARE_UNARY_EXECUTOR(cuda, Lgamma)
#endif

#undef HAHAHA_DECLARE_UNARY_EXECUTOR

} // namespace h3::core::compute

#endif // HAHAHA_UNARYOPERATOREXECUTOR_H
