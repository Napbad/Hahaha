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

#include "compute/operator_executor/impl/UnaryOperatorExecutor.h"
#include "compute/operator_executor/impl/detail/CudaElementwiseLaunch.cuh"
#include "compute/operator_executor/impl/detail/OperatorFunctors.h"

namespace h3::core::compute::cuda {

#define HAHAHA_DEFINE_UNARY_EXECUTOR(Name, Functor)                                    \
    std::expected<void, Error> Name##OperatorExecutor::execute(                          \
        ComputeContext& context,                                                       \
        std::vector<utils::OwnPointer<math::TensorInner>>& operands) {                 \
        return detail::runUnaryCuda<Functor>(context, operands);                       \
    }

HAHAHA_DEFINE_UNARY_EXECUTOR(Sqrt, SqrtFunctor)
HAHAHA_DEFINE_UNARY_EXECUTOR(Log, LogFunctor)
HAHAHA_DEFINE_UNARY_EXECUTOR(Exp, ExpFunctor)
HAHAHA_DEFINE_UNARY_EXECUTOR(Sin, SinFunctor)
HAHAHA_DEFINE_UNARY_EXECUTOR(Cos, CosFunctor)
HAHAHA_DEFINE_UNARY_EXECUTOR(Tan, TanFunctor)
HAHAHA_DEFINE_UNARY_EXECUTOR(Asin, AsinFunctor)
HAHAHA_DEFINE_UNARY_EXECUTOR(Acos, AcosFunctor)
HAHAHA_DEFINE_UNARY_EXECUTOR(Atan, AtanFunctor)
HAHAHA_DEFINE_UNARY_EXECUTOR(Abs, AbsFunctor)
HAHAHA_DEFINE_UNARY_EXECUTOR(Sign, SignFunctor)
HAHAHA_DEFINE_UNARY_EXECUTOR(Ceil, CeilFunctor)
HAHAHA_DEFINE_UNARY_EXECUTOR(Floor, FloorFunctor)
HAHAHA_DEFINE_UNARY_EXECUTOR(Round, RoundFunctor)
HAHAHA_DEFINE_UNARY_EXECUTOR(Trunc, TruncFunctor)
HAHAHA_DEFINE_UNARY_EXECUTOR(Sinh, SinhFunctor)
HAHAHA_DEFINE_UNARY_EXECUTOR(Cosh, CoshFunctor)
HAHAHA_DEFINE_UNARY_EXECUTOR(Tanh, TanhFunctor)
HAHAHA_DEFINE_UNARY_EXECUTOR(Asinh, AsinhFunctor)
HAHAHA_DEFINE_UNARY_EXECUTOR(Acosh, AcoshFunctor)
HAHAHA_DEFINE_UNARY_EXECUTOR(Atanh, AtanhFunctor)
HAHAHA_DEFINE_UNARY_EXECUTOR(Log10, Log10Functor)
HAHAHA_DEFINE_UNARY_EXECUTOR(Log2, Log2Functor)
HAHAHA_DEFINE_UNARY_EXECUTOR(Log1p, Log1pFunctor)
HAHAHA_DEFINE_UNARY_EXECUTOR(Exp2, Exp2Functor)
HAHAHA_DEFINE_UNARY_EXECUTOR(Expm1, Expm1Functor)
HAHAHA_DEFINE_UNARY_EXECUTOR(Cbrt, CbrtFunctor)
HAHAHA_DEFINE_UNARY_EXECUTOR(Erf, ErfFunctor)
HAHAHA_DEFINE_UNARY_EXECUTOR(Erfc, ErfcFunctor)
HAHAHA_DEFINE_UNARY_EXECUTOR(Tgamma, TgammaFunctor)
HAHAHA_DEFINE_UNARY_EXECUTOR(Lgamma, LgammaFunctor)

#undef HAHAHA_DEFINE_UNARY_EXECUTOR

} // namespace h3::core::compute::cuda
