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

#include "compute/operator_executor/impl/BinOperatorExecutor.h"
#include "compute/operator_executor/impl/detail/ElementwiseKernel.h"
#include "compute/operator_executor/impl/detail/OperatorFunctors.h"

namespace h3::core::compute::cpu {

#define HAHAHA_DEFINE_BIN_EXECUTOR(Name, Functor)                                        \
    std::expected<void, Error> Name##OperatorExecutor::execute(                          \
        ComputeContext& context,                                                         \
        std::vector<utils::OwnPointer<math::TensorInner>>& operands) {                  \
        return detail::runBinary<Functor>(context, operands);                          \
    }
std::expected<void, Error> AddOperatorExecutor::execute(
    ComputeContext& context,
    std::vector<utils::OwnPointer<math::TensorInner> >& operands) {
    return detail::runBinary<detail::AddFunctor>(context, operands);
}
HAHAHA_DEFINE_BIN_EXECUTOR(Add, detail::AddFunctor)
HAHAHA_DEFINE_BIN_EXECUTOR(Sub, detail::SubFunctor)
HAHAHA_DEFINE_BIN_EXECUTOR(Mul, detail::MulFunctor)
HAHAHA_DEFINE_BIN_EXECUTOR(Div, detail::DivFunctor)
HAHAHA_DEFINE_BIN_EXECUTOR(Mod, detail::ModFunctor)
HAHAHA_DEFINE_BIN_EXECUTOR(Pow, detail::PowFunctor)
HAHAHA_DEFINE_BIN_EXECUTOR(Max, detail::MaxFunctor)
HAHAHA_DEFINE_BIN_EXECUTOR(Min, detail::MinFunctor)

#undef HAHAHA_DEFINE_BIN_EXECUTOR

} // namespace h3::core::compute::cpu
