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

#include "compute/operator_executor/impl/BinOperatorExecutor.h"
#include "compute/operator_executor/impl/detail/CudaElementwiseLaunch.cuh"
#include "compute/operator_executor/impl/detail/OperatorFunctors.h"

namespace h3::core::compute::cuda {

#define HAHAHA_DEFINE_BIN_EXECUTOR(Name, Functor)                                        \
    std::expected<void, Error> Name##OperatorExecutor::execute(                          \
        ComputeContext& context,                                                         \
        std::vector<utils::OwnPointer<math::TensorInner>>& operands) {                 \
        return detail::runBinaryCuda<Functor>(context, operands);                      \
    }

HAHAHA_DEFINE_BIN_EXECUTOR(Add, AddFunctor)
HAHAHA_DEFINE_BIN_EXECUTOR(Sub, SubFunctor)
HAHAHA_DEFINE_BIN_EXECUTOR(Mul, MulFunctor)
HAHAHA_DEFINE_BIN_EXECUTOR(Div, DivFunctor)
HAHAHA_DEFINE_BIN_EXECUTOR(Mod, ModFunctor)
HAHAHA_DEFINE_BIN_EXECUTOR(Pow, PowFunctor)
HAHAHA_DEFINE_BIN_EXECUTOR(Max, MaxFunctor)
HAHAHA_DEFINE_BIN_EXECUTOR(Min, MinFunctor)

#undef HAHAHA_DEFINE_BIN_EXECUTOR

} // namespace h3::core::compute::cuda
