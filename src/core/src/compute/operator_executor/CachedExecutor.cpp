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

#include "compute/operator_executor/CachedExecutor.h"

#include "compute/operator_executor/impl/BinOperatorExecutor.h"
#include "compute/operator_executor/impl/MultiOperatorExecutor.h"
#include "compute/operator_executor/impl/UnaryOperatorExecutor.h"

namespace h3::core::compute {

utils::OwnPointer<OperatorExecutor> makeExecutor(const backend::Device device,
                                                 const Operator op) {
    switch (device.type()) {
    case backend::DeviceType::CPU:
        return makeCPUOperatorExecutor(op);
    case backend::DeviceType::CUDA:
        return makeCUDAOperatorExecutor(op);
    default:
        return {nullptr};
    }
}


utils::OwnPointer<OperatorExecutor> makeCPUOperatorExecutor(const Operator op) {
    switch (op) {
    case Operator::Add:
        return utils::make_own_ptr<cpu::AddOperatorExecutor>();
    case Operator::Sub:
        return utils::make_own_ptr<cpu::SubOperatorExecutor>();
    case Operator::Mul:
        return utils::make_own_ptr<cpu::MulOperatorExecutor>();
    case Operator::Div:
        return utils::make_own_ptr<cpu::DivOperatorExecutor>();
    case Operator::Mod:
        return utils::make_own_ptr<cpu::ModOperatorExecutor>();
    case Operator::Pow:
        return utils::make_own_ptr<cpu::PowOperatorExecutor>();
    case Operator::Max:
        return utils::make_own_ptr<cpu::MaxOperatorExecutor>();
    case Operator::Min:
        return utils::make_own_ptr<cpu::MinOperatorExecutor>();
    case Operator::Sqrt:
        return utils::make_own_ptr<cpu::SqrtOperatorExecutor>();
    case Operator::Log:
        return utils::make_own_ptr<cpu::LogOperatorExecutor>();
    case Operator::Exp:
        return utils::make_own_ptr<cpu::ExpOperatorExecutor>();
    case Operator::Sin:
        return utils::make_own_ptr<cpu::SinOperatorExecutor>();
    case Operator::Cos:
        return utils::make_own_ptr<cpu::CosOperatorExecutor>();
    case Operator::Tan:
        return utils::make_own_ptr<cpu::TanOperatorExecutor>();
    case Operator::Asin:
        return utils::make_own_ptr<cpu::AsinOperatorExecutor>();
    case Operator::Acos:
        return utils::make_own_ptr<cpu::AcosOperatorExecutor>();
    case Operator::Atan:
        return utils::make_own_ptr<cpu::AtanOperatorExecutor>();
    case Operator::Abs:
        return utils::make_own_ptr<cpu::AbsOperatorExecutor>();
    case Operator::Sign:
        return utils::make_own_ptr<cpu::SignOperatorExecutor>();
    case Operator::Ceil:
        return utils::make_own_ptr<cpu::CeilOperatorExecutor>();
    case Operator::Floor:
        return utils::make_own_ptr<cpu::FloorOperatorExecutor>();
    case Operator::Round:
        return utils::make_own_ptr<cpu::RoundOperatorExecutor>();
    case Operator::Trunc:
        return utils::make_own_ptr<cpu::TruncOperatorExecutor>();
    case Operator::Sinh:
        return utils::make_own_ptr<cpu::SinhOperatorExecutor>();
    case Operator::Cosh:
        return utils::make_own_ptr<cpu::CoshOperatorExecutor>();
    case Operator::Tanh:
        return utils::make_own_ptr<cpu::TanhOperatorExecutor>();
    case Operator::Asinh:
        return utils::make_own_ptr<cpu::AsinhOperatorExecutor>();
    case Operator::Acosh:
        return utils::make_own_ptr<cpu::AcoshOperatorExecutor>();
    case Operator::Atanh:
        return utils::make_own_ptr<cpu::AtanhOperatorExecutor>();
    case Operator::Log10:
        return utils::make_own_ptr<cpu::Log10OperatorExecutor>();
    case Operator::Log2:
        return utils::make_own_ptr<cpu::Log2OperatorExecutor>();
    case Operator::Log1p:
        return utils::make_own_ptr<cpu::Log1pOperatorExecutor>();
    case Operator::Exp2:
        return utils::make_own_ptr<cpu::Exp2OperatorExecutor>();
    case Operator::Expm1:
        return utils::make_own_ptr<cpu::Expm1OperatorExecutor>();
    case Operator::Cbrt:
        return utils::make_own_ptr<cpu::CbrtOperatorExecutor>();
    case Operator::Erf:
        return utils::make_own_ptr<cpu::ErfOperatorExecutor>();
    case Operator::Erfc:
        return utils::make_own_ptr<cpu::ErfcOperatorExecutor>();
    case Operator::Tgamma:
        return utils::make_own_ptr<cpu::TgammaOperatorExecutor>();
    case Operator::Lgamma:
        return utils::make_own_ptr<cpu::LgammaOperatorExecutor>();
    case Operator::Clamp:
        return utils::make_own_ptr<cpu::ClampOperatorExecutor>();
    default:
        return {nullptr};
    }
}

utils::OwnPointer<OperatorExecutor> makeCUDAOperatorExecutor(const Operator op) {
#ifdef HAHAHA_USE_CUDA
    switch (op) {
    case Operator::Add:
        return utils::make_own_ptr<cuda::AddOperatorExecutor>();
    case Operator::Sub:
        return utils::make_own_ptr<cuda::SubOperatorExecutor>();
    case Operator::Mul:
        return utils::make_own_ptr<cuda::MulOperatorExecutor>();
    case Operator::Div:
        return utils::make_own_ptr<cuda::DivOperatorExecutor>();
    case Operator::Mod:
        return utils::make_own_ptr<cuda::ModOperatorExecutor>();
    case Operator::Pow:
        return utils::make_own_ptr<cuda::PowOperatorExecutor>();
    case Operator::Max:
        return utils::make_own_ptr<cuda::MaxOperatorExecutor>();
    case Operator::Min:
        return utils::make_own_ptr<cuda::MinOperatorExecutor>();
    case Operator::Sqrt:
        return utils::make_own_ptr<cuda::SqrtOperatorExecutor>();
    case Operator::Log:
        return utils::make_own_ptr<cuda::LogOperatorExecutor>();
    case Operator::Exp:
        return utils::make_own_ptr<cuda::ExpOperatorExecutor>();
    case Operator::Sin:
        return utils::make_own_ptr<cuda::SinOperatorExecutor>();
    case Operator::Cos:
        return utils::make_own_ptr<cuda::CosOperatorExecutor>();
    case Operator::Tan:
        return utils::make_own_ptr<cuda::TanOperatorExecutor>();
    case Operator::Asin:
        return utils::make_own_ptr<cuda::AsinOperatorExecutor>();
    case Operator::Acos:
        return utils::make_own_ptr<cuda::AcosOperatorExecutor>();
    case Operator::Atan:
        return utils::make_own_ptr<cuda::AtanOperatorExecutor>();
    case Operator::Abs:
        return utils::make_own_ptr<cuda::AbsOperatorExecutor>();
    case Operator::Sign:
        return utils::make_own_ptr<cuda::SignOperatorExecutor>();
    case Operator::Ceil:
        return utils::make_own_ptr<cuda::CeilOperatorExecutor>();
    case Operator::Floor:
        return utils::make_own_ptr<cuda::FloorOperatorExecutor>();
    case Operator::Round:
        return utils::make_own_ptr<cuda::RoundOperatorExecutor>();
    case Operator::Trunc:
        return utils::make_own_ptr<cuda::TruncOperatorExecutor>();
    case Operator::Sinh:
        return utils::make_own_ptr<cuda::SinhOperatorExecutor>();
    case Operator::Cosh:
        return utils::make_own_ptr<cuda::CoshOperatorExecutor>();
    case Operator::Tanh:
        return utils::make_own_ptr<cuda::TanhOperatorExecutor>();
    case Operator::Asinh:
        return utils::make_own_ptr<cuda::AsinhOperatorExecutor>();
    case Operator::Acosh:
        return utils::make_own_ptr<cuda::AcoshOperatorExecutor>();
    case Operator::Atanh:
        return utils::make_own_ptr<cuda::AtanhOperatorExecutor>();
    case Operator::Log10:
        return utils::make_own_ptr<cuda::Log10OperatorExecutor>();
    case Operator::Log2:
        return utils::make_own_ptr<cuda::Log2OperatorExecutor>();
    case Operator::Log1p:
        return utils::make_own_ptr<cuda::Log1pOperatorExecutor>();
    case Operator::Exp2:
        return utils::make_own_ptr<cuda::Exp2OperatorExecutor>();
    case Operator::Expm1:
        return utils::make_own_ptr<cuda::Expm1OperatorExecutor>();
    case Operator::Cbrt:
        return utils::make_own_ptr<cuda::CbrtOperatorExecutor>();
    case Operator::Erf:
        return utils::make_own_ptr<cuda::ErfOperatorExecutor>();
    case Operator::Erfc:
        return utils::make_own_ptr<cuda::ErfcOperatorExecutor>();
    case Operator::Tgamma:
        return utils::make_own_ptr<cuda::TgammaOperatorExecutor>();
    case Operator::Lgamma:
        return utils::make_own_ptr<cuda::LgammaOperatorExecutor>();
    case Operator::Clamp:
        return utils::make_own_ptr<cuda::ClampOperatorExecutor>();
    default:
        return nullptr;
    }
#else
    (void)op;
    return nullptr;
#endif
}

} // namespace h3::core::compute