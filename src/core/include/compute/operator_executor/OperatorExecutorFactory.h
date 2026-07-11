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

#ifndef HAHAHA_OPERATOREXECUTORFACTORY_H_F5649BA80A094897B83503DFF124252F
#define HAHAHA_OPERATOREXECUTORFACTORY_H_F5649BA80A094897B83503DFF124252F

#include <expected>
#include <memory>
#include <vector>

#include "Error.h"
#include "OperatorExecutor.h"
#include "backend/Device.h"
#include "defines.h"

namespace h3::core::compute {

class OperatorExecutorFactory {
  public:
    OperatorExecutorFactory();

    std::expected<OperatorExecutor*, Error> get(Operator op,
                                                backend::DeviceType deviceType);

  private:
    static SizeT cacheIndex(Operator op, backend::DeviceType deviceType);
    static std::unique_ptr<OperatorExecutor> create(Operator op,
                                                    backend::DeviceType deviceType);

    std::vector<std::unique_ptr<OperatorExecutor>> m_cachedExecutors;
};

} // namespace h3::core::compute

#endif // HAHAHA_OPERATOREXECUTORFACTORY_H_F5649BA80A094897B83503DFF124252F
