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

#ifndef HAHAHA_CACHEDEXECUTOR_0FEE6E02917E4458AEF393076DAD10C2_H
#define HAHAHA_CACHEDEXECUTOR_0FEE6E02917E4458AEF393076DAD10C2_H
#include "OperatorExecutor.h"
#include "defines.h"

namespace h3::core::compute {
utils::OwnPointer<OperatorExecutor> makeExecutor(backend::Device device, Operator op);

utils::OwnPointer<OperatorExecutor> makeCPUOperatorExecutor(Operator op);
utils::OwnPointer<OperatorExecutor> makeCUDAOperatorExecutor(Operator op);

class CachedExecutor {
public:
    utils::OwnPointer<OperatorExecutor> get(const Operator op) {
        if (m_cachedPtr[static_cast<SizeT>(op)]) {
            return m_cachedPtr[static_cast<SizeT>(op)];
        }
        m_cachedPtr[static_cast<SizeT>(op)] = makeExecutor(m_device, op);
            return m_cachedPtr[static_cast<SizeT>(op)];
    }
private:
    backend::Device m_device;
    utils::OwnPointer<OperatorExecutor> m_cachedPtr[static_cast<SizeT>(Operator::Count)];
};
}

#endif // HAHAHA_CACHEDEXECUTOR_0FEE6E02917E4458AEF393076DAD10C2_H
