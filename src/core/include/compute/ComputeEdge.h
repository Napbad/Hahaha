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
// Created by napbad on 3/26/26.
//

#ifndef H3_CORE_COMPUTE_COMPUTE_NODE_H_
#define H3_CORE_COMPUTE_COMPUTE_NODE_H_

#include <memory>
#include "math/TensorInner.h"

namespace h3::core::compute {
class ComputeNode {
public:
    std::shared_ptr<math::TensorInner> tensor() {
        return m_tensor;
    }
private:
    std::shared_ptr<math::TensorInner> m_tensor;
};
}

#endif // H3_CORE_COMPUTE_COMPUTE_NODE_H_