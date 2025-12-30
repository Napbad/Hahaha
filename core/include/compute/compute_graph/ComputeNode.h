// Copyright (c) 2025 Napbad
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Email: napbad.sen@gmail.com
// GitHub: https://github.com/Napbad
//
// Created: 2025-12-30 05:21:41 by Napbad
//

#ifndef HAHAHA_COMPUTE_COMPUTE_GRAPH_COMPUTE_NODE_H
#define HAHAHA_COMPUTE_COMPUTE_GRAPH_COMPUTE_NODE_H

#include <memory>
#include <type_traits>

#include "math/TensorWrapper.h"
#include "utils/common/HelperStruct.h"
namespace hahaha::compute {

template <typename T,
          typename = std::enable_if<utils::isLegalDataType<T>::value>>
class ComputeNode {
  public:
    explicit ComputeNode<T> (math::TensorWrapper<T> wrapper) {
        this->data = wrapper;
    } 


    void addParent(ComputeNode<T> *node) {
        this->parents.push_back(node);
    }
  private:
    std::vector<ComputeNode*> parents;
    std::shared_ptr<math::TensorWrapper<T>> data;
   
};

} // namespace hahaha::compute

#endif // HAHAHA_COMPUTE_COMPUTE_GRAPH_COMPUTE_NODE_H
