// Copyright (c) ${original.year} - 2026 Contributors of
// Hahaha("https://github.com/Napbad/Hahaha")
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Contributors:
// Napbad (napbad.sen@gmail.com ) (https://github.com/Napbad )
//

#ifndef HAHAHA_TOPOSORT_H_6D4B37ED63D646A18B0AE1F4762A2F3B
#define HAHAHA_TOPOSORT_H_6D4B37ED63D646A18B0AE1F4762A2F3B
#include <memory>
#include <unordered_set>
#include <vector>

namespace hahaha::compute {
template <typename T> class ComputeNode;

template <typename T> class TopoSort {

  public:
    /**
     * @brief Generates a topologically sorted list of nodes.
     * 
     * The order ensures that parents appear before their children,
     * which is suitable for forward computation.
     * 
     * @param node The starting node (usually the output/loss node).
     * @return A vector of shared pointers to ComputeNodes in topo order.
     */
    std::vector<std::shared_ptr<ComputeNode<T>>>
    toTopoList(const std::shared_ptr<ComputeNode<T>>& node) {
        std::vector<std::shared_ptr<ComputeNode<T>>> res;
        std::unordered_set<std::shared_ptr<ComputeNode<T>>> visited;

        toTopoRecursiveList(node, visited, res);

        return res;
    }

  private:
    void
    toTopoRecursiveList(const std::shared_ptr<ComputeNode<T>>& node,
                        std::unordered_set<std::shared_ptr<ComputeNode<T>>>& visited,
                        std::vector<std::shared_ptr<ComputeNode<T>>>& vec) {
        if (node == nullptr || visited.contains(node)) {
            return;
        }

        visited.insert(node);
        for (auto &parent : node->parents_) {
            toTopoRecursiveList(parent, visited, vec);
        }
        vec.push_back(node);
    }
};

} // namespace hahaha::compute

#endif // HAHAHA_TOPOSORT_H_6D4B37ED63D646A18B0AE1F4762A2F3B
