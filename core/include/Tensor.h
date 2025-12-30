// Copyright (c) 2025 Contributors of hahaha(https://github.com/Napbad/Hahaha)
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

#ifndef HAHAHA_TENSOR_H
#define HAHAHA_TENSOR_H

#include <type_traits>
#include <utility>
#include <vector>

#include "compute/compute_graph/ControlBlock.h"
#include "utils/common/HelperStruct.h"

namespace hahaha {

using compute::ControlBlock;
template <typename T,
          typename = std::enable_if<utils::isLegalDataType<T>::value>>
class Tensor {
  public:
    Tensor() : controlBlock(nullptr) {
    }

    // NOLINTNEXTLINE
    Tensor(const Tensor<T>& other)
        : controlBlock(other.controlBlock), parents(other.parents),
          children(other.children), isWeak(other.isWeak) {

        for (auto& child : children) {
            bool exist = false;
            for (auto& childsParent : child.parents) {
                if (childsParent == this->controlBlock) {
                    exist = true;
                    break;
                }
            }
            if (!exist) {
                child.parents.push_back(*this);
            }
        }

        for (auto& parent : parents) {
            bool exist = false;
            for (auto& parentsChild : parent.children) {
                if (parentsChild == this->controlBlock) {
                    exist = true;
                    break;
                }
            }
            if (!exist) {
                parent.children.push_back(this->createWeakRef());
            }
        }

        if (controlBlock) {
            if (isWeak) {
                controlBlock->weakCount++;
            } else {
                controlBlock->strongCount++;
            }
        }
    }

    // NOLINTNEXTLINE
    Tensor(Tensor<T>&& other)
        : parents(std::move(other.parents)),
          children(std::move(other.children)),
          controlBlock(other.controlBlock) {

        other.controlBlock = nullptr;
    }

    ~Tensor() {
        release();
    }

    Tensor& operator=(const Tensor<T>& other) {
        if (other == this) {
            return *this;
        }

        this->release();

        for (auto& child : children) {
            bool exist = false;
            for (auto& childsParent : child.parents) {
                if (childsParent == this->controlBlock) {
                    exist = true;
                    break;
                }
            }
            if (!exist) {
                child.parents.push_back(*this);
            }
        }

        for (auto& parent : parents) {
            bool exist = false;
            for (auto& parentsChild : parent.children) {
                if (parentsChild == this->controlBlock) {
                    exist = true;
                    break;
                }
            }
            if (!exist) {
                parent.children.push_back(this->createWeakRef());
            }
        }

        if (controlBlock) {
            if (isWeak) {
                controlBlock->weakCount++;
            } else {
                controlBlock->strongCount++;
            }
        }
        return *this;
    }

    Tensor& operator=(Tensor<T>&& other) {
        if (other == this) {
            return *this;
        }

        this->release();
        this->controlBlock = other.controlBlock;
        other.controlBlock = nullptr;
        this->isWeak = other.isWeak;
        this->children = std::move(other.children);
        this->parents = std::move(other.parents);

        return *this;
    }

    bool operator==(const Tensor<T>& other) {
        return controlBlock == other.controlBlock;
    }

  private:
    std::vector<ControlBlock<T>*> parents;  // should be strong pointers
    std::vector<ControlBlock<T>*> children; // should be weak pointers

    // memory management
    ControlBlock<T>* controlBlock;
    bool isWeak = false;

    Tensor<T>(ControlBlock<T>* controlBlock, bool weak) {
        this->controlBlock = controlBlock;
        this->isWeak = weak;
    }

    Tensor<T> createWeakRef() {
        return Tensor<T>(controlBlock, true);
    }

    bool isInvalid() {
        return controlBlock == nullptr || controlBlock->strongCount == 0;
    }

    void release() {
        if (!controlBlock) {
            return;
        }
        if (isWeak) {
            controlBlock->weakCount--;
            if (controlBlock->strongCount == 0
                && controlBlock->weakCount == 0) {
                delete controlBlock->data;
                delete controlBlock;
            }
            return;
        }
        controlBlock->strongCount--;
        if (controlBlock->strongCount == 0 && controlBlock->weakCount == 0) {
            delete controlBlock->data;
            delete controlBlock;
        }
    }
};

} // namespace hahaha

#endif // HAHAHA_TENSOR_H