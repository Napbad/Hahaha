//  Copyright (c) 2026 Contributors of hahaha(https://github.com/Napbad/Hahaha)
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
//  jiansongshen (jason.shen111@outlook.com) (https://github.com/jiansongshen)
//
//

#ifndef HAHAHA_PARAMETERS_H_0562A02563D046C99A4827535147DD6E
#define HAHAHA_PARAMETERS_H_0562A02563D046C99A4827535147DD6E
#include <vector>

#include "Tensor.h"

namespace hahaha::ml {
template <typename T> class Parameters {
  public:
    void addParameter(Tensor<T> parameter) {
        parameters.push_back(parameter);
    }

    std::vector<Tensor<T>>& getParameters() const {
        return parameters;
    }

  private:
    std::vector<Tensor<T>> parameters;
};
} // namespace hahaha::ml
#endif // HAHAHA_PARAMETERS_H_0562A02563D046C99A4827535147DD6E
