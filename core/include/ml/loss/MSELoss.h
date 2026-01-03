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

#ifndef HAHAHA_MSELOSS_H_C3D1AB8B02D2451FA2170E88697A5793
#define HAHAHA_MSELOSS_H_C3D1AB8B02D2451FA2170E88697A5793

#include "Loss.h"

namespace hahaha::ml {

template <typename T> class MSELoss : public Loss<T> {
  public:
    Tensor<T> computeLoss(Tensor<T> yTrue, Tensor<T> yPredict) {
        // NOTE: Tensor::sum() currently returns a scalar value (T), not a Tensor.
        // Wrap it back into a scalar Tensor.
        return Tensor<T>(((yTrue - yPredict) * (yTrue - yPredict)).sum());
    }
};

template <typename T>
Tensor<T> computeMSELoss(Tensor<T> yTrue, Tensor<T> yPredict) {
    static MSELoss<T> loss;
    return loss.computeLoss(yTrue, yPredict);
}

} // namespace hahaha::ml

#endif // HAHAHA_MSELOSS_H_C3D1AB8B02D2451FA2170E88697A5793
