// Copyright (c) 2025 Napbad
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Email: napbad.sen@gmail.com
// GitHub: https://github.com/Napbad

#ifndef HIAHIAHIA_LINEARREGRESSION_H
#define HIAHIAHIA_LINEARREGRESSION_H

#include "ml/common/Tensor.h"
#include <common/ds/Vec.h>
#include <ml/model/Model.h>

namespace hahaha {

    class LinearRegression : public Model {
    public:
        LinearRegression() = default;

        void train(const ml::Tensor<f32>& features, const ml::Tensor<f32>& labels) override;
        [[nodiscard]] f32 predict(const ml::Tensor<f32>& features) const override;

    private:
        ml::Tensor<f32> _weights;
        f32 _bias{0.0f};
    };

} // namespace hahaha

#endif // HIAHIAHIA_LINEARREGRESSION_H
