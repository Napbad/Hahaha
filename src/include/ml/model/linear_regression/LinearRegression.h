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

//
// Created by Napbad on 7/19/25.
//

#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H
#include "include/common/ds/Tensor.h"
#include "include/ml/model/Model.h"

namespace hiahiahia {

    class LinearRegression : public Model {
    public:
        LinearRegression();
        void train(const h3vec<h3vec<float>> &features, const h3vec<float> &labels) override;
        [[nodiscard]] float predict(const h3vec<float> &features) const override;

    private:
        Tensor<float> _inputFeatures;
        Tensor<float> _outputFeatures;

        Tensor<float> _weights;
        Tensor<float> _bias;
    };

} // namespace hiahiahia



#endif //LINEARREGRESSION_H
