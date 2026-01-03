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

#ifndef HAHAHA_LINEARREGRESSION_H_4305E4B0E7784CD1969E389923F7743D
#define HAHAHA_LINEARREGRESSION_H_4305E4B0E7784CD1969E389923F7743D
#include "Model.h"
#include "ml/Parameters.h"
#include "ml/loss/MSELoss.h"
#include "ml/optimizer/SGDOptimizer.h"

namespace hahaha::ml {

template <typename T> class LinearRegression : public Model<T> {
  public:
    LinearRegression() : weight_(T(0)), bias_(T(0)) {
    }

    Parameters<T> getParameters() override {
        Parameters<T> parameters;

        parameters.addParameter(weight_);
        parameters.addParameter(bias_);
        return parameters;
    }

    void setWeights(std::vector<T> weights) {
        weight_ = Tensor<T>::buildFromVector(weights);
    }

    void setBias(std::vector<T> bias) {
        bias_ = Tensor<T>::buildFromVector(bias);
    }

    void train(Tensor<T> x, Tensor<T> y) override {
        // x shape is s * n1 (Size of samples and features Number)
        // y shape is s * n2 (Size of samples and output Number)

        // reshape to a matrix to support common situations
        auto xShape = x.getShape();
        if (xShape.size() != 2) {
            x = x.reshape({xShape[0], 1}); // n rows and 1 column
        }

        auto shape = x.getShape();
        auto yPredict = x.matmul(weight_) + bias_;
        auto mseLoss = computeMSELoss(y, yPredict);

        SGDOptimizer<T> sgdOptimizer({}, T(0.00001));
        sgdOptimizer.addParameter(weight_);
        sgdOptimizer.addParameter(bias_);

        sgdOptimizer.zeroGrad();
        mseLoss.backward();
        sgdOptimizer.step();
    }

  private:
    Tensor<T> weight_; // shape is n1 * n2 (Input features Number and output
                       // sample Number)
    Tensor<T> bias_;   // shape is n2 (output Number)
};

} // namespace hahaha::ml

#endif // HAHAHA_LINEARREGRESSION_H_4305E4B0E7784CD1969E389923F7743D
