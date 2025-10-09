//
// Created by Napbad on 2025/10/9.
//

#ifndef RELU_H
#define RELU_H

#include "Layer.h"
#include "ml/Tensor.h"

namespace hahaha::ml {

template<typename T>
class ReLU : public Layer<T> {
public:
    Tensor<T> forward(const Tensor<T>& input) override {
        Tensor<T> result(input.shape());
        for (sizeT i = 0; i < input.size(); ++i) {
            result[i] = std::max(static_cast<T>(0), input[i]);
        }
        return result;
    }
};

} // namespace hahaha::ml

#endif //RELU_H
