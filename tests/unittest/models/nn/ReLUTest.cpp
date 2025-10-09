#include <gtest/gtest.h>
#include "models/nn/layers/ReLU.h"
#include "core/compute/Variable.h"

using namespace hahaha::ml;

TEST(ReLUTest, ForwardPass) {
    // 1. Setup
    ReLU<float> relu_layer;
    auto input_tensor = Tensor<float>({4}, {-2.0f, -0.5f, 0.0f, 3.0f});
    auto input_variable = Variable<float>(input_tensor);

    // 2. Action
    auto output = relu_layer.forward(input_variable);
    auto& output_data = static_cast<Tensor<float>&>(output);

    // 3. Assertion
    EXPECT_FLOAT_EQ(output_data[0], 0.0f);
    EXPECT_FLOAT_EQ(output_data[1], 0.0f);
    EXPECT_FLOAT_EQ(output_data[2], 0.0f);
    EXPECT_FLOAT_EQ(output_data[3], 3.0f);
}
