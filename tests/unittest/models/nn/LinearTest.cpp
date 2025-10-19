#include "models/nn/layers/Linear.h"

#include <gtest/gtest.h>

#include "core/compute/Variable.h"

using namespace hahaha::ml;
using namespace hahaha::core::ds;

TEST(LinearTest, InitializationAndForwardPass)
{
    // 1. Setup
    sizeT input_features = 10;
    sizeT output_features = 5;
    Linear<float> linear_layer(input_features, output_features);

    // 2. Initialization assertions
    auto params = linear_layer.parameters();
    ASSERT_EQ(params.size(), 2);

    Variable<float>* weights = params[0];
    Variable<float>* bias = params[1];

    EXPECT_EQ(weights->shape(),
              Vector<sizeT>({input_features, output_features}));
    EXPECT_TRUE(weights->requiresGrad());

    EXPECT_EQ(bias->shape(), Vector<sizeT>({1, output_features}));
    EXPECT_TRUE(bias->requiresGrad());

    // 3. Forward pass
    auto input_tensor = Tensor<float>::ones({1, input_features});
    auto input_variable = Variable(input_tensor);

    auto output = linear_layer.forward(input_variable);

    // 4. Forward pass assertions
    EXPECT_EQ(output.shape(), Vector<sizeT>({1, output_features}));

    // Backward propagates to parameters
    output.backward();
    EXPECT_EQ(weights->grad().shape(), weights->shape());
    EXPECT_EQ(bias->grad().shape(), bias->shape());
}
