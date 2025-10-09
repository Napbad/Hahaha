#include <gtest/gtest.h>
#include "models/nn/Sequential.h"
#include "models/nn/layers/Linear.h"
#include "models/nn/layers/ReLU.h"
#include "core/compute/Variable.h"

using namespace hahaha::ml;

TEST(SequentialTest, ModelConstructionAndForwardPass) {
    // 1. Setup
    sizeT input_features = 10;
    sizeT hidden_features = 8;
    sizeT output_features = 3;

    auto model = Sequential<float>();
    model.add(new Linear<float>(input_features, hidden_features));
    model.add(new ReLU<float>());
    model.add(new Linear<float>(hidden_features, output_features));

    // 2. Parameters assertion
    auto params = model.parameters();
    // Linear1 (weights, bias) + Linear2 (weights, bias)
    ASSERT_EQ(params.size(), 4); 

    // 3. Forward pass
    auto input_tensor = Tensor<float>::ones({1, input_features});
    auto input_variable = Variable<float>(input_tensor);

    auto output = model.forward(input_variable);

    // 4. Forward pass assertion
    EXPECT_EQ(output.shape(), hahaha::common::ds::Vector<sizeT>({1, output_features}));
}
