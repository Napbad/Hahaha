#include <gtest/gtest.h>
#include "optimizers/SGDOptimizer.h"
#include "core/compute/Variable.h"

using namespace hahaha::ml;
using namespace hahaha::common::ds;

TEST(SGDOptimizerTest, Step) {
    // 1. Setup
    auto initial_data = Tensor<float>({2}, {10.0f, -5.0f});
    auto grad_data = Tensor<float>({2}, {2.0f, -4.0f});
    
    auto param = Variable<float>(initial_data, true);
    // Manually set the gradient for the test
    param.grad() = grad_data;

    ds::Vector<Variable<float>*> params = {&param};
    double learning_rate = 0.1;
    SGDOptimizer<float> optimizer(params, learning_rate);

    // 2. Action
    optimizer.step();

    // 3. Assertion
    auto& updated_data = static_cast<Tensor<float>&>(param);
    EXPECT_FLOAT_EQ(updated_data[0], 10.0f - (0.1f * 2.0f));
    EXPECT_FLOAT_EQ(updated_data[1], -5.0f - (0.1f * -4.0f));
}

TEST(SGDOptimizerTest, ZeroGrad) {
    // 1. Setup
    auto initial_data = Tensor<float>({2}, {10.0f, -5.0f});
    auto grad_data = Tensor<float>({2}, {2.0f, -4.0f});

    auto param = Variable<float>(initial_data, true);
    param.grad() = grad_data;

    ds::Vector<Variable<float>*> params = {&param};
    SGDOptimizer<float> optimizer(params, 0.1);
    
    // Ensure gradient is not zero initially
    ASSERT_NE(param.grad()[0], 0.0f);

    // 2. Action
    optimizer.zero_grad();

    // 3. Assertion
    EXPECT_FLOAT_EQ(param.grad()[0], 0.0f);
    EXPECT_FLOAT_EQ(param.grad()[1], 0.0f);
}
