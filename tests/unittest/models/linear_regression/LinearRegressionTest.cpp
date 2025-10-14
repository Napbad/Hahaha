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
// Created by Napbad on 10/2/25.
//

#include <cmath>
#include <gtest/gtest.h>
#include <linear_regression/LinearRegression.h>
#include <ml/Tensor.h>

HHH_NAMESPACE_IMPORT
using hahaha::ml::LinearRegression;

class LinearRegressionTest : public ::testing::Test
{
  public:
    void SetUp() override
    {
        // Create simple 2D dataset for testing: y = 2x + 1
        features_2d = ml::Tensor<f32>({3, 1});
        labels_2d = ml::Tensor<f32>({3, 1});

        // Sample 1: x=1, y=3
        features_2d.set({0, 0}, 1.0f);
        labels_2d.set({0, 0}, 3.0f);

        // Sample 2: x=2, y=5
        features_2d.set({1, 0}, 2.0f);
        labels_2d.set({1, 0}, 5.0f);

        // Sample 3: x=3, y=7
        features_2d.set({2, 0}, 3.0f);
        labels_2d.set({2, 0}, 7.0f);

        // Create multi-dimensional dataset: y = 2x1 + 3x2 + 1
        features_multi = ml::Tensor<f32>({3, 2});
        labels_multi = ml::Tensor<f32>({3, 1});

        // Sample 1: x1=1, x2=1, y=6
        features_multi.set({0, 0}, 1.0f);
        features_multi.set({0, 1}, 1.0f);
        labels_multi.set({0, 0}, 6.0f);

        // Sample 2: x1=2, x2=2, y=11
        features_multi.set({1, 0}, 2.0f);
        features_multi.set({1, 1}, 2.0f);
        labels_multi.set({1, 0}, 11.0f);

        // Sample 3: x1=3, x2=3, y=16
        features_multi.set({2, 0}, 3.0f);
        features_multi.set({2, 1}, 3.0f);
        labels_multi.set({2, 0}, 16.0f);
    }

    void TearDown() override
    {
    }

  protected:
    ml::Tensor<f32> features_2d;
    ml::Tensor<f32> labels_2d;
    ml::Tensor<f32> features_multi;
    ml::Tensor<f32> labels_multi;

    // Helper function to check if two floats are approximately equal
    static bool
    isApproxEqual(const f32 a, const f32 b, const f32 tolerance = 0.1f)
    {
        return std::abs(a - b) < tolerance;
    }
};

// Test basic model properties
TEST_F(LinearRegressionTest, BasicPropertiesTest)
{
    LinearRegression lr;

    // Test model name
    EXPECT_EQ(lr.modelName(), ds::String("LinearRegression"));

    // Test parameter count before training (should be 0 since weights are
    // empty)
    EXPECT_EQ(lr.parameterCount(), 1); // Only bias initially
}

// Test training with simple 2D data
TEST_F(LinearRegressionTest, TrainSimple2DTest)
{
    LinearRegression lr;

    // Train the model
    ASSERT_NO_THROW(lr.train(features_2d, labels_2d));

    f32 learning_rate = lr.learningRate();
    std::cout << "Learning rate: " << learning_rate << std::endl;

    // Check parameter count after training (1 weight + 1 bias = 2)
    EXPECT_EQ(lr.parameterCount(), 2);

    // Test predictions on training data
    ml::Tensor<f32> test_feature({1});
    test_feature.set({0}, 1.0f);
    const f32 pred1 = lr.predict(test_feature);
    std::cout << "Prediction for x=1: " << pred1 << std::endl;

    EXPECT_TRUE(isApproxEqual(pred1, 3.0f, 0.5f))
        << "Prediction for x=1 should be close to 3";

    test_feature.set({0}, 2.0f);
    f32 pred2 = lr.predict(test_feature);
    EXPECT_TRUE(isApproxEqual(pred2, 5.0f, 0.5f))
        << "Prediction for x=2 should be close to 5";
}

// Test training with multi-dimensional data
TEST_F(LinearRegressionTest, TrainMultiDimensionalTest)
{
    LinearRegression lr;

    // Train the model
    ASSERT_NO_THROW(lr.train(features_multi, labels_multi));

    auto weights = lr.weights();

    // Check parameter count after training (2 weights + 1 bias = 3)
    EXPECT_EQ(lr.parameterCount(), 3);

    // Test prediction
    ml::Tensor<f32> test_feature({2});
    test_feature.set({0}, 1.0f);
    test_feature.set({1}, 1.0f);
    f32 pred = lr.predict(test_feature);
    EXPECT_TRUE(isApproxEqual(pred, 6.0f, 1.0f))
        << "Prediction should be reasonably close to expected value";
}

// Test prediction before training (should handle gracefully)
TEST_F(LinearRegressionTest, PredictBeforeTrainingTest)
{
    LinearRegression lr;

    ml::Tensor<f32> test_feature({1});
    test_feature.set({0}, 1.0f);

    // Should return bias value (0.0) since weights are empty
    f32 pred = lr.predict(test_feature);
    EXPECT_EQ(pred, 0.0f);
}

// Test checkStatus method
TEST_F(LinearRegressionTest, CheckStatusTest)
{
    LinearRegression lr;

    // Train first
    lr.train(features_2d, labels_2d);

    // Now status should be OK
    EXPECT_NO_THROW(lr.checkStatus(features_2d, labels_2d));

    // Test with mismatched sample sizes
    ml::Tensor<f32> bad_labels({2, 1}); // Different number of samples
    bad_labels.fill(1.0f);
    EXPECT_THROW(lr.checkStatus(features_2d, bad_labels), std::runtime_error);
}

// Test save and load functionality (currently stubs)
TEST_F(LinearRegressionTest, SaveLoadTest)
{
    LinearRegression lr;

    // Current implementation just returns true
    EXPECT_TRUE(lr.save(ds::String("test_model.bin")));
    EXPECT_TRUE(lr.load(ds::String("test_model.bin")));
}

// Test edge cases
TEST_F(LinearRegressionTest, EdgeCasesTest)
{
    LinearRegression lr;

    // Test with single sample
    ml::Tensor<f32> single_feature({1, 1});
    ml::Tensor<f32> single_label({1, 1});
    single_feature.set({0, 0}, 5.0f);
    single_label.set({0, 0}, 10.0f);

    EXPECT_NO_THROW(lr.train(single_feature, single_label));

    // Test prediction with the same input
    ml::Tensor<f32> test_feature({1});
    test_feature.set({0}, 5.0f);
    f32 pred = lr.predict(test_feature);
    EXPECT_TRUE(isApproxEqual(pred, 10.0f, 1.0f))
        << "Should predict close to training value";
}

// Test multiple training sessions (retraining)
TEST_F(LinearRegressionTest, RetrainingTest)
{
    LinearRegression lr;

    // First training
    EXPECT_NO_THROW(lr.train(features_2d, labels_2d));

    // Second training with incompatible features should throw an error
    EXPECT_THROW(lr.train(features_multi, labels_multi), std::runtime_error);
}

// Performance test with larger dataset
TEST_F(LinearRegressionTest, LargerDatasetTest)
{
    LinearRegression lr;

    // Create a larger dataset
    const sizeT num_samples = 100;
    ml::Tensor<f32> large_features({num_samples, 1});
    ml::Tensor<f32> large_labels({num_samples, 1});

    // Generate y = 3x + 2 with some noise
    for (sizeT i = 0; i < num_samples; ++i)
    {
        f32 x = static_cast<f32>(i) / 10.0f;
        large_features.set({i, 0}, x);
        large_labels.set({i, 0}, 3.0f * x + 2.0f);
    }

    EXPECT_NO_THROW(lr.train(large_features, large_labels));

    // Test prediction accuracy
    ml::Tensor<f32> test_feature({1});
    test_feature.set({0}, 5.0f);
    f32 pred = lr.predict(test_feature);
    f32 expected = 3.0f * 5.0f + 2.0f; // 17.0f
    EXPECT_TRUE(isApproxEqual(pred, expected, 2.0f))
        << "Should learn the underlying pattern reasonably well";
}
