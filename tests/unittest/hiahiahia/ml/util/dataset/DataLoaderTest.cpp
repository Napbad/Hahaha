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

#include "ml/util/dataset/DataLoader.h"

#include "common/ds/Vec.h"
#include "common/ds/Str.h"
#include "ml/util/dataset/CSVDataset.h"
#include <fstream>
#include <gtest/gtest.h>
#include <memory>

using namespace hahaha::ml;
using namespace hahaha;
using namespace hahaha::common;

class DataLoaderTest : public ::testing::Test {
public:
    void SetUp() override {
        // Create a temporary CSV file for testing
        csvContent = R"(x1,x2,y
1.0,2.0,0
2.0,3.0,1
3.0,4.0,0
4.0,5.0,1
5.0,6.0,0
6.0,7.0,1)";

        std::ofstream csvFile("test_dataloader.csv");
        csvFile << csvContent;
        csvFile.close();

        // Create and load dataset
        dataset  = std::make_shared<CSVDataset<f32>>(
            ds::Str("test_dataloader.csv"), ds::Vec<sizeT>{0, 1}, ds::Vec<sizeT>{2}, true, ',', ds::Str());
        auto res = dataset->load();
        ASSERT_TRUE(res.isOk()) << "Failed to load dataset";
    }

    void TearDown() override {
        // Clean up temporary file
        std::remove("test_dataloader.csv");
    }

protected:
    std::string csvContent;
    std::shared_ptr<CSVDataset<f32>> dataset;
};

TEST_F(DataLoaderTest, ConstructorTest) {
    // Create data loader
    const DataLoader<f32> dataLoader(dataset, 2, false, false);

    // Check properties
    EXPECT_EQ(dataLoader.datasetSize(), 6);
    EXPECT_EQ(dataLoader.batchSize(), 2);
    EXPECT_EQ(dataLoader.numBatches(), 3); // 6 samples with batch size 2 = 3 batches
    EXPECT_TRUE(dataLoader.hasNext());
}

TEST_F(DataLoaderTest, NextBatchTest) {
    // Create data loader
    DataLoader<f32> dataLoader(dataset, 2, false, false);

    // Get first batch
    auto batchRes = dataLoader.nextBatch();
    ASSERT_TRUE(batchRes.isOk()) << "Failed to get batch";

    auto batch = batchRes.unwrap();
    EXPECT_EQ(batch.size(), 2);

    // Check first sample in batch
    auto& firstSample = batch[0];
    EXPECT_EQ(firstSample.featureDim(), 2);
    EXPECT_EQ(firstSample.labelDim(), 1);

    // Check second sample in batch
    auto& secondSample = batch[1];
    EXPECT_EQ(secondSample.featureDim(), 2);
    EXPECT_EQ(secondSample.labelDim(), 1);

}

TEST_F(DataLoaderTest, HasNextTest) {
    // Create data loader
    DataLoader<f32> dataLoader(dataset, 2, false, false);

    // Should have next initially
    EXPECT_TRUE(dataLoader.hasNext());

    // Get all batches
    auto batch1 = dataLoader.nextBatch();
    EXPECT_TRUE(dataLoader.hasNext());

    auto batch2 = dataLoader.nextBatch();
    EXPECT_TRUE(dataLoader.hasNext());

    auto batch3 = dataLoader.nextBatch();
    EXPECT_FALSE(dataLoader.hasNext()); // No more batches

    // Try to get another batch (should fail)
    auto batch4 = dataLoader.nextBatch();
    EXPECT_TRUE(batch4.isErr());
}

TEST_F(DataLoaderTest, DropLastTest) {
    // Create data loader with dropLast=true
    DataLoader<f32> dataLoader(dataset, 4, false, true); // Batch size 4 with 6 samples

    // With 6 samples and batch size 4, we should have 1 batch (dropping the last incomplete batch)
    EXPECT_EQ(dataLoader.numBatches(), 1);

    // Get first batch
    auto batch1Res = dataLoader.nextBatch();
    ASSERT_TRUE(batch1Res.isOk());
    EXPECT_EQ(batch1Res.unwrap().size(), 4);

    // Try to get second batch (should fail because we dropped the last incomplete batch)
    auto batch2Res = dataLoader.nextBatch();
    EXPECT_TRUE(batch2Res.isErr());
}

TEST_F(DataLoaderTest, NoDropLastTest) {
    // Create data loader with dropLast=false
    DataLoader<f32> dataLoader(dataset, 4, false, false); // Batch size 4 with 6 samples

    // With 6 samples and batch size 4, we should have 2 batches (not dropping the last incomplete batch)
    EXPECT_EQ(dataLoader.numBatches(), 2);

    // Get first batch
    auto batch1Res = dataLoader.nextBatch();
    ASSERT_TRUE(batch1Res.isOk());
    EXPECT_EQ(batch1Res.unwrap().size(), 4);

    // Get second batch (should be size 2)
    auto batch2Res = dataLoader.nextBatch();
    ASSERT_TRUE(batch2Res.isOk());
    EXPECT_EQ(batch2Res.unwrap().size(), 2);

    // Try to get third batch (should fail)
    auto batch3Res = dataLoader.nextBatch();
    EXPECT_TRUE(batch3Res.isErr());
}

TEST_F(DataLoaderTest, ResetTest) {
    // Create data loader
    DataLoader<f32> dataLoader(dataset, 2, false, false);

    // Get first batch
    auto batch1Res = dataLoader.nextBatch();
    ASSERT_TRUE(batch1Res.isOk());

    // Reset the data loader
    dataLoader.reset();

    // Get first batch again (should be the same)
    auto batch1ResAgain = dataLoader.nextBatch();
    ASSERT_TRUE(batch1ResAgain.isOk());

    // Batches should have the same size
    auto batch1      = batch1Res.unwrap();
    auto batch1Again = batch1ResAgain.unwrap();

    ASSERT_EQ(batch1.size(), batch1Again.size());

    // Check that samples have the correct dimensions
    for (sizeT i = 0; i < batch1.size(); ++i) {
        EXPECT_EQ(batch1[i].featureDim(), 2);
        EXPECT_EQ(batch1[i].labelDim(), 1);
        EXPECT_EQ(batch1Again[i].featureDim(), 2);
        EXPECT_EQ(batch1Again[i].labelDim(), 1);
    }
}

TEST_F(DataLoaderTest, ShuffleTest) {
    // Create data loader with shuffle
    DataLoader<f32> dataLoader(dataset, 2, true, false);

    // Get batch
    auto batchRes = dataLoader.nextBatch();
    EXPECT_TRUE(batchRes.isOk());
}

TEST_F(DataLoaderTest, SetShuffleTest) {
    // Create data loader without shuffle
    DataLoader<f32> dataLoader(dataset, 2, false, false);

    // Enable shuffle
    dataLoader.setShuffle(true);

    // Get batch
    auto batchRes = dataLoader.nextBatch();
    EXPECT_TRUE(batchRes.isOk());

    // Disable shuffle
    dataLoader.setShuffle(false);

    // Get another batch
    auto batchRes2 = dataLoader.nextBatch();
    EXPECT_TRUE(batchRes2.isOk());
}

TEST_F(DataLoaderTest, SetDropLastTest) {
    // Create data loader
    DataLoader<f32> dataLoader(dataset, 4, false, false);

    // Change to drop last
    dataLoader.setDropLast(true);

    // Should now have only 1 batch
    EXPECT_EQ(dataLoader.numBatches(), 1);

    // Change back
    dataLoader.setDropLast(false);

    // Should now have 2 batches again
    EXPECT_EQ(dataLoader.numBatches(), 2);
}
