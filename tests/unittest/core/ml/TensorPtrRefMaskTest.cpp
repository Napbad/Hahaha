#include <gtest/gtest.h>

#include "core/TensorPtr.h"

using namespace hahaha;
using namespace hahaha::core;

// Test fixture for TensorPtr ref/unref bit-mask semantics
class TensorPtrFixture : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        // Fresh pointer for each test
        ptr = std::make_unique<TensorPtr<float>>(std::initializer_list<sizeT>{2});
    }

    void TearDown() override
    {
        // Ensure pointer exists; explicit unref calls in tests should have
        // already released resources when both bits were set.
        ptr.reset();
    }

    std::unique_ptr<TensorPtr<float>> ptr;
};

TEST_F(TensorPtrFixture, VarBitSetAndCleared)
{
    // Initially no holders; setting VAR should succeed
    EXPECT_NO_THROW(ptr->refByVar());
    // Setting VAR again should throw
    EXPECT_THROW(ptr->refByVar(), std::runtime_error);
    // Clearing VAR should release; with only one bit set it should delete when cleared
    EXPECT_NO_THROW(ptr->unrefByVar());
}

TEST_F(TensorPtrFixture, NodeBitSetAndCleared)
{
    EXPECT_NO_THROW(ptr->refByNode());
    EXPECT_THROW(ptr->refByNode(), std::runtime_error);
    EXPECT_NO_THROW(ptr->unrefByNode());
}

TEST_F(TensorPtrFixture, BothBitsOrderDoesNotMatter)
{
    // Take both roles
    EXPECT_NO_THROW(ptr->refByVar());
    EXPECT_NO_THROW(ptr->refByNode());

    // Clearing one should not delete (still held by the other)
    EXPECT_NO_THROW(ptr->unrefByVar());

    // Clearing the last should delete without throwing
    EXPECT_NO_THROW(ptr->unrefByNode());
}

TEST_F(TensorPtrFixture, MoveDoesNotReleaseBits)
{
    ptr->refByVar();
    ptr->refByNode();

    // Move construct into a local object
    TensorPtr<float> q(std::move(*ptr));

    // Moved-from object remains valid but empty; unref on it should be no-op
    EXPECT_NO_THROW(ptr->unrefByVar());

    // q still carries both bits and must be able to unref both without double free
    EXPECT_NO_THROW(q.unrefByVar());
    EXPECT_NO_THROW(q.unrefByNode());
}
