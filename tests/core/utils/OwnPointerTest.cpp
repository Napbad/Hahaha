// Copyright (c) 2025-2026 Contributors of Hahaha(https://github.com/jason-is-debugging/Hahaha)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Contributors:
// Napbad (napbad.sen@gmail.com) (https://github.com/Napbad)
//

#include "utils/OwnPointer.h"

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

using h3::core::utils::make_own_ptr;
using h3::core::utils::OwnPointer;  

class OwnPointerTest : public ::testing::Test {
  protected:
    void SetUp() override {}
    void TearDown() override {}
};

struct TestStruct {
    int value;
    std::string name;

    TestStruct() : value(0), name("default") {}
    explicit TestStruct(int v) : value(v), name("int_ctor") {}
    TestStruct(int v, const std::string& n) : value(v), name(n) {}
};

struct DerivedStruct : TestStruct {
    double extra;

    DerivedStruct() : TestStruct(), extra(0.0) {}
    explicit DerivedStruct(int v) : TestStruct(v), extra(1.0) {}
    DerivedStruct(int v, const std::string& n) : TestStruct(v, n), extra(2.0) {}
};

// ===== Default Constructor Tests =====

TEST_F(OwnPointerTest, DefaultConstructorIsInvalid) {
    OwnPointer<TestStruct> ptr;
    EXPECT_FALSE(ptr.isOwner());
    EXPECT_FALSE(ptr.is_valid());
}

TEST_F(OwnPointerTest, DefaultConstructorIsFalsy) {
    OwnPointer<TestStruct> ptr;
    EXPECT_FALSE(static_cast<bool>(ptr));
}

TEST_F(OwnPointerTest, NullptrConstructorIsInvalid) {
    OwnPointer<TestStruct> ptr(nullptr);
    EXPECT_FALSE(ptr.isOwner());
    EXPECT_FALSE(ptr.is_valid());
    EXPECT_FALSE(static_cast<bool>(ptr));
}

TEST_F(OwnPointerTest, NullptrConstructorEqualsDefault) {
    OwnPointer<TestStruct> ptr1;
    OwnPointer<TestStruct> ptr2(nullptr);
    EXPECT_FALSE(ptr1.isOwner());
    EXPECT_FALSE(ptr2.isOwner());
    EXPECT_EQ(ptr1.is_valid(), ptr2.is_valid());
}

// ===== make_own_ptr Tests =====

TEST_F(OwnPointerTest, MakeOwnPtrBasic) {
    auto ptr = make_own_ptr<TestStruct>(123);
    ASSERT_NE(ptr.get(), nullptr);
    EXPECT_TRUE(ptr.isOwner());
    EXPECT_TRUE(ptr.is_valid());
    EXPECT_EQ(ptr->value, 123);
}

TEST_F(OwnPointerTest, MakeOwnPtrMultipleArgs) {
    auto ptr = make_own_ptr<TestStruct>(456, "test_name");
    ASSERT_NE(ptr.get(), nullptr);
    EXPECT_EQ(ptr->value, 456);
    EXPECT_EQ(ptr->name, "test_name");
}

TEST_F(OwnPointerTest, MakeOwnPtrDefaultCtor) {
    auto ptr = make_own_ptr<TestStruct>();
    ASSERT_NE(ptr.get(), nullptr);
    EXPECT_EQ(ptr->value, 0);
    EXPECT_EQ(ptr->name, "default");
}

TEST_F(OwnPointerTest, MakeOwnPtrCreatesOwner) {
    auto ptr = make_own_ptr<TestStruct>(1);
    EXPECT_TRUE(ptr.isOwner());
}

// ===== isOwner / isOwner Tests =====

TEST_F(OwnPointerTest, IsOwnerAfterConstruction) {
    auto ptr = make_own_ptr<TestStruct>(1);
    EXPECT_TRUE(ptr.isOwner());
}

TEST_F(OwnPointerTest, IsOwnerFalseForBorrower) {
    auto owner = make_own_ptr<TestStruct>(1);
    auto borrower = owner.borrow();
    EXPECT_TRUE(owner.isOwner());
    EXPECT_FALSE(borrower.isOwner());
}

TEST_F(OwnPointerTest, IsOwnerAfterMove) {
    auto owner = make_own_ptr<TestStruct>(1);
    EXPECT_TRUE(owner.isOwner());
    auto moved = owner.move();
    EXPECT_FALSE(owner.isOwner());
    EXPECT_TRUE(moved.isOwner());
}

TEST_F(OwnPointerTest, IsOwnerAfterCopyAssign) {
    auto owner = make_own_ptr<TestStruct>(1);
    OwnPointer<TestStruct> borrower;
    borrower = owner;
    EXPECT_TRUE(owner.isOwner());
    EXPECT_FALSE(borrower.isOwner());
}

TEST_F(OwnPointerTest, IsOwnerAfterMoveAssign) {
    auto owner = make_own_ptr<TestStruct>(1);
    OwnPointer<TestStruct> moved;
    moved = std::move(owner);
    EXPECT_FALSE(owner.isOwner());
    EXPECT_TRUE(moved.isOwner());
}

// ===== borrow() Tests =====

TEST_F(OwnPointerTest, BorrowCreatesBorrower) {
    auto owner = make_own_ptr<TestStruct>(100);
    auto borrower = owner.borrow();
    EXPECT_TRUE(owner.isOwner());
    EXPECT_FALSE(borrower.isOwner());
    EXPECT_EQ(owner.get(), borrower.get());
}

TEST_F(OwnPointerTest, BorrowFromDefaultIsInvalid) {
    OwnPointer<TestStruct> defaultPtr;
    auto borrower = defaultPtr.borrow();
    EXPECT_FALSE(borrower.isOwner());
    EXPECT_FALSE(borrower.is_valid());
}

TEST_F(OwnPointerTest, BorrowedPointerSeesOwnerChanges) {
    auto owner = make_own_ptr<TestStruct>(200);
    auto borrower = owner.borrow();
    EXPECT_TRUE(borrower.is_valid());
    owner.reset();
    EXPECT_FALSE(borrower.is_valid());
}

TEST_F(OwnPointerTest, MultipleBorrowers) {
    auto owner = make_own_ptr<TestStruct>(50);
    auto b1 = owner.borrow();
    auto b2 = owner.borrow();
    auto b3 = owner.borrow();

    EXPECT_TRUE(owner.isOwner());
    EXPECT_FALSE(b1.isOwner());
    EXPECT_FALSE(b2.isOwner());
    EXPECT_FALSE(b3.isOwner());

    EXPECT_EQ(owner.get(), b1.get());
    EXPECT_EQ(b1.get(), b2.get());
    EXPECT_EQ(b2.get(), b3.get());
}

// ===== move() Tests =====

TEST_F(OwnPointerTest, MoveOwnerSucceeds) {
    auto owner = make_own_ptr<TestStruct>(999);
    auto moved = owner.move();

    EXPECT_FALSE(owner.isOwner());
    EXPECT_TRUE(moved.isOwner());
    EXPECT_NE(moved.get(), nullptr);
    EXPECT_FALSE(owner.is_valid());
    EXPECT_TRUE(moved.is_valid());
}

TEST_F(OwnPointerTest, MovePreservesOwnershipSemantics) {
    auto owner = make_own_ptr<TestStruct>(111);
    auto moved = owner.move();
    auto borrower = moved.borrow();

    EXPECT_FALSE(owner.isOwner());
    EXPECT_TRUE(moved.isOwner());
    EXPECT_FALSE(borrower.isOwner());

    EXPECT_EQ(moved.get(), borrower.get());
}

TEST_F(OwnPointerTest, MoveFromBorrowerThrows) {
    auto owner = make_own_ptr<TestStruct>(1);
    auto borrower = owner.borrow();
    EXPECT_THROW(borrower.move(), std::runtime_error);
}

TEST_F(OwnPointerTest, MoveFromInvalidBorrowerThrows) {
    auto owner = make_own_ptr<TestStruct>(1);
    auto borrower = owner.borrow();
    owner.reset();
    EXPECT_THROW(borrower.move(), std::runtime_error);
}

TEST_F(OwnPointerTest, MoveFromDefaultThrows) {
    OwnPointer<TestStruct> defaultPtr;
    EXPECT_THROW(defaultPtr.move(), std::runtime_error);
}

// ===== Copy Constructor Tests =====

TEST_F(OwnPointerTest, CopyConstructorCreatesBorrower) {
    auto owner = make_own_ptr<TestStruct>(777);
    OwnPointer<TestStruct> copy(owner);

    EXPECT_TRUE(owner.isOwner());
    EXPECT_FALSE(copy.isOwner());
    EXPECT_EQ(owner.get(), copy.get());
    EXPECT_TRUE(copy.is_valid());
}

TEST_F(OwnPointerTest, CopyConstructorFromDefault) {
    OwnPointer<TestStruct> defaultPtr;
    OwnPointer<TestStruct> copy(defaultPtr);

    EXPECT_FALSE(copy.isOwner());
    EXPECT_FALSE(copy.is_valid());
}

TEST_F(OwnPointerTest, CopyConstructorFromBorrower) {
    auto owner = make_own_ptr<TestStruct>(888);
    auto borrower = owner.borrow();
    OwnPointer<TestStruct> copy(borrower);

    EXPECT_TRUE(owner.isOwner());
    EXPECT_FALSE(borrower.isOwner());
    EXPECT_FALSE(copy.isOwner());
}

// ===== Copy Assignment Tests =====

TEST_F(OwnPointerTest, CopyAssignmentFromOwner) {
    auto owner = make_own_ptr<TestStruct>(222);
    OwnPointer<TestStruct> assignee;
    assignee = owner;

    EXPECT_TRUE(owner.isOwner());
    EXPECT_FALSE(assignee.isOwner());
    EXPECT_EQ(owner.get(), assignee.get());
}

TEST_F(OwnPointerTest, CopyAssignmentToBorrower) {
    auto owner1 = make_own_ptr<TestStruct>(1);
    auto owner2 = make_own_ptr<TestStruct>(2);
    auto borrower1 = owner1.borrow();

    borrower1 = owner2;

    EXPECT_TRUE(owner2.isOwner());
    EXPECT_FALSE(borrower1.isOwner());
    EXPECT_EQ(borrower1.get(), owner2.get());
}

TEST_F(OwnPointerTest, CopyAssignmentSelfAssignment) {
    auto owner = make_own_ptr<TestStruct>(333);
    owner = owner;

    EXPECT_TRUE(owner.isOwner());
    EXPECT_TRUE(owner.is_valid());
}

TEST_F(OwnPointerTest, CopyAssignmentFromDefault) {
    auto owner = make_own_ptr<TestStruct>(444);
    OwnPointer<TestStruct> assignee;
    assignee = owner;
    assignee = OwnPointer<TestStruct>();

    EXPECT_FALSE(assignee.isOwner());
    EXPECT_FALSE(assignee.is_valid());
}

TEST_F(OwnPointerTest, CopyAssignmentToExistingOwner) {
    auto owner1 = make_own_ptr<TestStruct>(100);
    auto owner2 = make_own_ptr<TestStruct>(200);

    auto borrower1 = owner1.borrow();
    EXPECT_TRUE(borrower1.is_valid());

    borrower1 = owner2;

    EXPECT_TRUE(owner2.isOwner());
    EXPECT_FALSE(borrower1.isOwner());
    EXPECT_EQ(borrower1.get(), owner2.get());
}

// ===== Move Constructor Tests =====

TEST_F(OwnPointerTest, MoveConstructorOwner) {
    auto owner = make_own_ptr<TestStruct>(555);
    OwnPointer<TestStruct> moved(std::move(owner));

    EXPECT_FALSE(owner.isOwner());
    EXPECT_TRUE(moved.isOwner());
    EXPECT_FALSE(owner.is_valid());
}

TEST_F(OwnPointerTest, MoveConstructorBorrower) {
    auto owner = make_own_ptr<TestStruct>(666);
    auto borrower = owner.borrow();
    OwnPointer<TestStruct> moved(std::move(borrower));

    EXPECT_FALSE(moved.isOwner());
    EXPECT_EQ(moved.get(), owner.get());
}

TEST_F(OwnPointerTest, MoveConstructorFromDefault) {
    OwnPointer<TestStruct> defaultPtr;
    OwnPointer<TestStruct> moved(std::move(defaultPtr));

    EXPECT_FALSE(moved.isOwner());
    EXPECT_FALSE(moved.is_valid());
}

// ===== Move Assignment Tests =====

TEST_F(OwnPointerTest, MoveAssignmentFromOwner) {
    auto owner = make_own_ptr<TestStruct>(888);
    OwnPointer<TestStruct> assignee;
    assignee = std::move(owner);

    EXPECT_FALSE(owner.isOwner());
    EXPECT_TRUE(assignee.isOwner());
    EXPECT_FALSE(owner.is_valid());
}

TEST_F(OwnPointerTest, MoveAssignmentFromBorrower) {
    auto owner = make_own_ptr<TestStruct>(999);
    auto borrower = owner.borrow();
    OwnPointer<TestStruct> moved;
    moved = std::move(borrower);

    EXPECT_FALSE(moved.isOwner());
    EXPECT_EQ(moved.get(), owner.get());
}

TEST_F(OwnPointerTest, MoveAssignmentSelfAssignment) {
    auto owner = make_own_ptr<TestStruct>(111);
    owner = std::move(owner);

    EXPECT_TRUE(owner.isOwner());
    EXPECT_TRUE(owner.is_valid());
}

TEST_F(OwnPointerTest, MoveAssignmentOverExistingOwner) {
    auto owner1 = make_own_ptr<TestStruct>(10);
    auto owner2 = make_own_ptr<TestStruct>(20);

    auto b1 = owner1.borrow();
    EXPECT_TRUE(b1.is_valid());

    b1 = std::move(owner2);

    EXPECT_TRUE(owner2.isOwner());
    EXPECT_TRUE(b1.isOwner());
    EXPECT_EQ(b1.get(), owner2.get());
}

TEST_F(OwnPointerTest, MoveAssignmentOverExistingBorrower) {
    auto owner1 = make_own_ptr<TestStruct>(10);
    auto owner2 = make_own_ptr<TestStruct>(20);

    auto b1 = owner1.borrow();
    OwnPointer<TestStruct> existingBorrower = owner1.borrow();

    b1 = std::move(owner2);

    EXPECT_TRUE(owner2.isOwner());
    EXPECT_TRUE(b1.isOwner());
    EXPECT_EQ(b1.get(), owner2.get());
}

// ===== Upcasting Move Constructor Tests =====

TEST_F(OwnPointerTest, UpcastMoveConstructor) {
    auto derivedOwner = make_own_ptr<DerivedStruct>(123, "derived");
    OwnPointer<TestStruct> base(std::move(derivedOwner));

    EXPECT_FALSE(derivedOwner.isOwner());
    EXPECT_TRUE(base.isOwner());
    EXPECT_EQ(base->value, 123);
}

TEST_F(OwnPointerTest, UpcastMoveConstructorFromBorrower) {
    auto derivedOwner = make_own_ptr<DerivedStruct>(456, "derived_borrow");
    auto derivedBorrower = derivedOwner.borrow();
    OwnPointer<TestStruct> base(std::move(derivedBorrower));

    EXPECT_FALSE(base.isOwner());
    EXPECT_EQ(base.get(), derivedOwner.get());
}

TEST_F(OwnPointerTest, UpcastCopyConstructor) {
    auto derivedOwner = make_own_ptr<DerivedStruct>(100, "derived");
    OwnPointer<TestStruct> baseOwner(derivedOwner);

    EXPECT_TRUE(derivedOwner.isOwner());
    EXPECT_FALSE(baseOwner.isOwner());
    EXPECT_EQ(baseOwner.get(), derivedOwner.get());
    EXPECT_EQ(baseOwner->value, 100);
}

TEST_F(OwnPointerTest, UpcastBorrow) {
    auto derivedOwner = make_own_ptr<DerivedStruct>(200, "upcast_borrow");
    auto baseBorrower = derivedOwner.borrow();

    EXPECT_TRUE(derivedOwner.isOwner());
    EXPECT_FALSE(baseBorrower.isOwner());
    EXPECT_EQ(baseBorrower.get(), derivedOwner.get());
}

// ===== reset() Tests =====

TEST_F(OwnPointerTest, ResetOwnerDeletesResource) {
    auto owner = make_own_ptr<TestStruct>(321);

    owner.reset();

    EXPECT_FALSE(owner.isOwner());
    EXPECT_FALSE(owner.is_valid());
}

TEST_F(OwnPointerTest, ResetBorrowerDoesNotDelete) {
    auto owner = make_own_ptr<TestStruct>(333);
    auto borrower = owner.borrow();

    borrower.reset();

    EXPECT_TRUE(owner.isOwner());
    EXPECT_TRUE(owner.is_valid());
    EXPECT_TRUE(borrower.isOwner());
    EXPECT_TRUE(borrower.is_valid());
}

TEST_F(OwnPointerTest, ResetDefaultPointer) {
    OwnPointer<TestStruct> defaultPtr;
    defaultPtr.reset();

    EXPECT_FALSE(defaultPtr.isOwner());
    EXPECT_FALSE(defaultPtr.is_valid());
}

TEST_F(OwnPointerTest, ResetAfterMoveTransfersOwnership) {
    auto owner = make_own_ptr<TestStruct>(777);
    auto moved = owner.move();

    owner.reset();

    EXPECT_FALSE(owner.isOwner());
    EXPECT_FALSE(owner.is_valid());
    EXPECT_TRUE(moved.isOwner());
    EXPECT_TRUE(moved.is_valid());
    EXPECT_EQ(moved->value, 777);
}

TEST_F(OwnPointerTest, ResetBorrowerFromOwner) {
    auto owner1 = make_own_ptr<TestStruct>(1);
    auto owner2 = make_own_ptr<TestStruct>(2);
    auto borrower = owner1.borrow();

    borrower = owner2;

    EXPECT_TRUE(owner2.isOwner());
    EXPECT_FALSE(borrower.isOwner());
    EXPECT_EQ(borrower.get(), owner2.get());
}

TEST_F(OwnPointerTest, ResetBorrowerBecomesOwner) {
    auto owner = make_own_ptr<TestStruct>(1);
    auto borrower = owner.borrow();

    borrower.reset();

    EXPECT_TRUE(owner.isOwner());
    EXPECT_TRUE(borrower.isOwner());
    EXPECT_EQ(owner.get(), borrower.get());
}

// ===== is_valid() Tests =====

TEST_F(OwnPointerTest, IsValidForValidOwner) {
    auto owner = make_own_ptr<TestStruct>(1);
    EXPECT_TRUE(owner.is_valid());
}

TEST_F(OwnPointerTest, IsValidForValidBorrower) {
    auto owner = make_own_ptr<TestStruct>(1);
    auto borrower = owner.borrow();
    EXPECT_TRUE(borrower.is_valid());
}

TEST_F(OwnPointerTest, IsValidFalseForDefault) {
    OwnPointer<TestStruct> ptr;
    EXPECT_FALSE(ptr.is_valid());
}

TEST_F(OwnPointerTest, IsValidFalseAfterOwnerReset) {
    auto owner = make_own_ptr<TestStruct>(1);
    auto borrower = owner.borrow();
    owner.reset();
    EXPECT_FALSE(borrower.is_valid());
}

TEST_F(OwnPointerTest, IsValidFalseAfterOwnerDeleted) {
    auto owner = make_own_ptr<TestStruct>(1);
    auto borrower = owner.borrow();
    owner.reset();
    EXPECT_FALSE(borrower.is_valid());
}

// ===== operator* Tests =====

TEST_F(OwnPointerTest, DereferenceOwner) {
    auto owner = make_own_ptr<TestStruct>(777);
    EXPECT_EQ((*owner).value, 777);
    EXPECT_EQ((*owner).name, "int_ctor");
}

TEST_F(OwnPointerTest, DereferenceBorrower) {
    auto owner = make_own_ptr<TestStruct>(888, "borrow_test");
    auto borrower = owner.borrow();
    EXPECT_EQ((*borrower).value, 888);
    EXPECT_EQ((*borrower).name, "borrow_test");
}

TEST_F(OwnPointerTest, DereferenceInvalidThrows) {
    auto owner = make_own_ptr<TestStruct>(1);
    auto borrower = owner.borrow();
    owner.reset();

    EXPECT_THROW((*borrower), std::runtime_error);
}

TEST_F(OwnPointerTest, DereferenceDefaultThrows) {
    OwnPointer<TestStruct> ptr;
    EXPECT_THROW((*ptr), std::runtime_error);
}

// ===== operator-> Tests =====

TEST_F(OwnPointerTest, ArrowOperatorOwner) {
    auto owner = make_own_ptr<TestStruct>(555, "arrow_test");
    EXPECT_EQ(owner->value, 555);
    EXPECT_EQ(owner->name, "arrow_test");
}

TEST_F(OwnPointerTest, ArrowOperatorBorrower) {
    auto owner = make_own_ptr<TestStruct>(666, "borrow_arrow");
    auto borrower = owner.borrow();
    EXPECT_EQ(borrower->value, 666);
    EXPECT_EQ(borrower->name, "borrow_arrow");
}

TEST_F(OwnPointerTest, ArrowOperatorInvalidThrows) {
    auto owner = make_own_ptr<TestStruct>(1);
    auto borrower = owner.borrow();
    owner.reset();

    EXPECT_THROW(borrower->value, std::runtime_error);
}

TEST_F(OwnPointerTest, ArrowOperatorDefaultThrows) {
    OwnPointer<TestStruct> ptr;
    EXPECT_THROW(ptr->value, std::runtime_error);
}

// ===== operator bool Tests =====

TEST_F(OwnPointerTest, OperatorBoolTrue) {
    auto owner = make_own_ptr<TestStruct>(1);
    if (owner) {
        SUCCEED();
    } else {
        FAIL() << "Owner should be truthy";
    }
}

TEST_F(OwnPointerTest, OperatorBoolFalse) {
    OwnPointer<TestStruct> ptr;
    if (ptr) {
        FAIL() << "Default pointer should be falsy";
    } else {
        SUCCEED();
    }
}

TEST_F(OwnPointerTest, OperatorBoolFalseAfterInvalidation) {
    auto owner = make_own_ptr<TestStruct>(1);
    auto borrower = owner.borrow();
    owner.reset();

    EXPECT_FALSE(static_cast<bool>(borrower));
}

// ===== Destructor Tests =====

TEST_F(OwnPointerTest, DestructorDeletesOwnerResource) {
    bool deleted = false;
    struct DeletableStruct {
        bool& deletedRef;
        explicit DeletableStruct(bool& ref) : deletedRef(ref) {}
        ~DeletableStruct() { deletedRef = true; }
    };

    {
        auto owner = make_own_ptr<DeletableStruct>(deleted);
        auto borrower = owner.borrow();
        EXPECT_TRUE(borrower.is_valid());
    }
    EXPECT_TRUE(deleted);
}

TEST_F(OwnPointerTest, DestructorOfMovedFromOwner) {
    auto owner = make_own_ptr<TestStruct>(123);
    auto moved = owner.move();

    EXPECT_FALSE(owner.is_valid());
    EXPECT_TRUE(moved.is_valid());
}

TEST_F(OwnPointerTest, DestructorOfBorrowerDoesNotDeleteResource) {
    bool deleted = false;
    struct DeletableStruct {
        bool& deletedRef;
        explicit DeletableStruct(bool& ref) : deletedRef(ref) {}
        ~DeletableStruct() { deletedRef = true; }
    };

    bool ownerDeleted = false;
    auto owner = make_own_ptr<DeletableStruct>(ownerDeleted);
    auto borrower = owner.borrow();

    borrower.reset();
    EXPECT_TRUE(ownerDeleted);
}

// ===== Interaction Tests =====

TEST_F(OwnPointerTest, ChainOfBorrowers) {
    auto owner = make_own_ptr<TestStruct>(100);
    auto b1 = owner.borrow();
    auto b2 = b1.borrow();
    auto b3 = b2.borrow();

    EXPECT_TRUE(owner.isOwner());
    EXPECT_FALSE(b1.isOwner());
    EXPECT_FALSE(b2.isOwner());
    EXPECT_FALSE(b3.isOwner());

    EXPECT_EQ(owner.get(), b1.get());
    EXPECT_EQ(b1.get(), b2.get());
    EXPECT_EQ(b2.get(), b3.get());

    owner.reset();
    EXPECT_FALSE(b3.is_valid());
}

TEST_F(OwnPointerTest, ComplexOwnershipTransfer) {
    auto owner1 = make_own_ptr<TestStruct>(1);
    auto owner2 = make_own_ptr<TestStruct>(2);

    auto b1 = owner1.borrow();
    auto b2 = owner2.borrow();

    b1 = owner2.borrow();
    EXPECT_EQ(b1.get(), owner2.get());

    auto moved = owner1.move();
    EXPECT_FALSE(owner1.isOwner());
    EXPECT_TRUE(moved.isOwner());
}

TEST_F(OwnPointerTest, ResetBorrowerInvalidatesOtherBorrowers) {
    auto owner = make_own_ptr<TestStruct>(1);
    auto b1 = owner.borrow();
    auto b2 = owner.borrow();

    b1.reset();

    EXPECT_TRUE(b1.isOwner());
    EXPECT_TRUE(b1.is_valid());
    EXPECT_TRUE(owner.isOwner());
    EXPECT_TRUE(owner.is_valid());
    EXPECT_TRUE(b2.isOwner());
    EXPECT_TRUE(b2.is_valid());

    EXPECT_EQ(b1.get(), owner.get());
    EXPECT_EQ(b2.get(), owner.get());
}

TEST_F(OwnPointerTest, CopyAssignDefaultToDefault) {
    OwnPointer<TestStruct> ptr1;
    OwnPointer<TestStruct> ptr2;

    ptr1 = ptr2;

    EXPECT_FALSE(ptr1.isOwner());
    EXPECT_FALSE(ptr1.is_valid());
}

TEST_F(OwnPointerTest, MoveAssignDefaultToDefault) {
    OwnPointer<TestStruct> ptr1;
    OwnPointer<TestStruct> ptr2;

    ptr1 = std::move(ptr2);

    EXPECT_FALSE(ptr1.isOwner());
    EXPECT_FALSE(ptr1.is_valid());
}

TEST_F(OwnPointerTest, BorrowedPointsToSameResource) {
    auto owner = make_own_ptr<TestStruct>(999);
    auto b1 = owner.borrow();
    auto b2 = owner.borrow();

    EXPECT_EQ(&(*owner), &(*b1));
    EXPECT_EQ(&(*b1), &(*b2));
}

TEST_F(OwnPointerTest, CopyAssignOverwritesBorrowerStatus) {
    auto owner = make_own_ptr<TestStruct>(100);
    auto borrower = owner.borrow();

    borrower = owner;

    EXPECT_FALSE(borrower.isOwner());
    EXPECT_EQ(borrower.get(), owner.get());
}
