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
// Created by Napbad on 7/26/25.
//

#ifndef VEC_H
#define VEC_H
#include <algorithm>
#include <gtest/gtest-assertion-result.h>
#include <initializer_list>
#include <memory>
#include <sstream>

#include "defines/h3defs.h"
#include "ds/String.h"

namespace hahaha::ml
{
template<typename T>
    class Tensor;
}

namespace hahaha::common::ds
{
template <class T, class Allocator = std::allocator<T>> class Vector
{

    friend class ml::Tensor<T>;

  public:
    using ValueType = T;
    using Reference = T&;
    using ConstReference = const T&;
    using Pointer = T*;
    using ConstPointer = const T*;
    using Iterator = T*;
    using ConstIterator = const T*;
    using SizeType = sizeT;
    using DifferenceType = ptrDiffT;

    Allocator allocator_;

    Vector() noexcept : data_(nullptr), size_(0), capacity_(0)
    {
    }

    explicit Vector(const SizeType count) : data_(nullptr), size_(0), capacity_(0)
    {
        if (count > 0)
        {
            reserve(count);
            for (sizeT i = 0; i < count; ++i)
            {
                std::allocator_traits<Allocator>::construct(allocator_, data_ + i);
            }
            capacity_ = count;
        }
    }

    Vector(std::initializer_list<T> init) : data_(nullptr), size_(0), capacity_(0)
    {
        reserve(init.size());
        size_ = init.size();
        sizeT idx = 0;
        for (const auto& i : init)
        {
            std::allocator_traits<Allocator>::construct(allocator_, data_ + idx, i);
            idx++;
        }
    }

    explicit Vector(Vector<T>::ConstIterator begin, Vector<T>::ConstIterator end) : data_(nullptr), size_(0), capacity_(0)
    {
        size_ = end - begin;
        capacity_ = size_;
        if (size_ > 0)
        {
            data_ = allocator_.allocate(capacity_);
            for (auto i = begin; i < end; ++i)
            {
                std::allocator_traits<Allocator>::construct(allocator_, data_ + (i - begin), *i);
            }
        }
    }

    // Copy constructor
    Vector(const Vector& other) : size_(other.size_), capacity_(other.capacity_)
    {
        if (capacity_ > 0)
        {
            data_ = allocator_.allocate(capacity_);
            for (sizeT i = 0; i < size_; ++i)
            {
                std::allocator_traits<Allocator>::construct(allocator_, data_ + i, other.data_[i]);
            }
        }
    }

    // Move constructor
    Vector(Vector&& other) noexcept : data_(other.data_), size_(other.size_), capacity_(other.capacity_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    ~Vector()
    {
        for (sizeT i = 0; i < size_; ++i)
        {
            std::allocator_traits<Allocator>::destroy(allocator_, data_ + i);
        }
        allocator_.deallocate(data_, capacity_);
        data_ = nullptr;
    }

    // Copy assignment operator
    Vector& operator=(const Vector& other)
    {
        if (this != &other)
        {
            // Destroy existing elements
            for (sizeT i = 0; i < size_; ++i)
            {
                std::allocator_traits<Allocator>::destroy(allocator_, data_ + i);
            }
            // Deallocate old memory if needed
            if (capacity_ < other.size_)
            {
                allocator_.deallocate(data_, capacity_);
                capacity_ = other.capacity_;
                data_ = allocator_.allocate(capacity_);
            }
            // Copy new elements
            size_ = other.size_;
            for (sizeT i = 0; i < size_; ++i)
            {
                std::allocator_traits<Allocator>::construct(allocator_, data_ + i, other.data_[i]);
            }
        }
        return *this;
    }

    // Move assignment operator
    Vector& operator=(Vector&& other) noexcept
    {
        if (this != &other)
        {
            // Destroy existing elements
            for (sizeT i = 0; i < size_; ++i)
            {
                std::allocator_traits<Allocator>::destroy(allocator_, data_ + i);
            }
            allocator_.deallocate(data_, capacity_);

            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    void emplaceBack(const T& value)
    {
        if (size_ == capacity_)
        {
            reserve(size_ == 0 ? 1 : size_ * 2);
        }
        std::allocator_traits<Allocator>::construct(allocator_, data_ + size_, value);
        size_++;
    }

    void pushBack(const T& value)
    {
        emplaceBack(value);
    }

    Iterator begin()
    {
        return data_;
    }

    ConstIterator begin() const
    {
        return data_;
    }

    Iterator end()
    {
        return data_ + size_;
    }

    ConstIterator end() const
    {
        return data_ + size_;
    }

    bool operator==(const Vector& vec) const
    {
        if (size_ != vec.size_)
        {
            return false;
        }

        for (sizeT i = 0; i < size_; i++)
        {
            if (!(data_[i] == vec.data_[i]))
            { // Use ! for operator==
                return false;
            }
        }

        return true;
    }

    [[nodiscard]] sizeT size() const
    {
        return size_;
    }

    [[nodiscard]] sizeT capacity() const
    {
        return capacity_;
    }

    T& operator[](const SizeType index)
    {
        // No bounds checking for performance, like std::vector
        return data_[index];
    }

    const T& operator[](const SizeType index) const
    {
        // No bounds checking for performance, like std::vector
        return data_[index];
    }

    void clear()
    {
        for (sizeT i = 0; i < size_; ++i)
        {
            std::allocator_traits<Allocator>::destroy(allocator_, data_ + i);
        }
        size_ = 0;
    }

    void reserve(const SizeType newCap)
    {
        if (newCap > capacity_)
        {
            T* new_data = allocator_.allocate(newCap);

            // Move existing elements to new storage
            for (sizeT i = 0; i < size_; ++i)
            {
                std::allocator_traits<Allocator>::construct(allocator_, new_data + i, std::move(data_[i]));
                std::allocator_traits<Allocator>::destroy(allocator_, data_ + i);
            }

            allocator_.deallocate(data_, capacity_);
            data_ = new_data;
            capacity_ = newCap;
        }
    }

    void shrinkToFit()
    {
        if (capacity_ > size_)
        {
            T* new_data = nullptr;
            if (size_ > 0)
            {
                new_data = allocator_.allocate(size_);
                for (sizeT i = 0; i < size_; ++i)
                {
                    std::allocator_traits<Allocator>::construct(allocator_, new_data + i, std::move(data_[i]));
                    std::allocator_traits<Allocator>::destroy(allocator_, data_ + i);
                }
            }
            allocator_.deallocate(data_, capacity_);
            data_ = new_data;
            capacity_ = size_;
        }
    }

    [[nodiscard]] bool empty() const
    {
        return size_ == 0;
    }

    Vector<T> subVector(const sizeT start, const sizeT length) const
    {
        if (start + length > size_)
        {
            // Handle error: sub-vector out of bounds
            throw std::out_of_range("subVector: sub-vector range out of bounds");
        }
        Vector<T> res(length);
        for (sizeT i = 0; i < length; ++i)
        {
            auto var = data_[start + i];
            res.pushBack(data_[start + i]);
        }
        return res;
    }

    T* data()
    {
        return data_;
    }

    const T* data() const
    {
        return data_;
    }

    void popBack()
    {
        if (size_ > 0)
        {
            std::allocator_traits<Allocator>::destroy(allocator_, data_ + size_ - 1);
            size_--;
        }
    }

    void resize(const SizeType newSize)
    {
        if (newSize > capacity_)
        {
            reserve(newSize);
        }
        if (newSize > size_)
        {
            for (sizeT i = size_; i < newSize; ++i)
            {
                std::allocator_traits<Allocator>::construct(allocator_, data_ + i);
            }
        }
        else if (newSize < size_)
        {
            for (sizeT i = newSize; i < size_; ++i)
            {
                std::allocator_traits<Allocator>::destroy(allocator_, data_ + i);
            }
        }
        size_ = newSize;
    }

    void swap(Vector& other)
    {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
        std::swap(capacity_, other.capacity_);
    }

    void insert(Iterator pos, const T& value)
    {
        sizeT index = pos - begin();
        if (size_ == capacity_)
        {
            reserve(size_ == 0 ? 1 : size_ * 2);
        }
        // Shift elements to the right
        for (sizeT i = size_; i > index; --i)
        {
            std::allocator_traits<Allocator>::construct(allocator_, data_ + i, std::move(data_[i - 1]));
            std::allocator_traits<Allocator>::destroy(allocator_, data_ + (i - 1));
        }
        std::allocator_traits<Allocator>::construct(allocator_, data_ + index, value);
        size_++;
    }

    Iterator erase(Iterator pos)
    {
        sizeT index = pos - begin();
        if (index >= size_)
        {
            return end();
        }
        // Shift elements to the left
        for (sizeT i = index; i < size_ - 1; ++i)
        {
            std::allocator_traits<Allocator>::construct(allocator_, data_ + i, std::move(data_[i + 1]));
            std::allocator_traits<Allocator>::destroy(allocator_, data_ + (i + 1));
        }
        std::allocator_traits<Allocator>::destroy(allocator_, data_ + (size_ - 1));
        size_--;
        return begin() + index;
    }

    Iterator erase(Iterator first, Iterator last)
    {
        sizeT startIdx = first - begin();
        const sizeT endIdx = last - begin();
        if (startIdx >= size_)
        {
            return end();
        }
        if (endIdx >= size_)
        {
            for (sizeT i = startIdx; i < size_; ++i)
            {
                std::allocator_traits<Allocator>::destroy(allocator_, data_ + i);
            }
            size_ = startIdx;
            return begin() + startIdx;
        }
        // Shift elements to the left
        for (sizeT i = startIdx; i < size_ - (endIdx - startIdx); ++i)
        {
            std::allocator_traits<Allocator>::construct(allocator_, data_ + i, std::move(data_[i + (endIdx - startIdx)]));
            std::allocator_traits<Allocator>::destroy(allocator_, data_ + (i + (endIdx - startIdx)));
        }
        size_ = size_ - (endIdx - startIdx);
        return begin() + startIdx;
    }

    [[nodiscard]] T& front()
    {
        if (size_ == 0)
        {
            throw std::out_of_range("Vector is empty");
        }
        return data_[0];
    }

    [[nodiscard]] const T& front() const
    {
        if (size_ == 0)
        {
            throw std::out_of_range("Vector is empty");
        }
        return data_[0];
    }

    [[nodiscard]] T& back()
    {
        if (size_ == 0)
        {
            throw std::out_of_range("Vector is empty");
        }
        return data_[size_ - 1];
    }

    [[nodiscard]] const T& back() const
    {
        if (size_ == 0)
        {
            throw std::out_of_range("Vector is empty");
        }
        return data_[size_ - 1];
    }

    [[nodiscard]] T& at(const SizeType index)
    {
        if (index >= size_)
        {
            throw std::out_of_range("Vector::at: index out of bounds");
        }
        return data_[index];
    }

    [[nodiscard]] const T& at(const SizeType index) const
    {
        if (index >= size_)
        {
            throw std::out_of_range("Vector::at: index out of bounds");
        }
        return data_[index];
    }

    [[nodiscard]] String toString() const
    {
        std::ostringstream oss;
        oss << "[";
        for (sizeT i = 0; i < size_; ++i)
        {
            if (i > 0)
            {
                oss << ", ";
            }
            oss << data_[i];
        }
        oss << "]";
        return String(oss.str());
    }

  private:
    T* data_ = nullptr;
    sizeT size_ = 0;
    sizeT capacity_ = 0;
};

template<class T, class Allocator>
inline std::ostream& operator<<(std::ostream& os, const Vector<T, Allocator>& vec)
{
    os << vec.toString().c_str();
    return os;
}
} // namespace hahaha::common::ds

#endif // VEC_H
