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
// Created by Napbad on 8/3/25.
//

#ifndef ALLOCATOR_H
#define ALLOCATOR_H
#include <limits>
#include <new>
#include <utility>

#include "common/defines/h3defs.h"

namespace hahaha::common::util
{

template <class T> class Allocator
{
  public:
    using valueTp = T;
    using pointer = T*;
    using constPointer = const T*;
    using reference = T&;
    using constReference = const T&;
    using sizeTp = sizeT;
    using differenceTp = ptrDiffT;

    template <typename U> struct rebind
    {
        using other = Allocator<U>;
    };

    Allocator() noexcept = default;

    Allocator(const Allocator&) noexcept = default;

    template <typename U> explicit Allocator(const Allocator<U>&) noexcept
    {
    }

    ~Allocator() noexcept = default;

    pointer allocate(const sizeTp n)
    {
        if (n == 0)
        {
            return nullptr;
        }

        if (n > maxSize())
        {
            throw std::bad_alloc();
        }

        auto ptr = static_cast<pointer>(::operator new(n * sizeof(T)));

        if (!ptr)
        {
            throw std::bad_alloc();
        }

        return ptr;
    }

    void deallocate(constPointer p, sizeTp)
    {
        if (p)
        {
            ::operator delete(p);
        }
    }

    template <typename U, typename... Args> void construct(U* ptr, Args&&... args)
    {
        ::new (static_cast<void*>(ptr)) U(std::forward<Args>(args)...);
    }

    template <typename U> void destroy(U* ptr) noexcept
    {
        ptr->~U();
    }

    [[nodiscard]] static sizeTp maxSize() noexcept
    {
        return std::numeric_limits<sizeT>::max();
    }

    bool operator==(const Allocator&) const noexcept
    {
        return true;
    }

    bool operator!=(const Allocator& other) const noexcept
    {
        return !(*this == other);
    }
};
} // namespace hahaha::common::util

template <class T> class Allocator
{
  public:
    using valueTp = T;
    using pointer = T*;
    using constPointer = const T*;
    using reference = T&;
    using constReference = const T&;
    using sizeTp = sizeT;
    using differenceTp = ptrDiffT;

    pointer allocate(const sizeTp n)
    {
    }

    template <typename U, typename... Args> void construct(U* ptr, Args&&... args)
    {
        ::new (static_cast<void*>(ptr)) U(std::forward<Args>(args)...);
    }
};
#endif // ALLOCATOR_H
