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

#ifndef DQUEUE_H
#define DQUEUE_H

#include <initializer_list>
#include <stdexcept>

#include "list.h"

namespace hahaha::core::ds
{

// Double-ended queue implementation using a doubly-linked list
template <typename T> class Deque
{
  private:
    struct Node
    {
        T data;
        Node* prev;
        Node* next;

        explicit Node(const T& val) : data(val), prev(nullptr), next(nullptr)
        {
        }
    };

  public:
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using size_type = size_t;

    // Constructors
    Deque() : head_(nullptr), tail_(nullptr), size_(0)
    {
    }

    Deque(std::initializer_list<T> init)
        : head_(nullptr), tail_(nullptr), size_(0)
    {
        for (const auto& val : init)
        {
            pushBack(val);
        }
    }

    // Destructor
    ~Deque()
    {
        clear();
    }

    // Copy constructor
    Deque(const Deque& other) : head_(nullptr), tail_(nullptr), size_(0)
    {
        Node* current = other.head_;
        while (current)
        {
            pushBack(current->data);
            current = current->next;
        }
    }

    // Copy assignment
    Deque& operator=(const Deque& other)
    {
        if (this != &other)
        {
            clear();
            Node* current = other.head_;
            while (current)
            {
                pushBack(current->data);
                current = current->next;
            }
        }
        return *this;
    }

    // Move constructor
    Deque(Deque&& other) noexcept
        : head_(other.head_), tail_(other.tail_), size_(other.size_)
    {
        other.head_ = nullptr;
        other.tail_ = nullptr;
        other.size_ = 0;
    }

    // Move assignment
    Deque& operator=(Deque&& other) noexcept
    {
        if (this != &other)
        {
            clear();
            head_ = other.head_;
            tail_ = other.tail_;
            size_ = other.size_;
            other.head_ = nullptr;
            other.tail_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // Capacity
    [[nodiscard]] bool empty() const noexcept
    {
        return size_ == 0;
    }

    [[nodiscard]] size_type size() const noexcept
    {
        return size_;
    }

    // Element access
    reference front()
    {
        if (empty())
            throw std::out_of_range("Deque::front: deque is empty");
        return head_->data;
    }

    const_reference front() const
    {
        if (empty())
            throw std::out_of_range("Deque::front: deque is empty");
        return head_->data;
    }

    reference back()
    {
        if (empty())
            throw std::out_of_range("Deque::back: deque is empty");
        return tail_->data;
    }

    const_reference back() const
    {
        if (empty())
            throw std::out_of_range("Deque::back: deque is empty");
        return tail_->data;
    }

    reference at(size_type index)
    {
        if (index >= size_)
            throw std::out_of_range("Deque::at: index out of range");
        return getNode(index)->data;
    }

    const_reference at(size_type index) const
    {
        if (index >= size_)
            throw std::out_of_range("Deque::at: index out of range");
        return getNode(index)->data;
    }

    reference operator[](size_type index)
    {
        return getNode(index)->data;
    }

    const_reference operator[](size_type index) const
    {
        return getNode(index)->data;
    }

    // Modifiers
    void pushFront(const T& value)
    {
        Node* newNode = new Node(value);
        if (empty())
        {
            head_ = tail_ = newNode;
        }
        else
        {
            newNode->next = head_;
            head_->prev = newNode;
            head_ = newNode;
        }
        ++size_;
    }

    void pushBack(const T& value)
    {
        Node* newNode = new Node(value);
        if (empty())
        {
            head_ = tail_ = newNode;
        }
        else
        {
            tail_->next = newNode;
            newNode->prev = tail_;
            tail_ = newNode;
        }
        ++size_;
    }

    void popFront()
    {
        if (empty())
            throw std::out_of_range("Deque::popFront: deque is empty");

        Node* oldHead = head_;
        head_ = head_->next;

        if (head_)
            head_->prev = nullptr;
        else
            tail_ = nullptr;

        delete oldHead;
        --size_;
    }

    void popBack()
    {
        if (empty())
            throw std::out_of_range("Deque::popBack: deque is empty");

        const Node* oldTail = tail_;
        tail_ = tail_->prev;

        if (tail_)
            tail_->next = nullptr;
        else
            head_ = nullptr;

        delete oldTail;
        --size_;
    }

    template <typename... Args> void emplaceFront(Args&&... args)
    {
        T value(std::forward<Args>(args)...);
        pushFront(value);
    }

    template <typename... Args> void emplaceBack(Args&&... args)
    {
        T value(std::forward<Args>(args)...);
        pushBack(value);
    }

    void clear() noexcept
    {
        while (head_)
        {
            Node* next = head_->next;
            delete head_;
            head_ = next;
        }
        tail_ = nullptr;
        size_ = 0;
    }

  private:
    Node* head_;
    Node* tail_;
    size_type size_;

    Node* getNode(size_type index) const
    {
        // Optimize by starting from the closer end
        if (index < size_ / 2)
        {
            Node* current = head_;
            for (size_type i = 0; i < index; ++i)
                current = current->next;
            return current;
        }
        else
        {
            Node* current = tail_;
            for (size_type i = size_ - 1; i > index; --i)
                current = current->prev;
            return current;
        }
    }
};

} // namespace hahaha::core::ds

#endif // DQUEUE_H
