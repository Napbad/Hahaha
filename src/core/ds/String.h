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
// Created by Napbad on 7/27/25.
//

#ifndef STRING_H
#define STRING_H

// Standard Library
#include <algorithm>
#include <cstring>
#include <sstream>

// Project
#include "Res.h"
#include "defines/h3defs.h"

namespace hahaha::common
{
class IndexOutOfBoundError;
}

namespace hahaha::common::ds
{
class String
{
  public:
    static constexpr sizeT npos = static_cast<sizeT>(-1);

    String() : data_(new char[1]), size_(0), capacity_(0)
    {
        data_[0] = '\0';
    }

    explicit String(const char* c) : size_(std::strlen(c)), capacity_(size_)
    {
        data_ = new char[capacity_ + 1];
        std::strcpy(data_, c);
        data_[size_] = '\0';
    }

    // Copy constructor
    String(const String& other) : size_(other.size_), capacity_(other.capacity_)
    {
        data_ = new char[capacity_ + 1];
        std::strcpy(data_, other.data_);
        data_[size_] = '\0';
    }

    // Move constructor
    String(String&& other) noexcept : data_(other.data_), size_(other.size_), capacity_(other.capacity_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    String(const char* begin, const char* end) : size_(end - begin), capacity_(size_)
    {
        data_ = new char[capacity_ + 1];
        std::memcpy(data_, begin, size_);
        data_[size_] = '\0';
    }

    explicit String(const std::string& string) : size_(string.size()), capacity_(size_)
    {
        data_ = new char[capacity_ + 1];
        std::strcpy(data_, string.c_str());
        data_[size_] = '\0';
    }

    explicit String(const std::stringstream& ss) : size_(ss.str().size()), capacity_(size_)
    {
        data_ = new char[capacity_ + 1];
        std::strcpy(data_, ss.str().c_str());
        data_[size_] = '\0';
    }

    explicit String(const std::ostringstream& oss) : size_(oss.str().size()), capacity_(size_)
    {
        data_ = new char[capacity_ + 1];
        std::strcpy(data_, oss.str().c_str());
        data_[size_] = '\0';
    }

    ~String()
    {
        delete[] data_;
    }

    // Copy assignment operator
    String& operator=(const String& other)
    {
        if (this != &other)
        {
            if (capacity_ < other.size_)
            {
                delete[] data_;
                capacity_ = other.capacity_;
                data_ = new char[capacity_ + 1];
            }
            size_ = other.size_;
            std::strcpy(data_, other.data_);
            data_[size_] = '\0';
        }
        return *this;
    }

    // Move assignment operator
    String& operator=(String&& other) noexcept
    {
        if (this != &other)
        {
            delete[] data_;
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    [[nodiscard]] sizeT size() const
    {
        return size_;
    }

    [[nodiscard]] sizeT length() const
    {
        return size_;
    }

    [[nodiscard]] sizeT capacity() const
    {
        return capacity_;
    }

    [[nodiscard]] bool empty() const
    {
        return size_ == 0;
    }

    [[nodiscard]] Res<char, IndexOutOfBoundError> at(sizeT i) const;
    [[nodiscard]] char* begin()
    {
        return data_;
    }

    [[nodiscard]] const char* begin() const
    {
        return data_;
    }

    [[nodiscard]] char* end()
    {
        return data_ + size_;
    }

    [[nodiscard]] const char* end() const
    {
        return data_ + size_;
    }

    void clear()
    {
        size_ = 0;
        data_[0] = '\0';
    }

    void resize(const sizeT n, const char c = '\0')
    {
        if (n > capacity_)
        {
            reserve(n);
        }
        if (n > size_)
        {
            std::fill(data_ + size_, data_ + n, c);
        }
        size_ = n;
        data_[size_] = '\0';
    }

    [[nodiscard]] const char* cStr() const
    {
        return data_;
    }

    [[nodiscard]] const char* c_str() const
    {
        return data_;
    }

    [[nodiscard]] const char* data() const
    {
        return data_;
    }

    char& operator[](const sizeT i)
    {
        return data_[i];
    }

    const char& operator[](const sizeT i) const
    {
        return data_[i];
    }

    String& operator+=(const String& other)
    {
        reserve(size_ + other.size_);
        std::strcat(data_, other.data_);
        size_ += other.size_;
        data_[size_] = '\0';
        return *this;
    }

    String& operator+=(const char* s)
    {
        const sizeT len = std::strlen(s);
        reserve(size_ + len);
        std::strcat(data_, s);
        size_ += len;
        data_[size_] = '\0';
        return *this;
    }

    String& operator+=(const char c)
    {
        reserve(size_ + 1);
        data_[size_++] = c;
        data_[size_] = '\0';
        return *this;
    }

    bool operator==(const String& other) const
    {
        return size_ == other.size_ && std::strcmp(data_, other.data_) == 0;
    }

    bool operator!=(const String& other) const
    {
        return !(*this == other);
    }

    bool operator<(const String& other) const
    {
        return std::strcmp(data_, other.data_) < 0;
    }

    bool operator>(const String& other) const
    {
        return other < *this;
    }

    bool operator<=(const String& other) const
    {
        return !(*this > other);
    }

    bool operator>=(const String& other) const
    {
        return !(*this < other);
    }

    bool startsWith(const String& prefix) const
    {
        if (prefix.size() > this->size())
        {
            return false;
        }
        return std::strncmp(this->data_, prefix.data_, prefix.size()) == 0;
    }

    bool endsWith(const String& suffix) const
    {
        if (suffix.size() > this->size())
        {
            return false;
        }
        return std::strncmp(this->data_ + this->size() - suffix.size(), suffix.data_, suffix.size()) == 0;
    }

    void push_back(const char c)
    {
        *this += c;
    }

    String& insert(sizeT pos, const char c)
    {
        if (pos > size_)
        {
            pos = size_;
        }
        reserve(size_ + 1);
        std::memmove(data_ + pos + 1, data_ + pos, size_ - pos + 1);
        data_[pos] = c;
        size_++;
        return *this;
    }

    String& insert(sizeT pos, const String& o)
    {
        if (pos > size_)
        {
            pos = size_;
        }
        if (o.empty())
        {
            return *this;
        }

        reserve(size_ + o.size());
        // Move existing characters to make space
        std::memmove(data_ + pos + o.size(), data_ + pos, size_ - pos + 1); // +1 for null terminator
        std::ranges::copy(o, data_ + pos);
        size_ += o.size();
        return *this;
    }

    String& erase(const sizeT pos, const sizeT count = 1)
    {
        if (pos >= size_)
        {
            return *this;
        }
        const sizeT actual_count = std::min(count, size_ - pos);

        std::memmove(data_ + pos, data_ + pos + actual_count, size_ - pos - actual_count + 1); // +1 for null terminator
        size_ -= actual_count;
        return *this;
    }

    [[nodiscard]] sizeT find(const String& s, const size_t pos = 0) const
    {
        if (pos >= size_ || s.empty())
        {
            return npos;
        }

        if (const char* result = std::strstr(data_ + pos, s.data_))
        {
            return result - data_;
        }
        return npos;
    }

    void reserve(const sizeT n)
    {
        if (n >= capacity_)
        { // Use >= to handle resizing to current capacity if needed for null terminator
            const sizeT new_capacity = std::max(n, capacity_ == 0 ? 1 : capacity_ * 2);
            const auto new_data = new char[new_capacity + 1]; // +1 for null terminator

            if (data_)
            {
                std::strcpy(new_data, data_);
                delete[] data_;
            }
            data_ = new_data;
            capacity_ = new_capacity;
            data_[size_] = '\0'; // Ensure null termination
        }
    }
    void append(const char* contents, const sizeT size)
    {
        reserve(size_ + size);
        std::memcpy(data_ + size_, contents, size);
        size_ += size;
        data_[size_] = '\0';
    }
  private:
    char* data_;
    sizeT size_;
    sizeT capacity_;
};

inline String operator+(const String& lhs, const String& rhs)
{
    String result = lhs;
    result += rhs;
    return result;
}

inline String operator+(const String& lhs, const char* rhs)
{
    String result = lhs;
    result += rhs;
    return result;
}

inline String operator+(const char* lhs, const String& rhs)
{
    auto result = String(lhs);
    result += rhs;
    return result;
}

inline String operator+(const String& lhs, const char rhs)
{
    String result = lhs;
    result += rhs;
    return result;
}

inline String operator+(const char lhs, const String& rhs)
{
    String result;
    result += lhs;
    result += rhs;
    return result;
}

inline std::ostream& operator<<(std::ostream& os, const String& str)
{
    os << str.cStr();
    return os;
}

inline String operator+(const String& a, const std::string& b)
{
    return a + String(b);
}

inline String operator+(const std::string& a, const String& b)
{
    return String(a) + b;
}
} // namespace hahaha::common::ds

// Hash specialization for ds::String
template <> struct std::hash<hahaha::common::ds::String>
{
    size_t operator()(const hahaha::common::ds::String& s) const noexcept
    {
        size_t hash = 5381;
        const char* str = s.cStr();
        while (*str)
        {
            hash = (hash << 5) + hash + *str++;
        }
        return hash;
    }
}; // namespace std

#endif // STRING_H
