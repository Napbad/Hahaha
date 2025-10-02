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

#ifndef STR_H
#define STR_H
#include "common/Res.h"
#include "common/defines/h3defs.h"
#include <algorithm>
#include <cstring>
#include <sstream>

namespace hahaha::common {
    class IndexOutOfBoundError;
}
namespace hahaha::common::ds {
    class Str {
    public:
        static constexpr sizeT npos = static_cast<sizeT>(-1);

        Str() : _data(new char[1]), _size(0), _capacity(0) {
            _data[0] = '\0';
        }

        explicit Str(const char* c) : _size(std::strlen(c)), _capacity(_size) {
            _data = new char[_capacity + 1];
            std::strcpy(_data, c);
            _data[_size] = '\0';
        }

        // Copy constructor
        Str(const Str& other) : _size(other._size), _capacity(other._capacity) {
            _data = new char[_capacity + 1];
            std::strcpy(_data, other._data);
            _data[_size] = '\0';
        }

        // Move constructor
        Str(Str&& other) noexcept : _data(other._data), _size(other._size), _capacity(other._capacity) {
            other._data     = nullptr;
            other._size     = 0;
            other._capacity = 0;
        }

        Str(const char* begin, const char* end) : _size(end - begin), _capacity(_size) {
            _data = new char[_capacity + 1];
            std::memcpy(_data, begin, _size);
            _data[_size] = '\0';
        }

        explicit Str(const std::string& string) : _size(string.size()), _capacity(_size) {
            _data = new char[_capacity + 1];
            std::strcpy(_data, string.c_str());
            _data[_size] = '\0';
        }

        explicit Str(const std::stringstream& ss) : _size(ss.str().size()), _capacity(_size) {
            _data = new char[_capacity + 1];
            std::strcpy(_data, ss.str().c_str());
            _data[_size] = '\0';
        }

        explicit Str(const std::ostringstream& oss) : _size(oss.str().size()), _capacity(_size) {
            _data = new char[_capacity + 1];
            std::strcpy(_data, oss.str().c_str());
            _data[_size] = '\0';
        }

        ~Str() {
            delete[] _data;
        }

        // Copy assignment operator
        Str& operator=(const Str& other) {
            if (this != &other) {
                if (_capacity < other._size) {
                    delete[] _data;
                    _capacity = other._capacity;
                    _data     = new char[_capacity + 1];
                }
                _size = other._size;
                std::strcpy(_data, other._data);
                _data[_size] = '\0';
            }
            return *this;
        }

        // Move assignment operator
        Str& operator=(Str&& other) noexcept {
            if (this != &other) {
                delete[] _data;
                _data           = other._data;
                _size           = other._size;
                _capacity       = other._capacity;
                other._data     = nullptr;
                other._size     = 0;
                other._capacity = 0;
            }
            return *this;
        }

        [[nodiscard]] sizeT size() const {
            return _size;
        }

        [[nodiscard]] sizeT length() const {
            return _size;
        }

        [[nodiscard]] sizeT capacity() const {
            return _capacity;
        }

        [[nodiscard]] bool empty() const {
            return _size == 0;
        }

        [[nodiscard]] Res<char, IndexOutOfBoundError> at(sizeT i) const;
        [[nodiscard]] char* begin() {
            return _data;
        }

        [[nodiscard]] const char* begin() const {
            return _data;
        }

        [[nodiscard]] char* end() {
            return _data + _size;
        }

        [[nodiscard]] const char* end() const {
            return _data + _size;
        }

        void clear() {
            _size    = 0;
            _data[0] = '\0';
        }

        void resize(const sizeT n, const char c = '\0') {
            if (n > _capacity) {
                reserve(n);
            }
            if (n > _size) {
                std::fill(_data + _size, _data + n, c);
            }
            _size        = n;
            _data[_size] = '\0';
        }

        [[nodiscard]] const char* c_str() const {
            return _data;
        }

        [[nodiscard]] const char* data() const {
            return _data;
        }

        char& operator[](const sizeT i) {
            return _data[i];
        }

        const char& operator[](const sizeT i) const {
            return _data[i];
        }

        Str& operator+=(const Str& other) {
            reserve(_size + other._size);
            std::strcat(_data, other._data);
            _size += other._size;
            _data[_size] = '\0';
            return *this;
        }

        Str& operator+=(const char* s) {
            const sizeT len = std::strlen(s);
            reserve(_size + len);
            std::strcat(_data, s);
            _size += len;
            _data[_size] = '\0'; 
            return *this;
        }

        Str& operator+=(const char c) {
            reserve(_size + 1);
            _data[_size++] = c;
            _data[_size]   = '\0';
            return *this;
        }

        bool operator==(const Str& other) const {
            return _size == other._size && std::strcmp(_data, other._data) == 0;
        }

        bool operator!=(const Str& other) const {
            return !(*this == other);
        }

        bool operator<(const Str& other) const {
            return std::strcmp(_data, other._data) < 0;
        }

        bool operator>(const Str& other) const {
            return other < *this;
        }

        bool operator<=(const Str& other) const {
            return !(*this > other);
        }

        bool operator>=(const Str& other) const {
            return !(*this < other);
        }

        void push_back(const char c) {
            *this += c;
        }

        Str& insert(sizeT pos, const char c) {
            if (pos > _size) {
                pos = _size;
            }
            reserve(_size + 1);
            std::memmove(_data + pos + 1, _data + pos, _size - pos + 1); 
            _data[pos] = c;
            _size++;
            return *this;
        }

        Str& insert(sizeT pos, const Str& o) {
            if (pos > _size) {
                pos = _size;
            }
            if (o.empty()) {
                return *this;
            }

            reserve(_size + o.size());
            // Move existing characters to make space
            std::memmove(_data + pos + o.size(), _data + pos, _size - pos + 1); // +1 for null terminator
            std::copy(o.begin(), o.end(), _data + pos);
            _size += o.size();
            return *this;
        }

        Str& erase(const sizeT pos, const sizeT count = 1) {
            if (pos >= _size) {
                return *this;
            }
            const sizeT actual_count = std::min(count, _size - pos);

            std::memmove(_data + pos, _data + pos + actual_count, _size - pos - actual_count + 1); // +1 for null terminator
            _size -= actual_count;
            return *this;
        }

        [[nodiscard]] sizeT find(const Str& s, const size_t pos = 0) const {
            if (pos >= _size || s.empty()) {
                return npos;
            }

            if (const char* result = std::strstr(_data + pos, s._data)) {
                return result - _data;
            }
            return npos;
        }

        void reserve(const sizeT n) {
            if (n >= _capacity) { // Use >= to handle resizing to current capacity if needed for null terminator
                sizeT new_capacity = std::max(n, _capacity == 0 ? 1 : _capacity * 2);
                char* new_data     = new char[new_capacity + 1]; // +1 for null terminator
                
                if (_data) {
                    std::strcpy(new_data, _data);
                    delete[] _data;
                }
                _data     = new_data;
                _capacity = new_capacity;
                _data[_size] = '\0'; // Ensure null termination
            }
        }
        void append(char* contents, sizeT size) {
            reserve(_size + size);
            std::memcpy(_data + _size, contents, size);
            _size += size;
            _data[_size] = '\0';
        }

    private:
        char* _data;
        sizeT _size;
        sizeT _capacity;
    };

    inline Str operator+(const Str& lhs, const Str& rhs) {
        Str result = lhs;
        result += rhs;
        return result;
    }

    inline Str operator+(const Str& lhs, const char* rhs) {
        Str result = lhs;
        result += rhs;
        return result;
    }

    inline Str operator+(const char* lhs, const Str& rhs) {
        auto result = Str(lhs);
        result += rhs;
        return result;
    }

    inline Str operator+(const Str& lhs, const char rhs) {
        Str result = lhs;
        result += rhs;
        return result;
    }

    inline Str operator+(const char lhs, const Str& rhs) {
        Str result;
        result += lhs;
        result += rhs;
        return result;
    }

    inline std::ostream& operator<<(std::ostream& os, const ds::Str& str) {
        os << str.c_str();
        return os;
    }

    inline Str operator+(const Str& a, const std::string& b) {
        return a + Str(b);
    }

    inline Str operator+(const std::string& a, const Str& b) {
        return Str(a) + b;
    }
} // namespace hahaha::common::ds

// Hash specialization for ds::Str
namespace std {
    template <>
    struct hash<hahaha::common::ds::Str> {
        size_t operator()(const hahaha::common::ds::Str& s) const noexcept {
            size_t hash     = 5381;
            const char* str = s.c_str();
            while (*str) {
                hash = ((hash << 5) + hash) + *str++;
            }
            return hash;
        }
    };
} // namespace std

#endif // STR_H

