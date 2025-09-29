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
#include <algorithm>
#include <cstring>
#include <sstream>

#include "common/defines/h3defs.h"

namespace hahaha::common::ds {
  class Str {
public:
    static const sizeT npos = static_cast<sizeT>(-1);

    Str() : _data(new char[1]), _size(0), _capacity(0) { _data[0] = '\0'; }

    explicit Str(const char *c) {
      _data = new char[std::strlen(c) + 1];
      _size = std::strlen(c);
      _capacity = _size;
      std::strcpy(_data, c);
    }

    Str(const Str &other) : _size(other._size), _capacity(other._capacity) {
      _data = new char[_size];
      std::copy_n(other._data, _size, _data);
    }

    Str(Str &&other) noexcept : _size(other._size), _capacity(other._capacity) {
      _data = other._data;

      other._data = nullptr;
      other._size = 0;
      other._capacity = 0;
    }
    Str(char *begin, const char *end) {
      _size = end - begin;
      _capacity = end - begin;

      _data = new char[_size];
      std::copy_n(begin, _size, _data);
    }
    explicit Str(const std::string &string) {
      _data = new char[string.size() + 1];
      _size = string.size();
      std::strcpy(_data, string.c_str());
      _capacity = string.size();
      _data[_size] = '\0';
    }

    explicit Str(const std::stringstream &ss) {
      _data = new char[ss.str().size() + 1];
      _size = ss.str().size();
      std::strcpy(_data, ss.str().c_str());
      _capacity = ss.str().size();
      _data[_size] = '\0';
    }
    explicit Str(const std::ostringstream &oss) {
      _data = new char[oss.str().size() + 1];
      _size = oss.str().size();
      std::strcpy(_data, oss.str().c_str());
      _capacity = oss.str().size();
      _data[_size] = '\0';
    }
    ~Str() { delete[] _data; }

    Str &operator=(const Str &other) {
      if (this != &other) {
        delete[] _data;
        _size = other._size;
        _capacity = other._capacity;
        _data = new char[_size];
        std::copy_n(other._data, _size, _data);
      }
      return *this;
    }

    Str &operator=(Str &&other) noexcept {
      if (this != &other) {
        delete[] _data;
        _data = other._data;
        _size = other._size;
        _capacity = other._capacity;
        other._data = nullptr;
        other._size = 0;
        other._capacity = 0;
      }
      return *this;
    }

    [[nodiscard]] sizeT size() const { return _size; }
    [[nodiscard]] sizeT length() const { return _size; }
    [[nodiscard]] sizeT capacity() const { return _capacity; }
    [[nodiscard]] bool empty() const { return _size == 0; }

    [[nodiscard]] char at(const sizeT i) const { return _data[i]; }
    [[nodiscard]] char *begin() const { return &_data[0]; }
    [[nodiscard]] char *end() const { return &_data[_size]; }

    void clear() {
      _size = 0;
      _data[0] = '\0';
    }

    void resize(const sizeT n, const char c = '\0') {
      if (n > _size) {
        reserve(n);
        std::fill(_data + _size, _data + n, c);
      }
      _size = n;
      _data[_size] = '\0';
    }

    [[nodiscard]] const char *c_str() const { return _data; }
    [[nodiscard]] const char *data() const { return _data; }

    char &operator[](const sizeT i) { return _data[i]; }
    const char &operator[](const sizeT i) const { return _data[i]; }

    Str &operator+=(const Str &other) {
      reserve(_size + other._size);
      std::strcat(_data, other._data);
      _size += other._size;
      return *this;
    }

    Str &operator+=(const char *s) {
      const sizeT len = std::strlen(s);
      reserve(_size + len);
      std::strcat(_data, s);
      _size += len;
      return *this;
    }

    Str &operator+=(const char c) {
      reserve(_size + 1);
      _data[_size++] = c;
      _data[_size] = '\0';
      return *this;
    }

    bool operator==(const Str &other) const { return _size == other._size && std::strcmp(_data, other._data) == 0; }

    bool operator!=(const Str &other) const { return !(*this == other); }

    bool operator<(const Str &other) const { return std::strcmp(_data, other._data) < 0; }

    bool operator>(const Str &other) const { return other < *this; }

    bool operator<=(const Str &other) const { return !(*this > other); }

    bool operator>=(const Str &other) const { return !(*this < other); }

    void push_back(const char c) { *this += c; }

    Str &insert(sizeT pos, const char c) {
      if (pos > _size)
        pos = _size;
      reserve(_size + 1);
      for (sizeT i = _size; i > pos; --i) {
        _data[i] = _data[i - 1];
      }
      _data[pos] = c;
      _size++;
      _data[_size] = '\0';
      return *this;
    }

    Str &insert(sizeT pos, const Str &o) {
      if (pos > _size)
        pos = _size;
      if (o.size() == 0)
        return *this;

      reserve(_size + o.size());
      for (sizeT i = _size + o.size() - 1; i > pos; --i) {
        _data[i] = _data[i - o.size()];
      }
      for (sizeT i = 0; i < o.size(); ++i) {
        _data[pos + i] = o[i];
      }
      _size += o.size();
      _data[_size] = '\0';
      return *this;
    }

    Str &erase(const sizeT pos, const sizeT count = 1) {
      if (pos > _size)
        return *this;
      const sizeT cnt = std::min(count, _size - pos);

      for (int i = 0; i < cnt; ++i) {
        _data[pos + i] = _data[pos + cnt + i];
      }
      _size -= cnt;
      _data[_size] = '\0';
      return *this;
    }

    [[nodiscard]] sizeT find(const Str &s, const size_t pos = 0) const {
      if (pos >= _size)
        return npos;

      if (const char *result = std::strstr(_data + pos, s._data)) {
        return result - _data;
      }
      return npos;
    }


    void reserve(const sizeT n) {
      if (n > _capacity) {
        const auto new_data = new char[n + 1];
        std::copy_n(_data, n, new_data);
        delete[] _data;
        _data = new_data;
        _capacity = n;
      }
    }
    void append(char *contents, sizeT size);

private:
    char *_data;
    sizeT _size;
    sizeT _capacity;
  };

  inline Str operator+(const Str &lhs, const Str &rhs) {
    Str result = lhs;
    result += rhs;
    return result;
  }

  inline Str operator+(const Str &lhs, const char *rhs) {
    Str result = lhs;
    result += rhs;
    return result;
  }

  inline Str operator+(const char *lhs, const Str &rhs) {
    auto result = Str(lhs);
    result += rhs;
    return result;
  }

  inline Str operator+(const Str &lhs, const char rhs) {
    Str result = lhs;
    result += rhs;
    return result;
  }

  inline Str operator+(const char lhs, const Str &rhs) {
    Str result;
    result += lhs;
    result += rhs;
    return result;
  }

  inline std::ostream &operator<<(std::ostream &os, const ds::Str &str) {
    os << str.c_str();
    return os;
  }

  inline Str operator+(const Str &a, const std::string &b) { return a + Str(b); }

  inline Str operator+(const std::string &a, const Str &b) { return Str(a) + b; }
} // namespace hahaha::common::ds

// Hash specialization for ds::Str
namespace std {
  template<>
  struct hash<hahaha::common::ds::Str> {
    size_t operator()(const hahaha::common::ds::Str &s) const noexcept {
      size_t hash = 5381;
      const char *str = s.c_str();
      while (*str) {
        hash = ((hash << 5) + hash) + *str++;
      }
      return hash;
    }
  };
} // namespace std

#endif // STR_H
