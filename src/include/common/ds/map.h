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

#ifndef MAP_H
#define MAP_H
#include <map>


#include "str.h"
#include "vec.h"

namespace hiahiahia::ds {
  template <typename Key, typename T, typename Compare = std::less<Key>, typename Allocator = std::allocator<std::pair<const Key, T>>>
  class Map {
  public:
    // Type definitions
    using key_type = Key;
    using mapped_type = T;
    using value_type = std::pair<const Key, T>;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using key_compare = Compare;
    using allocator_type = Allocator;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = typename std::allocator_traits<Allocator>::pointer;
    using const_pointer = typename std::allocator_traits<Allocator>::const_pointer;
    // Iterator types (required for resolving the symbol)
    using iterator = typename std::map<Key, T, Compare, Allocator>::iterator;
    using const_iterator = typename std::map<Key, T, Compare, Allocator>::const_iterator;

    // Constructor/destructor
    Map() : _data() {}
    ~Map() = default;

    // Iterator-related methods
    iterator begin() { return _data.begin(); }
    const_iterator begin() const { return _data.begin(); }
    iterator end() { return _data.end(); }
    const_iterator end() const { return _data.end(); }

    // Capacity-related methods
    [[nodiscard]] size_type size() const { return _data.size(); }
    [[nodiscard]] bool empty() const { return _data.empty(); }

    // Element access
    T& operator[](const Key& key) {
      return _data[key];
    }

    // Insertion/removal operations
    std::pair<iterator, bool> insert(const value_type& value) {
      return _data.insert(value);
    }

    size_type erase(const Key& key) {
      return _data.erase(key);
    }

    void clear() { _data.clear(); }

  private:
    // Internal storage using std::map (for interface demonstration only)
    std::map<Key, T, Compare, Allocator> _data;
  };

} // namespace hiahiahia::ds

#endif // MAP_H

