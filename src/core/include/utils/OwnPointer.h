//  Copyright (c) 2025-2026 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//       https://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//  Contributors:
//  Napbad (napbad.sen@gmail.com) (https://github.com/Napbad)
//

//
// Created by napbad on 5/11/26.
//

#ifndef HAHAHA_OWNPOINTER_H_E25F4B9F87244CA59C7BF66CCA03A943
#define HAHAHA_OWNPOINTER_H_E25F4B9F87244CA59C7BF66CCA03A943
#include <utility>

namespace h3::core::utils {
/**
 * this smart pointer is used to identify the Owner and Borrower
 * @tparam T this is the exact element type it holds
 */
template<typename T>
class OwnPointer {
public:
    // Constructor - takes ownership of the pointer
    explicit OwnPointer(T* t = nullptr) : m_ptr(t), m_is_owner(t != nullptr) {}

    // Destructor - only delete if we're the owner
    ~OwnPointer() {
        if (m_is_owner && m_ptr) {
            delete m_ptr;
            m_ptr = nullptr;
        }
    }

    // Copy constructor - become a borrower
    OwnPointer(const OwnPointer& other) : m_ptr(other.m_ptr), m_is_owner(false) {}

    // Copy assignment operator - become a borrower
    OwnPointer& operator=(const OwnPointer& other) {
        if (this != &other) {
            // If we were the owner, clean up our resource
            if (m_is_owner && m_ptr) {
                delete m_ptr;
            }
            m_ptr = other.m_ptr;
            m_is_owner = false;  // Always become borrower on copy
        }
        return *this;
    }

    // Move constructor - transfer ownership
    OwnPointer(OwnPointer&& other) noexcept : m_ptr(other.m_ptr), m_is_owner(other.m_is_owner) {
        other.m_ptr = nullptr;
        other.m_is_owner = false;
    }

    // Move assignment operator - transfer ownership
    OwnPointer& operator=(OwnPointer&& other) noexcept {
        if (this != &other) {
            // If we were the owner, clean up our resource
            if (m_is_owner && m_ptr) {
                delete m_ptr;
            }
            m_ptr = other.m_ptr;
            m_is_owner = other.m_is_owner;
            other.m_ptr = nullptr;
            other.m_is_owner = false;
        }
        return *this;
    }

    // Equality comparison
    bool operator==(const OwnPointer& other) const {
        return m_ptr == other.m_ptr;
    }

    // Inequality comparison
    bool operator!=(const OwnPointer& other) const {
        return m_ptr != other.m_ptr;
    }

    // Dereference operator
    T& operator*() const {
        return *m_ptr;
    }

    // Arrow operator
    T* operator->() const {
        return m_ptr;
    }

    // Check if this is the owner
    bool is_owner() const {
        return m_is_owner;
    }

    // Get the raw pointer (const version)
    const T* get() const {
        return m_ptr;
    }

    // Get the raw pointer (non-const version)
    T* get() {
        return m_ptr;
    }

    // Release ownership without deleting
    T* release() {
        T* temp = m_ptr;
        m_ptr = nullptr;
        m_is_owner = false;
        return temp;
    }

    // Reset with a new pointer
    void reset(T* t = nullptr) {
        if (m_is_owner && m_ptr) {
            delete m_ptr;
        }
        m_ptr = t;
        m_is_owner = (t != nullptr);
    }
    
    // Explicit conversion to bool
    explicit operator bool() const {
        return m_ptr != nullptr;
    }


private:
    T *m_ptr;
    bool m_is_owner;
};

template<typename T, typename... Args>
OwnPointer<T> make_own_ptr(Args&&... args) {
    return OwnPointer<T>(new T(std::forward<Args>(args)...));
}

}

#endif //HAHAHA_OWNPOINTER_H_E25F4B9F87244CA59C7BF66CCA03A943
