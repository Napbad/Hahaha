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
#include <type_traits>
#include <utility>

namespace h3::core::utils {
/**
 * @brief A smart pointer implementation utilizing an Owner-Borrower semantics model.
 * * Unlike standard smart pointers, OwnPointer allows multiple instances to point to the
 * same resource, but enforces strict runtime ownership rules:
 * - Only **one** instance can act as the **Owner** at any given time.
 * - Multiple instances can act as **Borrowers** holding a shared reference.
 * - The resource is automatically deleted *only* when the Owner is destroyed or reset.
 * - A `nullptr` managed pointer can never hold ownership status.
 * * @tparam T The underlying element type managed by this pointer.
 */
template <typename T> class OwnPointer {
  public:
    // Constructor - takes ownership of the pointer
    explicit OwnPointer(T* t = nullptr) : m_ptr(t), m_isOwner(t != nullptr) {
    }

    OwnPointer& operator=(std::nullptr_t) {
        if (m_isOwner && m_ptr) {
            delete m_ptr;
        }
        m_ptr = nullptr;
        m_isOwner = false;
        return *this;
    }

    // Destructor - only delete if we're the owner
    ~OwnPointer() {
        if (m_isOwner && m_ptr) {
            delete m_ptr;
            m_ptr = nullptr;
        }
    }

    OwnPointer move() {
        if (!m_isOwner) {
            throw std::runtime_error("OwnPointer move called when not owned");
        }
        OwnPointer temp(m_ptr, true);
        this->m_isOwner = false;
        this->m_ptr = nullptr;
        return temp;
    }

    // Copy constructor - become a borrower
    OwnPointer(const OwnPointer& other) : m_ptr(other.m_ptr), m_isOwner(false) {
    }

    // Copy assignment operator - become a borrower
    OwnPointer& operator=(const OwnPointer& other) {
        if (this != &other) {
            // If we were the owner, clean up our resource
            if (m_isOwner && m_ptr) {
                delete m_ptr;
            }
            m_ptr = other.m_ptr;
            m_isOwner = false; // Always become borrower on copy
        }
        return *this;
    }

    OwnPointer borrow() {
        return OwnPointer(m_ptr, false);
    }

    // Move constructor - transfer ownership
    OwnPointer(OwnPointer&& other) noexcept
        : m_ptr(other.m_ptr), m_isOwner(other.m_isOwner) {
        other.m_ptr = nullptr;
        other.m_isOwner = false;
    }

    // Move assignment operator - transfer ownership
    OwnPointer& operator=(OwnPointer&& other) noexcept {
        if (this != &other) {
            // If we were the owner, clean up our resource
            if (m_isOwner && m_ptr) {
                delete m_ptr;
            }
            m_ptr = other.m_ptr;
            m_isOwner = other.m_isOwner;
            other.m_ptr = nullptr;
            other.m_isOwner = false;
        }
        return *this;
    }

    // Upcast move from OwnPointer<Derived> when Derived* converts to T*
    template <typename U,
              typename = std::enable_if_t<std::is_convertible_v<U*, T*>
                                          && !std::is_same_v<U, T>>>
    explicit OwnPointer(OwnPointer<U>&& other) noexcept
        : m_ptr(other.release()), m_isOwner(m_ptr != nullptr) {
    }

    template <typename U,
              typename = std::enable_if_t<std::is_convertible_v<U*, T*>
                                          && !std::is_same_v<U, T>>>
    OwnPointer& operator=(OwnPointer<U>&& other) noexcept {
        if (m_isOwner && m_ptr) {
            delete m_ptr;
        }
        m_ptr = other.release();
        m_isOwner = m_ptr != nullptr;
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
    [[nodiscard]] bool isOwner() const {
        return m_isOwner;
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
        m_isOwner = false;
        return temp;
    }

    // Reset with a new pointer
    void reset(T* t = nullptr) {
        if (m_isOwner && m_ptr) {
            delete m_ptr;
        }
        m_ptr = t;
        m_isOwner = (t != nullptr);
    }

    // Explicit conversion to bool
    explicit operator bool() const {
        return m_ptr != nullptr;
    }

  private:
    T* m_ptr;
    bool m_isOwner;

    OwnPointer(T* ptr, const bool isOwner) : m_ptr(ptr), m_isOwner(isOwner) {
    }
};

template <typename T, typename... Args> OwnPointer<T> make_own_ptr(Args&&... args) {
    return OwnPointer<T>(new T(std::forward<Args>(args)...));
}

} // namespace h3::core::utils

#endif // HAHAHA_OWNPOINTER_H_E25F4B9F87244CA59C7BF66CCA03A943
