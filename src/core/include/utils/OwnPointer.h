//  Copyright (c) 2025-2026 Contributors of Hahaha(https://github.com/jason-is-debugging/Hahaha)
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
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace h3::core::utils {
/**
 * @brief A smart pointer implementation utilizing an Owner-Borrower semantics model.
 * * Unlike standard smart pointers, OwnPointer allows multiple instances to point to
 * the same resource, but enforces strict runtime ownership rules:
 * - Only **one** instance can act as the **Owner** at any given time.
 * - Multiple instances can act as **Borrowers** holding a shared reference.
 * - The resource is automatically deleted *only* when the Owner is destroyed or
 * reset.
 * - A `nullptr` managed pointer can never hold ownership status.
 * * @tparam T The underlying element type managed by this pointer.
 */
template <typename T> class OwnPointer {
  public:
    // Forward declaration for upcasting friendship
    template <typename U> friend class OwnPointer;

    /**
     * @brief Default constructor. Creates an invalid, non-owning pointer.
     */
    OwnPointer() : m_ptr(nullptr), m_isOwner(false), m_validity(nullptr) {}

    /**
     * @brief Construct an OwnPointer from nullptr.
     */
    // NOLINTNEXTLINE
    OwnPointer(std::nullptr_t) : m_ptr(nullptr), m_isOwner(false), m_validity(nullptr) {}

    /**
     * @brief Destructor. If Owner, invalidates the control block and deletes the
     * resource.
     */
    ~OwnPointer() {
        reset();
    }

    /**
     * @brief Explicitly transfers ownership out.
     * @throw std::runtime_error If called by a Borrower.
     */
    OwnPointer move() {
        if (!m_isOwner) {
            throw std::runtime_error("OwnPointer move called when not owned");
        }
        OwnPointer temp;
        temp.m_ptr = m_ptr;
        temp.m_isOwner = true;
        temp.m_validity = m_validity;

        // Strip ownership from this instance, transforming it into a borrower
        this->m_isOwner = false;
        return temp;
    }

    /**
     * @brief Copy constructor. Creates a Borrower pointing to the same resource and
     * validity block.
     */
    OwnPointer(const OwnPointer& other)
        : m_ptr(other.m_ptr), m_isOwner(false), m_validity(other.m_validity) {
    }

    /**
     * @brief Copy assignment. Turns this instance into a Borrower.
     */
    OwnPointer& operator=(const OwnPointer& other) {
        if (this != &other) {
            reset(); // Clean up current resource if we were the owner
            m_ptr = other.m_ptr;
            m_isOwner = false;
            m_validity = other.m_validity;
        }
        return *this;
    }

    OwnPointer& operator=(std::nullptr_t) {
        reset();
        m_ptr = nullptr;
        m_isOwner = false;
        m_validity = nullptr;
        return *this;
    }

    /**
     * @brief Move constructor. Transfers the exact state and role.
     */
    OwnPointer(OwnPointer&& other) noexcept
        : m_ptr(other.m_ptr), m_isOwner(other.m_isOwner),
          m_validity(other.m_validity) {
        other.m_ptr = nullptr;
        other.m_isOwner = false;
        other.m_validity = nullptr;
    }

    /**
     * @brief Move assignment. Transfers the exact state and role.
     */
    OwnPointer& operator=(OwnPointer&& other) noexcept {
        if (this != &other) {
            reset();
            m_ptr = other.m_ptr;
            m_isOwner = other.m_isOwner;
            m_validity = other.m_validity;

            other.m_ptr = nullptr;
            other.m_isOwner = false;
            other.m_validity = nullptr;
        }
        return *this;
    }

    /**
     * @brief Upcasting move constructor supporting polymorphism (Derived* to Base*).
     */
    template <typename U,
              typename = std::enable_if_t<std::is_convertible_v<U*, T*>
                                          && !std::is_same_v<U, T>>>
                                          // NOLINTNEXTLINE
    OwnPointer(OwnPointer<U>&& other) noexcept
        : m_ptr(other.m_ptr), m_isOwner(other.m_isOwner),
          m_validity(other.m_validity) {
        other.m_ptr = nullptr;
        other.m_isOwner = false;
        other.m_validity = nullptr;
    }

    /**
     * @brief Upcasting move assignment supporting polymorphism (Derived* to Base*).
     */
    template <typename U,
              typename = std::enable_if_t<std::is_convertible_v<U*, T*>
                                          && !std::is_same_v<U, T>>>
                                          // NOLINTNEXTLINE
    OwnPointer& operator=(OwnPointer<U>&& other) noexcept {
        if (static_cast<void*>(this) != static_cast<void*>(&other)) {
            reset();
            m_ptr = other.m_ptr;
            m_isOwner = other.m_isOwner;
            m_validity = other.m_validity;
            other.m_ptr = nullptr;
            other.m_isOwner = false;
            other.m_validity = nullptr;
        }
        return *this;
    }

    /**
     * @brief Upcasting copy constructor supporting polymorphism (Derived* to Base*).
     */
    template <typename U,
              typename = std::enable_if_t<std::is_convertible_v<U*, T*>
                                          && !std::is_same_v<U, T>>>
    OwnPointer(const OwnPointer<U>& other) noexcept
        : m_ptr(other.m_ptr), m_isOwner(false),
          m_validity(other.m_validity) {
    }

    /**
     * @brief Upcasting copy assignment supporting polymorphism (Derived* to Base*).
     */
    template <typename U,
              typename = std::enable_if_t<std::is_convertible_v<U*, T*>
                                          && !std::is_same_v<U, T>>>
    OwnPointer& operator=(const OwnPointer<U>& other) noexcept {
        reset();
        m_ptr = other.m_ptr;
        m_isOwner = false;
        m_validity = other.m_validity;
        return *this;
    }

    /**
     * @brief Explicitly spawn a non-owning Borrower.
     */
    OwnPointer borrow() const {
        return OwnPointer(*this); // Triggers copy constructor (creates borrower)
    }

    /**
     * @brief Resets the current smart pointer, releasing ownership if Owner.
     */
    void reset() {
        if (m_isOwner) {
            if (m_validity) {
                *m_validity = false;
            }
            delete m_ptr;
        }

        m_ptr = nullptr;
        m_isOwner = false;
        m_validity = nullptr;
    }

    /**
     * @brief Safety Check: Determines if the managed resource is safely accessible.
     */
    [[nodiscard]] bool is_valid() const {
        return m_ptr != nullptr && m_validity != nullptr && *m_validity;
    }

    [[nodiscard]] bool isOwner() const {
        return m_isOwner;
    }

    T& operator*() const {
        if (!is_valid())
            throw std::runtime_error(
                "Attempted to dereference an expired Borrower pointer");
        return *m_ptr;
    }

    T* operator->() const {
        if (!is_valid())
            throw std::runtime_error(
                "Attempted to access member of an expired Borrower pointer");
        return m_ptr;
    }

    explicit operator bool() const {
        return is_valid();
    }

  private:
    T* m_ptr;
    bool m_isOwner;
    bool* m_validity;

    /**
     * @brief Private constructor for internal use (make_own_ptr).
     */
    explicit OwnPointer(T* t)
        : m_ptr(t), m_isOwner(t != nullptr),
          m_validity(t != nullptr ? new bool(true) : nullptr) {}

    template <typename U, typename... Args>
    friend OwnPointer<U> make_own_ptr(Args&&... args);

    friend class OwnPointerTest;
};

template <typename T, typename... Args> OwnPointer<T> make_own_ptr(Args&&... args) {
    return OwnPointer<T>(new T(std::forward<Args>(args)...));
}

} // namespace h3::core::utils

#endif // HAHAHA_OWNPOINTER_H_E25F4B9F87244CA59C7BF66CCA03A943
