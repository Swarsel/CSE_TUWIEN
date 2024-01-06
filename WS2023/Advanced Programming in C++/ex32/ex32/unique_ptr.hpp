/// \file
/// \brief didactic implementation of a std::unique_ptr (not fully compatible)
#include <utility>
#pragma once

/// \todo add standard library headers as needed

namespace ex32 {

/// \todo implement unique_ptr here

  template <typename T> void dd(T *ptr) { delete ptr; }

  template <typename T> class unique_ptr {
  public:
    using element_type = T;
    using pointer = T*;
    using deleter_type = void (*)(T*);

    unique_ptr(pointer ptr = nullptr, deleter_type d = dd<T>) : _ptr(ptr), _delete(d) {}

    unique_ptr(unique_ptr&& other) : _ptr(nullptr), _delete(dd<T>) {
      std::swap(_ptr, other._ptr);
      std::swap(_delete, other._delete);
    }

    unique_ptr& operator=(unique_ptr&& other) {
      if (*this != other) {
        std::swap(_ptr, other._ptr);
        std::swap(_delete, other._delete);
      }
      return *this;
    }

    unique_ptr& operator=(pointer ptr) {
      reset(ptr);
      return *this;
    }

    operator bool() const { return _ptr == nullptr; }

    // Interestingly, I do not need to disable manually
    // unique_ptr(const unique_ptr&) = delete;
    // unique_ptr& operator=(const unique_ptr&) = delete;

    ~unique_ptr() { _delete(_ptr); }

    pointer get() const { return _ptr; }

    void reset(pointer ptr = nullptr) {
      this->~unique_ptr();
      _ptr = ptr;
    }

    pointer release() {
      pointer ptr = _ptr;
      _ptr = nullptr;
      return ptr;
    }

    friend bool operator==(const unique_ptr& arg1, const unique_ptr& arg2) { return arg1._ptr == arg2._ptr; }
    friend bool operator!=(const unique_ptr& arg1, const unique_ptr& arg2) { return !(arg1._ptr == arg2._ptr); }
    friend bool operator==(const unique_ptr& arg1, std::nullptr_t arg2) { return arg1._ptr == arg2; }
    friend bool operator!=(const unique_ptr& arg1, std::nullptr_t arg2) { return !(arg1._ptr == arg2); }
    friend bool operator==(const std::nullptr_t arg1, unique_ptr& arg2) { return arg1 == arg2._ptr; }
    friend bool operator!=(const std::nullptr_t arg1, unique_ptr& arg2) { return !(arg1 == arg2._ptr); }

  private:
    pointer _ptr;
    deleter_type _delete;
  };
} // namespace ex32
