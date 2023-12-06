/// \file
/// \brief Generic fixed size vector class (header-only)

#pragma once

#include <array>    // std::array
#include <cmath>    // std::sqrt
#include <iostream> // std::ostream

/// \brief Vector type supporting some operator overloads
template <class T, unsigned D = 3> class VecN {
private:
  std::array<T, D> x = {}; ///< holds the set of values
public:
  using size_type = decltype(D);
  using value_type = T;

  /// \brief Support all construction mechanisms of member 'x' by forwarding
  template <class... ARGS> VecN(ARGS&&... args) : x{static_cast<T>(std::forward<ARGS>(args))...} {}

  /// \brief Initialize all elements with a constant value
  template <class ARG> VecN(const ARG& init) { std::fill(x.begin(), x.end(), init); }

  /// \brief Element-wise addition
  VecN operator+(const VecN& other) const {
    VecN res;
    for (unsigned i = 0; i < D; ++i) {
      res[i] = x[i] + other.x[i];
    }
    return res;
  }

  /// \brief Element-wise subtraction
  VecN operator-(const VecN& other) const {
    VecN res;
    for (unsigned i = 0; i < D; ++i) {
      res[i] = x[i] - other.x[i];
    }
    return res;
  }

  /// \brief Subtraction of a scalar
  template <class U> VecN operator-(const T& value) const {
    VecN res;
    for (unsigned i = 0; i < D; ++i) {
      res[i] = x[i] - value;
    }
    return res;
  }

  /// Element-wise multiplication
  VecN operator*(const VecN& other) const {
    VecN res;
    for (unsigned i = 0; i < D; ++i) {
      res[i] = x[i] * other.x[i];
    }
    return res;
  }

  /// \brief Division by a scalar
  VecN operator/(const T& value) const {
    VecN res;
    for (unsigned i = 0; i < D; ++i) {
      res[i] = x[i] / value;
    }
    return res;
  }

  /// \brief Modifying indexed element access
  T& operator[](const size_type& index) { return x[index]; }

  /// \brief Non-modifying indexed element access
  const T& operator[](const size_type& index) const { return x[index]; }
};

/// \brief Element-wise square root
template <class T, unsigned D> VecN<T, D> sqrt(const VecN<T, D>& vec) {
  VecN<T, D> res;
  for (unsigned i = 0; i < D; ++i) {
    res[i] = std::sqrt(vec[i]);
  }
  return res;
}

/// \brief Overload for printing using <<
template <class T, unsigned D> std::ostream& operator<<(std::ostream& os, const VecN<T, D>& vec) {
  os << "[ ";
  for (unsigned i = 0; i < D - 1; ++i) {
    os << vec[i] << ", ";
  }
  os << vec[D - 1] << " ]";
  return os;
}