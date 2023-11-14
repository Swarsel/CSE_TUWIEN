/// \file
/// \brief Vector class declaration

#pragma once

#include <cstddef>

namespace ex12 {

class Vector {
public:
  using value_type = double;
  using size_type = std::size_t;

private:
  size_type _size = 0;         ///< current size
  value_type* _data = nullptr; ///< current data holder

public:
  /// \brief Default constructor
  Vector();

  /// \brief Custom constructor
  /// \param n number of elements to create
  /// \param init initial value for all elements
  Vector(size_type n, const value_type& init);

  /// \brief Copy constructor
  /// \param other object to be copied
  Vector(const Vector& other);

  /// \brief Move constructor
  /// \param other object to be moved from
  Vector(Vector&& other);

  /// \brief Destructor
  ~Vector();

  /// \brief Getter for _size
  size_type size() const;

  /// \brief Getter for _data
  value_type* data();

  /// \brief Getter for _data (read-only version)
  const value_type* data() const;

  /// \brief Access via []
  /// \param idx index of the element which is accessed
  value_type& operator[](size_type idx);

  /// \brief Access via [] (read-only version)
  /// \param idx index of the element which is accessed
  const value_type& operator[](size_type idx) const;
};

} // namespace ex12
