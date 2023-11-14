/// \file
/// \brief Vector class (definitions)

#include "Vector.hpp"
#include <algorithm>

/// \todo Add standard library headers if needed

namespace ex12 {

    Vector::Vector() {}

    Vector::Vector(size_type n, const value_type& init) : _size(n), _data(new Vector::value_type[n]) {
        for (size_type i = 0; i < _size; ++i) {
            _data[i] = init;
        }
    }

    /// \todo Implement copy constructor
    Vector::Vector(const Vector& other)  : _size(other._size), _data(new value_type[other._size]) {
        std::copy(other._data, other._data + _size, _data);
    }

    /// \todo Implement move constructor
    Vector::Vector(Vector&& other) : _size(0), _data(nullptr) {
        std::swap(other._size, _size);
        std::swap(other._data, _data);
    }

    /// \todo Implement desctuctor
    Vector::~Vector() {
        delete[] _data;
    }

    Vector::size_type Vector::size() const { return _size; }

    Vector::value_type* Vector::data() { return _data; }
    const Vector::value_type* Vector::data() const { return _data; }

    Vector::value_type& Vector::operator[](size_type idx) { return _data[idx]; }
    const Vector::value_type& Vector::operator[](size_type idx) const { return _data[idx]; }

} // namespace ex12
