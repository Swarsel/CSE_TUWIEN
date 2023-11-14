/// \file
/// \brief Test ex12::Vector::Vector(const Vector&)

#include "Vector.hpp"

#include <cassert> // assert
#include <cmath>   // std::abs
#include <cstddef>

int main() {

  using namespace ex12;

  Vector vec(4, 0.0);
  vec[0] = 0.0;
  vec[1] = 1.0;
  vec[2] = 2.0;
  vec[3] = 3.0;

  // record original state
  [[maybe_unused]] auto size = vec.size();
  [[maybe_unused]] auto data = vec.data();

  // invoke copy constructor
  Vector vec2(vec);

  assert(data == vec.data());        // expect original data holder to be unchanged
  assert(size == vec.size());        // expect original size to be unchanged
  assert(vec2.size() == vec.size()); // expect new size to be indentical
  assert(vec2.data() != vec.data()); // expect copied data to no alias with the old data
  [[maybe_unused]] std::ptrdiff_t diff = vec2.data() - vec.data();
  assert(diff >= static_cast<ptrdiff_t>(4) || diff <= static_cast<ptrdiff_t>(-4));

  // alias with the old data
  for (Vector::size_type i = 0; i < vec2.size(); ++i)
    assert(vec2[i] == vec[i]); // expect identical values at identical indices

  return 0;
}
