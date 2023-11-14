/// \file
/// \brief Test ex12::Vector::Vector(Vector&&)

#include "Vector.hpp"

#include <cassert> // assert
#include <utility> // std::move

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
  Vector copy(vec);

  // invoke move constructor
  Vector vec2(std::move(vec));

  assert(data != vec.data());  // expect moved-from data holder to have changed
  assert(vec2.data() == data); // expect new data to be held by original data holder
  assert(vec2.size() == size); // expect new size to be indentical
  for (Vector::size_type i = 0; i < vec2.size(); ++i)
    assert(vec2[i] == copy[i]); // expect identical values at identical indices

  for (Vector::size_type i = 0; i < vec.size(); ++i)
    assert(&vec[i]); // expect a valid state of moved-from object

  return 0;
}
