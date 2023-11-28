/// \file
/// \brief Test ex11::rref::init

#include "Widget.hpp"

#include <cassert> // assert
#include <utility> // std::move

int main() {

  using namespace ex11;

  [[maybe_unused]] auto vec = Vector(1000, 10);
  [[maybe_unused]] auto data = vec.data();
  [[maybe_unused]] auto copy = vec;
  [[maybe_unused]] auto widget = rref::init(std::move(vec)); // passing an r-value (explicit move)

  assert(vec.data() == nullptr);     // expect rvalue argument to be empty after moving-from
  assert(widget.vec.data() == data); // expect widget to now own the original data/pointer
  assert(widget.vec == copy);        // expect widget to have the original values

  return 0;
}
