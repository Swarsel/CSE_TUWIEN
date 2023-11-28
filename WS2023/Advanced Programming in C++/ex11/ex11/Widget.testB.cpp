/// \file
/// \brief Test ex11::lref::init

#include "Widget.hpp"

#include <cassert> // assert
#include <utility> // std::move

int main() {

  using namespace ex11;

{
  [[maybe_unused]] auto vec = Vector(1000, 10);
  [[maybe_unused]] auto data = vec.data();
  [[maybe_unused]] auto copy = vec;
  [[maybe_unused]] auto widget = lref::init(vec); // passing an l-value

  assert(vec.data() == data);        // expect lvalue argument to keep owing its data/pointer
  assert(vec == copy);               // expect lvalue argument to keep its values
  assert(widget.vec.data() != data); // expect widget to own its own copy of the data
  assert(widget.vec == copy);        // expect widget to hold the same values
}

{
  [[maybe_unused]] auto vec = Vector(1000, 10);
  [[maybe_unused]] auto data = vec.data();
  [[maybe_unused]] auto copy = vec;
  [[maybe_unused]] auto widget = lref::init(std::move(vec)); // passing an r-value (explicit move)

  assert(vec.data() == data);        // expect lvalue argument to keep owing its data/pointer
  assert(vec == copy);               // expect lvalue argument to keep its values
  assert(widget.vec.data() != data); // expect widget to own its own copy of the data
  assert(widget.vec == copy);        // expect widget to hold the same values
}

  return 0;
}
