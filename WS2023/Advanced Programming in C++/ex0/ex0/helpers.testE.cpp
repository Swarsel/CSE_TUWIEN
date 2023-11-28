/// \file
/// \brief Test ex::swap

#include "helpers.hpp"
#include <cassert> // assert

int main() {
  using namespace ex0;

  auto a = Vector(1, 1);
  auto b = Vector(1, 2);

  // capture original memory addresses
  const auto a_data = a.data();
  const auto b_data = b.data();

  swap(a, b);

  // check values
  assert(a[0] == 2);
  assert(b[0] == 1);

  // check memory addresses
  assert(a_data == b.data());
  assert(b_data == a.data());

  return 0;
}
