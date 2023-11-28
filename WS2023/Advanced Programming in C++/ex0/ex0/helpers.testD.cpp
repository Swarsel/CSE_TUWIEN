/// \file
/// \brief Test ex0::concat

#include "helpers.hpp"
#include <cassert> // assert

int main() {
  using namespace ex0;

  auto a = Vector(1, 1);
  auto b = Vector(2, 2);

  auto c = concat(a, b);

  assert((c[0] == a[0]));
  assert((c[1] == b[0]));
  assert((c[2] == b[1]));  

  return 0;
}
