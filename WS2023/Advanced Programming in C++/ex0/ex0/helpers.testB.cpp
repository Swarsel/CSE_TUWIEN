/// \file
/// \brief Test ex0::reset

#include "helpers.hpp"
#include <cassert> // assert

int main() {
  using namespace ex0;

  auto vec = Vector();
  
  vec.push_back(1);
  vec.push_back(2);
  vec.push_back(3);

  reset(vec);

  assert(vec.empty());

  return 0;
}
