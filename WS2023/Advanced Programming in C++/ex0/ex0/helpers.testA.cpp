/// \file
/// \brief Test ex0::print

#include "helpers.hpp"
#include <cassert> // assert

int main() {
  using namespace ex0;

  auto vec = Vector();
  
  vec.push_back(42);
  vec.push_back(24);
  vec.push_back(11);

  print(vec); // ctest implements assertions on the output of this test
  return 1;   // this output is neglected
}
