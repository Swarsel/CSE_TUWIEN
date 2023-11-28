/// \file
/// \brief Test ex0::copy

#include "helpers.hpp"
#include <cassert> // assert

int main() {
  using namespace ex0;

  auto vec = Vector();
  
  vec.push_back(1);
  vec.push_back(2);
  vec.push_back(3);

  auto clone = copy(vec);

  assert((clone[0] == vec[0]));
  assert((clone[1] == vec[1]));
  assert((clone[2] == vec[2]));
  assert((clone.size() == vec.size()));

  return 0;
}
