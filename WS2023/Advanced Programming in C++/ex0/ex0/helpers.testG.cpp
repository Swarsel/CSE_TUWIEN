/// \file
/// \brief Test ex::sort

#include "helpers.hpp"
#include <cassert> // assert

int main() {
  using namespace ex0;

  auto vec = Vector();
  
  vec.push_back(2);  
  vec.push_back(3);
  vec.push_back(1);
  vec.push_back(2);

  auto compare_func = [](const auto &a, const auto &b) -> bool { return a < b; };

  sort(vec, compare_func);

  assert(vec[0] == 1);
  assert(vec[1] == 2);
  assert(vec[2] == 2);  
  assert(vec[3] == 3);

  return 0;
}
