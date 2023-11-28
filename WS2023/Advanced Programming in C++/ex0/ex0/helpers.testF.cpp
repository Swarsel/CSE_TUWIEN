/// \file
/// \brief Test ex0::fill_uniform_random

#include "helpers.hpp"
#include <cassert> // assert
#include <cmath>   // std::abs
#include <numeric> // std::accumulate

int main() {
  using namespace ex0;

  auto vec = Vector();

  fill_uniform_random(vec, 1'000, -1.0, 1.0);

  assert(vec.size() == 1'000);

  auto avg = std::accumulate(vec.begin(), vec.end(), 0.0) / vec.size();
  assert((std::abs(avg) < 1e-1));  // expect average around zero

  return 0;
}
