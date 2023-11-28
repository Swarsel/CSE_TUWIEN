/// \file
/// \brief Testing ex21::timeit using two parameters of the same type

#include "Wrappers.hpp"

#include <cassert>  // assert
#include <cstddef>  // std::size_t
#include <iostream> // std::cout|endl

double sum2(const double& a, const double& b) { return a + b; }

int main() {

  using namespace ex21;

  double a = 1.;
  double b = 3.;

  { // normal function call
    [[maybe_unused]] auto res = sum2(a, b);
  }

  { // measuring runtime
    [[maybe_unused]] auto res = timeit(sum2, a, b);
  }

  return 0;
}
