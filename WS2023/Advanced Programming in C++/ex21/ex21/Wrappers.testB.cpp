/// \file
/// \brief Testing ex21::timeit using a varying number of parameters of same type

#include "Wrappers.hpp"

void inc(double& a) { ++a; }
double sum2(const double& a, const double& b) { return a + b; }
double sum3(const double& a, const double& b, const double& c) { return a + b + c; }
double sum4(const double& a, const double& b, const double& c, const double& d) {
  return a + b + c + d;
}

int main() {

  using namespace ex21;

  double a = 1.;
  double b = 2.;
  double c = 3.;
  double d = 4.;

  { // normal function calls
    inc(a);
    [[maybe_unused]] auto res1 = sum2(a, b);
    [[maybe_unused]] auto res2 = sum3(a, b, c);
    [[maybe_unused]] auto res3 = sum4(a, b, c, d);
  }
  { // measuring runtime
    timeit(inc, a);
    [[maybe_unused]] auto res1 = timeit(sum2, a, b);
    [[maybe_unused]] auto res2 = timeit(sum3, a, b, c);
    [[maybe_unused]] auto res3 = timeit(sum4, a, b, c, d);
  }

  return 0;
}
