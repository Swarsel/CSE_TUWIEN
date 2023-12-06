/// \file
/// \brief Testing ex22::Distribution nested types and member access

#include "Distribution.hpp"

#include <cassert>     // assert
#include <type_traits> // std::is_same

int main() {

  using namespace ex22;

  // expect nested type 'value_type' reflecting template parameter
  static_assert(std::is_same<double, Distribution<double>::value_type>::value);
  static_assert(std::is_same<float, Distribution<float>::value_type>::value);
  static_assert(std::is_same<long, Distribution<long>::value_type>::value);

  // check if constructor for member initialization works
  double mean = 42.;
  double stddev = 3.14159265;

  // expect automatic deduction from ctor argument type
  static_assert(std::is_same<decltype(Distribution(mean, stddev))::value_type, double>::value);

  // expect
  Distribution dist(mean, stddev);

  { // expect public member variable access
    [[maybe_unused]] auto mean = dist.mean;
    [[maybe_unused]] auto stddev = dist.stddev;
  }

  // expect data member values being set from ctor arguments
  assert(mean == dist.mean);
  assert(stddev == dist.stddev);

  return 0;
}
