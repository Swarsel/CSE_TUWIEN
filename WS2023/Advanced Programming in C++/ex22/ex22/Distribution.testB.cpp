/// \file
/// \brief Testing ex22::Distribution using a std::vector<double>

#include "Distribution.hpp"

#include <algorithm>   // std::generate
#include <cassert>     // assert
#include <iostream>    // std::cout|endl
#include <random>      // std::mt19937|normal_distribution
#include <type_traits> // std::is_same
#include <vector>      // std::vector

int main() {

  using namespace ex22;

  // prepare vector
  double mean = 10.0;
  double stddev = 1.0;
  std::vector<double> vec(10'000);
  {
    std::mt19937 engine(1);
    std::normal_distribution<double> normal(mean, stddev);
    auto gen = [&normal, &engine]() { return normal(engine); };
    std::generate(vec.begin(), vec.end(), gen);
  }

  // expect construction from vector w/o explicit template parameter type
  auto dist = Distribution(vec);
  static_assert(std::is_same<double, Distribution<double>::value_type>::value);

  // expect member values to be calculated from samples
  std::cout << "mean=" << dist.mean << " stddev=" << dist.stddev << std::endl;
  assert(std::abs(mean - dist.mean) < mean * 0.05);
  assert(std::abs(stddev - dist.stddev) < mean * 0.05);

  return 0;
}
