/// \file
/// \brief Testing ex22::Distribution using a std::list<double>

#include "Distribution.hpp"

#include <algorithm>   // std::generate
#include <cassert>     // assert
#include <deque>       // std::deque
#include <iostream>    // std::cout|endl
#include <random>      // std::mt19937|normal_distribution
#include <type_traits> // std::is_same

int main() {

  using namespace ex22;

  // prepare list
  double mean = 10.0;
  double stddev = 1.0;
  std::deque<double> deque(10'000);
  {
    std::mt19937 engine(1);
    std::normal_distribution<double> normal(mean, stddev);
    auto gen = [&normal, &engine]() { return normal(engine); };
    std::generate(deque.begin(), deque.end(), gen);
  }

  // expect construction from deque w/o explicit template parameter type
  auto dist = Distribution(deque);
  static_assert(std::is_same<double, Distribution<double>::value_type>::value);

  // expect member values to be calculated from samples
  std::cout << "mean=" << dist.mean << " stddev=" << dist.stddev << std::endl;
  assert(std::abs(mean - dist.mean) < mean * 0.05);
  assert(std::abs(stddev - dist.stddev) < mean * 0.05);

  return 0;
}