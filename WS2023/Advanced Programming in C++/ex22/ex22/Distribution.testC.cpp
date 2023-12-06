/// \file
/// \brief Testing ex22::Distribution using a std::vector<VecN<double, 3>>

#include "Distribution.hpp"
#include "VecN.hpp"

#include <algorithm>   // std::generate
#include <cassert>     // assert
#include <iostream>    // std::cout|endl
#include <random>      // std::mt19937|normal_distribution
#include <type_traits> // std::is_same
#include <vector>      // std::vector

int main() {

  using namespace ex22;

  using Vec3d = VecN<double, 3>;

  // prepare vector of VecN
  double mean = 10.0;
  double stddev = 1.0;
  std::vector<Vec3d> vec(10'000);
  {
    std::mt19937 engine(1);
    std::normal_distribution<double> normal(mean, stddev);
    auto gen = [&normal, &engine]() {
      return Vec3d{normal(engine), normal(engine), normal(engine)};
    };
    std::generate(vec.begin(), vec.end(), gen);
  }

  // expect construction from vector w/o explicit template parameter type
  auto dist = Distribution(vec);
  static_assert(std::is_same<Vec3d, Distribution<Vec3d>::value_type>::value);

  // expect member values to be calculated from samples
  std::cout << "mean=" << dist.mean << " stddev=" << dist.stddev << std::endl;
  for (auto i : {0, 1, 2}) {
    assert(std::abs(mean - dist.mean[i]) < mean * 0.05);
    assert(std::abs(stddev - dist.stddev[i]) < mean * 0.05);
  }

  return 0;
}
