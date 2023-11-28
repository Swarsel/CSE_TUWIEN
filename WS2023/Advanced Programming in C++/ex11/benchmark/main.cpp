/// \file
/// \brief Benchmarking the three initialization functions value::init, lref::init, rref::init

#include <ex11/Widget.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include <numeric> // std::accumulate
#include <utility> // std::forward

using namespace ex11;

/// \brief Calls the function provided as first argument and forwards all subsequent arguments as
/// arguments to the function
/// \return Runtime of the function call in seconds
template <typename CALLABLE, typename... ARGS> double timeit(CALLABLE f, ARGS&&... args) {
  using Clock = std::chrono::steady_clock;
  using Duration = std::chrono::duration<double>;
  auto start = Clock::now();
  f(std::forward<ARGS>(args)...);
  auto stop = Clock::now();
  return Duration(stop - start).count();
}

int main() {

  std::size_t size = 10'000'000; // length for test vectors: 10M*sizeof(double) =~ 40MB
  std::size_t n = 10;            // iterations for averaging run time

  // prepare n vectors
  std::vector<Vector> inputs(n, Vector(size, 42));

  std::cout << std::setprecision(2);

  {
    auto walltime = 0.0;
    for (auto input : inputs) {
      walltime += timeit(value::init, input);
    }
    std::cout << "      value::init(Vector) passing l-value: " << walltime << "s" << std::endl;
  }

  {
    auto walltime = 0.0;
    for (auto input : inputs) {
      walltime += timeit(value::init, std::move(input));
    }
    std::cout << "      value::init(Vector) passing r-value: " << walltime << "s" << std::endl;
  }

  {
    auto walltime = 0.0;
    for (auto input : inputs) {
      walltime += timeit(lref::init, input);
    }
    std::cout << "lref::init(const Vector&) passing l-value: " << walltime << "s" << std::endl;
  }

  {
    auto walltime = 0.0;
    for (auto input : inputs) {
      walltime += timeit(lref::init, std::move(input));
    }
    std::cout << "lref::init(const Vector&) passing r-value: " << walltime << "s" << std::endl;
  }

  {
    auto walltime = 0.0;
    for (auto input : inputs) {
      walltime += timeit(rref::init, std::move(input));
    }
    std::cout << "     rref::init(Vector&&) passing r-value: " << walltime << "s" << std::endl;
  }

  return 0;
}
