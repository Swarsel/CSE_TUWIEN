/// \file
/// \brief Benchmarking the move/copy constructor

#include <ex12/Vector.hpp>

#include <chrono>   // std::chrono*
#include <iomanip>  // std::setprecision
#include <iostream> // std::cout|endl
#include <vector>   // std::vector

#include <utility> // std::forward

using namespace ex12;

/// \brief Calls the function provided as first argument and forwards all subsequent arguments as
/// arguments to the function
/// \return Runtime of the function call in seconds
template <typename... ARGS> double timeit(ARGS&&... args) {
  using Clock = std::chrono::steady_clock;
  using Duration = std::chrono::duration<double>;
  auto start = Clock::now();
  Vector res(std::forward<ARGS>(args)...);
  auto stop = Clock::now();
  return Duration(stop - start).count();
}

int main() {

  std::size_t size = 10'000'000; // length for test vectors: 10M*sizeof(double) =~ 40MB
  std::size_t n = 10;            // iterations

  // prepare n vectors
  std::vector<Vector> inputs(n, Vector(size, 42.0));

  std::cout << std::setprecision(2);

  {
    auto walltime = 0.0;
    for (auto input : inputs) {
      walltime += timeit(input);
    }
    std::cout << "Vector::Vector(const Vector&): " << walltime << "s" << std::endl;
  }

  {
    auto walltime = 0.0;
    for (auto input : inputs) {
      walltime += timeit(std::move(input));
    }
    std::cout << "Vector::Vector(Vector&&):      " << walltime << "s" << std::endl;
  }
  return 0;
}
