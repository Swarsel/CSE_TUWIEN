/// \file
/// \brief Benchmarking the move/copy constructor and assignment

#include <ex13/List.hpp>

#include <chrono>   // std::chrono*
#include <iomanip>  // std::setprecision
#include <iostream> // std::cout|endl
#include <vector>   // std::vector

#include <utility> // std::forward

using namespace ex13;

/// \brief Calls the copy/move constructor with the provided arguments
/// \return Runtime of the function call in seconds
template <typename LIST> double timeit_ctor(LIST&& lst) {
  using Clock = std::chrono::steady_clock;
  using Duration = std::chrono::duration<double>;
  auto start = Clock::now();
  List res(std::forward<LIST>(lst));
  auto stop = Clock::now();
  return Duration(stop - start).count();
}

/// \brief Calls the copy/move assignment operator with the provided arguments
/// \return Runtime of the function call in seconds
template <typename LIST> double timeit_assign(LIST&& lst) {
  using Clock = std::chrono::steady_clock;
  using Duration = std::chrono::duration<double>;
  List other(lst);  
  auto start = Clock::now();
  other = std::forward<LIST>(lst);
  auto stop = Clock::now();
  return Duration(stop - start).count();
}


int main() {

  std::size_t size = 1'000'000; // length for test lists
  std::size_t n = 10;         // iterations

  // prepare n lists
  std::vector<List> inputs(n, List(size, 42.0));

  std::cout << std::setprecision(2);
  std::cout << std::scientific;

  {
    auto walltime = 0.0;
    for (auto input : inputs) {
      walltime += timeit_ctor(input);
    }
    std::cout << "           List::List(const List&):" << walltime << "s" << std::endl;
  }

  {
    auto walltime = 0.0;
    for (auto input : inputs) {
      walltime += timeit_ctor(std::move(input));
    }
    std::cout << "                List::List(List&&):" << walltime << "s" << std::endl;
  }

  {
    auto walltime = 0.0;
    for (auto input : inputs) {
      walltime += timeit_assign(input);
    }
    std::cout << "List::operator=(const List& other):" << walltime << "s" << std::endl;
  }

  {
    auto walltime = 0.0;
    for (auto input : inputs) {
      walltime += timeit_assign(std::move(input));
    }
    std::cout << "     List::operator=(List&& other):" << walltime << "s" << std::endl;
  }  


  return 0;
}
