/// \file
/// \brief Wrapper function for timing (header-only)

#pragma once

#include <chrono>     // std::chrono::*
#include <functional> // std:invoke
#include <iomanip>    // std::setprecision
#include <iostream>   // std::cout|endl
#include <ratio>      // std::milli
#include <utility>    // std::forward

namespace ex21 {

/// \brief Measures the time starting with its construction and prints the elapsed time on
/// destruction
struct AutoTimer {
  using Clock = std::chrono::high_resolution_clock;
  using Timepoint = std::chrono::time_point<Clock>;
  using Duration = std::chrono::duration<double, std::milli>;
  Timepoint start = Clock::now();
  AutoTimer() = default;
  AutoTimer(const AutoTimer&) = delete;
  AutoTimer(AutoTimer&&) = delete;
  AutoTimer& operator=(const AutoTimer&) = delete;
  AutoTimer& operator=(AutoTimer&&) = delete;
  ~AutoTimer() {
    Timepoint stop = Clock::now();
    Duration walltime = stop - start;
    std::cout << std::setprecision(2);
    std::cout << std::scientific;
    std::cout << walltime.count() << "ms" << std::endl;
  }
};

/// \brief Wraps a call to function/callable with two copyable-constructable
/// arguments of the same type
/// \tparam CALLABLE the type/signature of the function to be called
/// \tparam ARG the type of the arguments to the function to be called
/// \param callable the object/function to be called
/// \param arg1 the first argument to be forwarded to the call
/// \param arg2 the second argument to be forwarded to the call
/// \return the value returned from the call
/// \todo Adapt this function (including its template parameters and parameters)
/// to be able to wrap arbitrary call signatures
template <typename CALLABLE, typename ... ARGS>
decltype(auto) timeit(CALLABLE callable, ARGS &&... args ) {
  auto timer = AutoTimer();
  return callable(std::forward<ARGS>(args)...);
}

} // namespace ex21
