/// \file
/// \brief Distribution class (header-only)


#pragma once

/// \todo add standard library headers as needed
#include "VecN.hpp"

namespace ex22 {

/// \todo Task 1: Implement a type "Distribution" with the following properties:
/// - class template with a single template type parameter T
/// - two public data member 'mean' and 'stddev' of type T
/// - public member typedef "value_type" which aliases the template parameter
/// - two constructors:
///   - Distribution(T mean, T stddev) which sets the two member variables
///   - Distribution(const std::vector<T> &data) which calculates
///     the two member variables from the samples in the container

/// \todo Task 2: Extend the construction mechanism:
/// - change the constructor "Distribution(const std::vector<T>&)" so it
///   accepts any sequential container from the standard library
/// - the template type parameter T is still deduced automatically

  template <typename T>
  struct Distribution {
    T mean;
    T stddev;
    using value_type = T;

    Distribution(T mean, T stddev): mean(mean), stddev(stddev) {}

    template <typename CONT>
    Distribution(const CONT &cont) {
      mean = 0;
      stddev = 0;
      int csize = cont.size();
      if (csize != 0) {
        for (T item : cont) {
            mean = mean + item;
          }
        mean = mean / csize;
        for (T item : cont) {
          stddev  = stddev + (item - mean) * (item - mean);
        }
        stddev = sqrt(stddev / csize);
      }
    }
  };
template <typename CONT>
Distribution(const CONT &cont) -> Distribution<typename CONT::value_type>;
} // namespace ex22
