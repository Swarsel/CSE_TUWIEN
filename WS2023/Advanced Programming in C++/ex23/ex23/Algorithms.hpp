/// \file
/// \brief Re-implementation of algo-functions from iue::* *WITHOUT* explicit loops

#include <cmath>   // std::abs
#include <cstddef> // std::size_t
#include <ostream> // std::ostream

#pragma once

/// \todo add standard library headers as needed
#include <algorithm>
#include <iterator>
#include <numeric>

namespace ex23 {

/// \todo Implement below functions identical to the functionality present in iue::* but without
/// using explicit loops

template <typename C> void set_to_value(C& ctr, typename C::value_type value) {
  /// \todo re-implement w/o explicit loops
  std::fill(ctr.begin(), ctr.end(), value);
}

template <typename C> void set_to_sequence(C& ctr, typename C::value_type start) {
  /// \todo re-implement w/o explicit loops
  std::generate(ctr.begin(), ctr.end(), [&start]() { return start++; });
}

template <typename C> void multiply_with_value(C& ctr, typename C::value_type value) {
  /// \todo re-implement w/o explicit loops
  std::for_each(ctr.begin(), ctr.end(),
                [value](typename C::value_type& element) { element *= value; });
}

template <typename C> void reverse_order(C& ctr) {
  /// \todo re-implement w/o explicit loops
  std::reverse(ctr.begin(), ctr.end());
}

template <typename C, typename UnaryPredicate>
std::size_t count_fulfills_cond(C& ctr, UnaryPredicate condition) {
  /// \todo re-implement w/o explicit loops
  return std::count_if(ctr.begin(), ctr.end(), condition);
}

template <typename C> bool first_n_equal(const C& a, const C& b, std::size_t n) {
  /// \todo re-implement w/o explicit loops
  return std::equal(a.begin(), std::next(a.begin(), n), b.begin(), std::next(b.begin(), n));
}

template <typename C> auto sum_of_elements(const C& ctr) {
  /// \todo re-implement w/o explicit loops
  return std::accumulate(ctr.begin(), ctr.end(), 0);
}

template <typename C> auto abssum_of_elements(C& ctr) {
  /// \todo re-implement w/o explicit loops
  return std::accumulate(ctr.begin(), ctr.end(), typename C::value_type(0),
                         [](typename C::value_type acc, const typename C::value_type& element) {
                           return acc + std::abs(element);
                         });
}

template <typename C> void print(std::ostream& os, const C& ctr) {
  /// \todo re-implement w/o explicit loops
  os << "[ ";
  std::copy(ctr.begin(), ctr.end(), std::ostream_iterator<typename C::value_type>(os, " "));
  os << "]" << std::endl;
}

} // namespace ex23
