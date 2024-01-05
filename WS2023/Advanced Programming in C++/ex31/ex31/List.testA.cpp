/// \file
/// \brief Testing ex31::List::iterator

#include "List.hpp" // ex31::List

#include <cassert>     // assert
#include <type_traits> // std::is_same

int main() {

  using namespace ex31;

  using T = double;

  List<T> list;
  list.push_back(3.1415);

  [[maybe_unused]] auto it = list.begin(); // obtain iterator to begin

  static_assert(
      std::is_same<T&, decltype(*it)>::value); // expect dereferencing returning a reference to T

  assert(*it == 3.1415); // expect iterator to point to pushed values

  *it = 3.0;

  assert(*it == 3.0); // expect value to have changed due to value assignment

  return 0;
}
