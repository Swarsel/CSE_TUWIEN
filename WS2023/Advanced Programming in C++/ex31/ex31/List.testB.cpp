/// \file
/// \brief Testing ex31::List::iterator

#include "List.hpp" // ex31::List

#include <cassert>     // assert
#include <type_traits> // std::is_same

int main() {

  using namespace ex31;

  using T = double;
  List<T> list{};
  list.push_back(1.0);
  list.push_back(2.0);
  list.push_back(3.0);

  [[maybe_unused]] auto it = list.begin(); // obtain iterator to begin

  static_assert(std::is_same<decltype(it), decltype(++it)>::value ||
                    std::is_same<decltype(it)&, decltype(++it)&>::value,
                "expect increment of iterator to return an iterator type");

  static_assert(std::is_same<decltype(it), decltype(--it)>::value ||
                    std::is_same<decltype(it)&, decltype(--it)&>::value,
                "expect decrement of iterator to return an iterator type");

  ++it;

  assert(*it == 2.0); // expect incremented iterator to point to next value

  --it;

  assert(*it == 1.0); // expect decremented iterator to point to prev value

  return 0;
}
