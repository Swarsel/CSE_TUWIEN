/// \file
/// \brief Testing ex31::List::iterator

#include "List.hpp" // ex31::List

#include <cassert>     // assert
#include <type_traits> // std::is_same

int main() {

  using namespace ex31;
  using T = double;

  static_assert(std::is_same<decltype(List<T>{}.begin() != List<T>{}.end()), bool>::value,
                "expect comparison using != to return a bool");
  static_assert(std::is_same<decltype(List<T>{}.begin() == List<T>{}.end()), bool>::value,
                "expect comparison using == to return a bool");

  {
    List<T> list{};
    [[maybe_unused]] auto begin = list.begin();
    [[maybe_unused]] auto end = list.end();
    assert(begin == end); // expect begin and end to be equal for an empty list
  }

  {
    List<T> list{};
    list.push_back(42.0);
    [[maybe_unused]] auto begin = list.begin();
    [[maybe_unused]] auto end = list.end();
    assert(begin != end); // expect begin and end not to be equal for a non-empty list
  }

  return 0;
}
