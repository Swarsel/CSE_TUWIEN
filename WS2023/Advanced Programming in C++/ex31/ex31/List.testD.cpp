/// \file
/// \brief Testing ex31::List::iterator

#include "List.hpp" // ex31::List

#include <cassert>     // assert
#include <type_traits> // std::is_same

int main() {

  using namespace ex31;

  struct Widget {
    double member;
  };

  using T = Widget;
  List<T> list{};
  list.push_back(Widget{1.0});
  list.push_back(Widget{2.0});
  list.push_back(Widget{3.0});

  [[maybe_unused]] auto it = list.begin(); // obtain iterator to begin

  static_assert(std::is_same<decltype(T{}.member), decltype(it->member)>::value,
                "expect operator-> to return type of respective member of stored value");

  ++it;

  [[maybe_unused]] auto value = it->member;

  assert(it->member == 2.0); // expect access via operator-> to access member of stored value

  it->member = value + 1.0;

  assert(it->member == 3.0); // expect access via operator-> to access member of stored value

  return 0;
}
