/// \file
/// \brief Testing ex31::List::iterator

#include "List.hpp" // ex31::List

#include <cassert>     // assert

int main() {

  using namespace ex31;

  using T = double;
  List<T> list{};
  list.push_back(1.0);
  list.push_back(2.0);
  list.push_back(3.0);

  [[maybe_unused]] auto it = list.end(); // obtain iterator to begin

  --it;

  assert(*it == 3.0); // expect iterator to end to point to last element after decrement

  return 0;
}
