/// \file
/// \brief Test ex13::List(const List&other)

#include "List.hpp"

#include <cassert> // assert
#include <cstddef> // std::size_t

int main() {

  using namespace ex13;

  List lst(7, 42.0);
  lst.push_front(1.0);

  [[maybe_unused]] const auto data = lst.data();
  [[maybe_unused]] const auto size = lst.size();

  List lst2(lst); // copy ctor

  assert(data == lst.data()); // expect unchanged state of original list
  assert(size == lst.size()); // expect unchanged state of original list

  assert(data != lst2.data()); // expect new list to hold own data
  assert(size == lst2.size()); // expect identical sizes

  auto iter = lst.data();
  auto iter2 = lst2.data();
  std::size_t counter = 0;
  while (iter != nullptr && iter2 != nullptr) {
    assert(iter->value == iter2->value); // expect identical values in both lists
    iter = iter->next;
    iter2 = iter2->next;
    ++counter;
  }

  assert(counter == size); // expect sizes from iterating and member _size to match

  return 0;
}
