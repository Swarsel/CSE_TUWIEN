/// \file
/// \brief Test ex13::List(List&& other)

#include "List.hpp"

#include <cassert> // assert
#include <cstddef> // std::size_t
#include <memory>  // std::move

int main() {

  using namespace ex13;

  List lst(7, 42.0);
  lst.push_front(1.0);

  [[maybe_unused]] const auto data = lst.data();
  [[maybe_unused]] const auto size = lst.size();
  [[maybe_unused]] const auto copy = lst;

  List lst2(std::move(lst)); // move ctor

  {
    assert(data != lst.data()); // expect changed ownership of moved-from list
    std::size_t counter = 0;
    auto iter = lst.data();
    while (iter != nullptr) {
      iter = iter->next;
      ++counter;
    }
    assert(counter == lst.size()); // expect valid state of moved-from list
  }

  {
    assert(data == lst2.data()); // expect new list to hold original data
    assert(size == lst2.size()); // expect new list to have original size

    auto iter = lst2.data();
    auto iter2 = copy.data();
    std::size_t counter = 0;
    while (iter != nullptr && iter2 != nullptr) {
      assert(iter->value == iter2->value); // expect original values in moved list
      iter = iter->next;
      iter2 = iter2->next;
      ++counter;
    }
    assert(counter == lst2.size()); // expect sizes from iterating and member _size to match
  }

  return 0;
}
