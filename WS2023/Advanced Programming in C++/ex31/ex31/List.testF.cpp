/// \file
/// \brief Testing ex31::List::iterator

#include "List.hpp" // ex31::List

#include <cassert> // assert

int main() {

  using namespace ex31;
  using T = double;

  { // explicit for loop
    List<T> list{};
    for (const auto& value : {1, 2, 3, 4, 5}) {
      list.push_back(value);
    }
    [[maybe_unused]] int count = 1;
    for (auto it = list.begin(); it != list.end(); ++it) {
      assert(*it == count++); // expect accesses values to be in order
    }
  }
  { // range-based for loop
    List<T> list{};
    for (const auto& value : {1, 2, 3, 4, 5}) {
      list.push_back(value);
    }
    [[maybe_unused]] int count = 1;
    for ([[maybe_unused]] const auto& item : list) {
      assert(item == count++); // expect accesses values to be in order
    }
  }
  { // while loop
    List<T> list{};
    for (const auto& value : {1, 2, 3, 4, 5}) {
      list.push_back(value);
    }
    auto it = list.begin();
    [[maybe_unused]] int count = 1;
    while (it != list.end()) {
      assert(*it == count++); // expect accesses values to be in order
      ++it;
    }
  }
  return 0;
}
