/// \file
/// \brief Testing ex31::List::iterator

#include "List.hpp" // ex31::List

#include <algorithm>   // std::stable_partition
#include <cassert>     // assert
#include <cstddef>     // std::ptrdiff_t
#include <iterator>    // std::bidirectional_iterator_tag
#include <type_traits> // std::is_same

int main() {

  using namespace ex31;
  using T = double;

  { // check nested types of iterator

    // List::iterator

    static_assert(std::is_same<List<T>::iterator::value_type, T>::value,
                  "expect iterator::value_type to be T");
    static_assert(
        std::is_same<List<T>::iterator::iterator_category, std::bidirectional_iterator_tag>::value,
        "expect iterator::iterator_category to be std::bidirectional_iterator_tag");
    static_assert(std::is_same<List<T>::iterator::difference_type, std::ptrdiff_t>::value,
                  "expect iterator::difference_type to be std::ptrdiff_t");
    static_assert(std::is_same<List<T>::iterator::pointer, T*>::value,
                  "expect iterator::pointer to be T*");
    static_assert(std::is_same<List<T>::iterator::reference, T&>::value,
                  "expect iterator::reference to be T&");

    // List::const_iterator

    static_assert(std::is_same<List<const T>::iterator::value_type, T>::value,
                  "expect iterator::value_type to be T");
    static_assert(std::is_same<List<const T>::iterator::iterator_category,
                               std::bidirectional_iterator_tag>::value,
                  "expect iterator::iterator_category to be std::bidirectional_iterator_tag");
    static_assert(std::is_same<List<const T>::iterator::difference_type, std::ptrdiff_t>::value,
                  "expect iterator::difference_type to be std::ptrdiff_t");
    static_assert(std::is_same<List<const T>::iterator::pointer, const T*>::value,
                  "expect iterator::pointer to be const T*");
    static_assert(std::is_same<List<const T>::iterator::reference, const T&>::value,
                  "expect iterator::reference to be const T&");
  }

  { // use stdlib functionality
    List<T> list;
    list.push_back(2.0);
    list.push_back(30.0);
    list.push_back(1.0);
    list.push_back(3.0);
    list.push_back(40.0);

    // use std::distance (requires difference_type)
    [[maybe_unused]] auto difference = std::distance(list.begin(), list.end());

    assert(difference == 5); // expect difference obtained via std::difference ist size of list

    // apply an stdlib algorithm relying on bidirectional_iterator_tag
    // elements > 10 to the front (while keeping relative order)
    [[maybe_unused]] auto it = std::stable_partition(
        list.begin(), list.end(), [](const List<T>::value_type& value) { return value > 10.0; });

    assert(*it == 2.0); // expect relative order
    ++it;
    assert(*it == 1.0); // expect relative order
    ++it;
    assert(*it == 3.0); // expect relative order
    ++it;
    assert(it == list.end()); // expect relative order
  }
  return 0;
}
