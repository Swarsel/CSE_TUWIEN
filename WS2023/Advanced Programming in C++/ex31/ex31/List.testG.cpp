/// \file
/// \brief Testing ex31::List::iterator

#include "List.hpp" // ex31::List

#include <cassert>     // assert
#include <type_traits> // std::is_same

int main() {

  using namespace ex31;
  using T = double;

  List<T> list;
  list.push_back(1.0);
  list.push_back(2.0);
  list.push_back(3.0);

  { // check if cbegin and cend return iterators providing const access

    static_assert(std::is_same<decltype(*list.cbegin()), const T&>::value,
                  "expect cbegin() to return iterator with const access");
    static_assert(std::is_same<decltype(*(--list.cend())), const T&>::value,
                  "expect cend() to return iterator with const access");
  }

  { // check if begin and end return iterators providing non-const access
    static_assert(std::is_same<decltype(List<T>{}.begin())::value_type, T>::value,
                  "expect begin() to return iterator with non-const access");
    static_assert(std::is_same<decltype(List<T>{}.end())::value_type, T>::value,
                  "expect to return iterator with non-const access");
  }

  { // check if begin and end obtained from a const list result in iterators
    // with const access
    [[maybe_unused]] const List<T>& clist = list;

    static_assert(std::is_same<decltype(*clist.begin()), const T&>::value,
                  "expect begin() to return iterator with const access for const list");
    static_assert(std::is_same<decltype(*(--clist.end())), const T&>::value,
                  "expect end() to return iterator with const access for const list");
  }
  return 0;
}
