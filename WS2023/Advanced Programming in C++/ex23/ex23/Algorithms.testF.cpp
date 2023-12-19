/// \file
/// \brief Testing ex23::first_n_equal

#include "Algorithms.hpp" // ex23::first_n_equal
#include "Reference.hpp"  // iue::first_n_equal

#include <array>    // std::array
#include <cassert>  // assert
#include <iostream> // std::cout|endl
#include <list>     // std::list
#include <typeinfo> // typeid
#include <vector>   // std::vector

int main() {

  {
    std::vector<int> a = {1, 2, 3, 4};
    std::vector<int> b = {1, 2, 0, 3, 4};

    iue::print(std::cout, a);
    iue::print(std::cout, b);

    {
      auto res1 = ex23::first_n_equal(a, b, 2);
      auto res2 = iue::first_n_equal(a, b, 2);

      assert(res1 == true); // expect true as first two elements match
      assert(res1 == res2); // expect identical effect of function calls ex23::* and iue::*
    }

    {
      auto res1 = ex23::first_n_equal(a, b, 3);
      auto res2 = iue::first_n_equal(a, b, 3);

      assert(res1 == false); // expect false as first three elements do not match
      assert(res1 == res2);  // expect identical effect of function calls ex23::* and iue::*
    }
  }

  {
    std::array<int, 4> a = {1, 2, 3, 4};
    std::array<int, 4> b = {1, 2, 0, 3};
    ex23::first_n_equal(a, b, 2); // expect to accept a std::array
  }

  {
    std::list<int> a = {1, 2, 3, 4};
    std::list<int> b = {1, 2, 0, 3, 4};
    ex23::first_n_equal(a, b, 2); // expect to accept a std::list
  }

  return 0;
}
