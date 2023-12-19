/// \file
/// \brief Testing ex23::sum_of_elements

#include "Algorithms.hpp" // ex23::sum_of_elements
#include "Reference.hpp"  // iue::sum_of_elements

#include <array>    // std::array
#include <cassert>  // assert
#include <iostream> // std::cout|endl
#include <list>     // std::list
#include <typeinfo> // typeid
#include <vector>   // std::vector

int main() {

  {
    std::vector<int> ctr = {1, 2, 3, 4};

    auto res1 = ex23::sum_of_elements(ctr);
    auto res2 = iue::sum_of_elements(ctr);

    iue::print(std::cout, ctr);
    assert(res1 == 10);
    assert(res1 == res2); // expect identical effect of function calls ex23::* and iue::*
  }

  {
    std::array<int, 4> ctr = {1, 2, 3, 4};
    ex23::sum_of_elements(ctr); // expect to accept a std::array
  }

  {
    std::list<int> ctr = {1, 2, 3, 4};
    ex23::sum_of_elements(ctr); // expect to accept a std::list
  }

  return 0;
}
