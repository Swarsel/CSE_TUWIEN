/// \file
/// \brief Testing ex23::reverse_order

#include "Algorithms.hpp" // ex23::reverse_order
#include "Reference.hpp"  // iue::reverse_order

#include <array>    // std::array
#include <cassert>  // assert
#include <iostream> // std::cout|endl
#include <list>     // std::list
#include <typeinfo> // typeid
#include <vector>   // std::vector

int main() {

  { 
    std::vector<double> ctr = {1, 2, 3, 4};
    auto copy = ctr;

    iue::reverse_order(copy);
    ex23::reverse_order(ctr);

    iue::print(std::cout, ctr);
    iue::print(std::cout, copy);
    assert(copy == ctr); // expect identical effect of function calls ex23::* and iue::*
  }

  {
    std::array<double, 4> ctr = {1, 2, 3, 4};
    ex23::reverse_order(ctr); // expect to accept a std::array
  }

  {
    std::list<double> ctr = {1, 2, 3, 4};
    ex23::reverse_order(ctr); // expect to accept a std::list
  }

  return 0;
}
