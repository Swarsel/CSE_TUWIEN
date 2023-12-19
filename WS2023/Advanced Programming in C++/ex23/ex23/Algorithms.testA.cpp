/// \file
/// \brief Testing ex23::set_to_value

#include "Algorithms.hpp" // ex23::set_to_value
#include "Reference.hpp"  // iue::set_to_value

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

    double init = 42.0;
    iue::set_to_value(copy, init);
    ex23::set_to_value(ctr, init);

    iue::print(std::cout, ctr);
    iue::print(std::cout, copy);
    assert(copy == ctr); // expect identical effect of function calls ex23::* and iue::*
  }

  {
    std::array<double, 4> ctr = {1, 2, 3, 4};
    double init = 42.0;
    ex23::set_to_value(ctr, init); // expect to accept a std::array
  }

  {
    std::list<double> ctr = {1, 2, 3, 4};
    double init = 42.0;
    ex23::set_to_value(ctr, init); // expect to accept a std::list
  }

  return 0;
}
