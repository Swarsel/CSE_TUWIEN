/// \file
/// \brief Testing ex23::multiply_with_value

#include "Algorithms.hpp" // ex23::multiply_with_value
#include "Reference.hpp"  // iue::multiply_with_value

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

    double value = 42.0;
    iue::multiply_with_value(copy, value);
    ex23::multiply_with_value(ctr, value);

    iue::print(std::cout, ctr);
    iue::print(std::cout, copy);
    assert(copy == ctr); // expect identical effect of function calls ex23::* and iue::*
  }

  {
    std::array<double, 4> ctr = {1, 2, 3, 4};
    double value = 42.0;
    ex23::multiply_with_value(ctr, value); // expect to accept a std::array
  }

  {
    std::list<double> ctr = {1, 2, 3, 4};
    double value = 42.0;
    ex23::multiply_with_value(ctr, value); // expect to accept a std::list
  }

  return 0;
}
