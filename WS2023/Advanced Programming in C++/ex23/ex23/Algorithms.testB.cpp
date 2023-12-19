/// \file
/// \brief Testing ex23::set_to_sequence

#include "Algorithms.hpp" // ex23::set_to_sequence
#include "Reference.hpp"  // iue::set_to_sequence

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

    double start = 42.0;
    iue::set_to_sequence(copy, start);
    ex23::set_to_sequence(ctr, start);

    iue::print(std::cout, ctr);
    iue::print(std::cout, copy);
    assert(copy == ctr); // expect identical effect of function calls ex23::* and iue::*
  }

  {
    std::array<double, 4> ctr = {1, 2, 3, 4};
    double start = 42.0;
    ex23::set_to_sequence(ctr, start); // expect to accept a std::array
  }

  {
    std::list<double> ctr = {1, 2, 3, 4};
    double start = 42.0;
    ex23::set_to_sequence(ctr, start); // expect to accept a std::list
  }

  return 0;
}
