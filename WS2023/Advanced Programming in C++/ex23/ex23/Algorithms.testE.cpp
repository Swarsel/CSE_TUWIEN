/// \file
/// \brief Testing ex23::count_fulfills_cond

#include "Algorithms.hpp" // ex23::count_fulfills_cond
#include "Reference.hpp"  // iue::count_fulfills_cond

#include <array>    // std::array
#include <cassert>  // assert
#include <iostream> // std::cout|endl
#include <list>     // std::list
#include <typeinfo> // typeid
#include <vector>   // std::vector

int main() {

  auto isEven = [](const int& val) -> bool { return !(val % 2); };

  {
    std::vector<int> ctr = {1, 2, 3, 4};

    auto res1 = ex23::count_fulfills_cond(ctr, isEven);
    auto res2 = iue::count_fulfills_cond(ctr, isEven);

    iue::print(std::cout, ctr);
    assert(res1 == res2); // expect identical effect of function calls ex23::* and iue::*
  }

  {
    std::array<int, 4> ctr = {1, 2, 3, 4};
    ex23::count_fulfills_cond(ctr, isEven); // expect to accept a std::array
  }

  {
    std::list<int> ctr = {1, 2, 3, 4};
    ex23::count_fulfills_cond(ctr, isEven); // expect to accept a std::list
  }

  return 0;
}
