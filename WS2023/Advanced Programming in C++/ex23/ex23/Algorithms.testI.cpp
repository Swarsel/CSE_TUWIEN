/// \file
/// \brief Testing ex23::print

#include "Algorithms.hpp" // ex23::print
#include "Reference.hpp"  // iue::print

#include <array>    // std::array
#include <cassert>  // assert
#include <iostream> // std::cout|endl
#include <list>     // std::list
#include <sstream>  // std::ostringstream
#include <typeinfo> // typeid
#include <vector>   // std::vector

int main() {

  {
    std::vector<int> ctr = {-1, -2, 3, 4};

    std::ostringstream res1;
    std::ostringstream res2;

    ex23::print(res1, ctr);
    iue::print(res2, ctr);

    std::cout << res1.str() << std::endl;
    std::cout << res2.str() << std::endl;
    
    assert(res1.str() == res2.str()); // expect identical effect of function calls ex23::* and iue::*
  }

  {
    std::ostringstream oss;
    std::array<int, 4> ctr = {-1, -2, 3, 4};
    ex23::print(oss, ctr); // expect to accept a std::array
  }

  {
    std::ostringstream oss;
    std::list<int> ctr = {-1, -2, 3, 4};
    ex23::print(oss, ctr); // expect to accept a std::list
  }

  return 0;
}
