/// \file
/// \brief Testing ex21::timeit using different member functions

#include "Wrappers.hpp"

#include <cstddef>  // std::size_t
#include <iostream> // std::cout|endl
#include <list>     // std::list
#include <vector>   // std::vector

using Vector = std::vector<double>;
using List = std::list<Vector>;

int main() {

  using namespace ex21;

  std::size_t n = 1'000'000;

  { // normal function calls

    List lst = {Vector(n, 0.0), Vector(n, 0.0), Vector(n, 0.0)};
    lst.push_back(Vector(n, 1.0));
    lst.push_front(Vector(n, 2.0));
    lst.emplace_back(n, 0.0);
  }

   // Note: the extra wrappers via lambdas below are used to let the compiler do the overload resolution 
   // before passing the resulting callable to the timeit function

  { // measuring runtime
    List lst = {Vector(n, 0.0), Vector(n, 0.0), Vector(n, 0.0)};
 
    auto push_back = [&lst](auto&&... args) -> auto {
      lst.push_back(std::forward<decltype(args)>(args)...);
    };

    timeit(push_back, Vector(n, 1.0));

    auto push_front = [&lst](auto&&... args) -> auto {
      lst.push_front(std::forward<decltype(args)>(args)...);
    };

    timeit(push_front, Vector(n, 2.0));

    auto emplace_back = [&lst](auto&&... args) -> auto {
      lst.emplace_back(std::forward<decltype(args)>(args)...);
    };

    timeit(emplace_back, n, 3.0);
  }

  return 0;
}
