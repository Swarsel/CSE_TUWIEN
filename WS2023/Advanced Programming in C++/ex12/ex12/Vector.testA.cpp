/// \file
/// \brief Test ex12::Vector::~Vector()

#include "Vector.hpp"

#include <cassert>  // assert
#include <iostream> // std::cout|endl

int main() {

  using namespace ex12;

  {
    Vector vec(7, 42.0);
    std::cout << "vec.size()=" << vec.size() << std::endl;
    std::cout << "vec[0]=" << vec[0] << std::endl;
    std::cout << "vec[6]=" << vec[6] << std::endl;
  }
  // - vec goes out of scope here
  // - therefore the destructor is called.

  return 0;
}
