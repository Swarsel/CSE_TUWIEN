// compile: clang++ -g -fsanitize=address -std=c++17 assoc.cpp && ./a.out

#include <iostream>

int main() { // associative math
  float a = -500000000;
  float b = 500000000;
  float c = 1;
  std::cout << "a + (b + c) is equal to " << a + (b + c) << std::endl;
  std::cout << "(a + b) + c is equal to " << (a + b) + c << std::endl;
}